import glob
import json
import math
import os
import pathlib
import time
from functools import reduce
from itertools import chain
from collections import OrderedDict

import cv2
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar

from scipy.ndimage import binary_fill_holes
from skimage import morphology, segmentation

import torch
import torch.nn.functional as F
from skimage import morphology

from misc.utils import (
    get_bounding_box,
    log_info,
    rm_n_mkdir,
)
from misc.wsi_handler import get_file_handler

from . import base


def _infer_patch(patch, net, mask=None, scaled_patch_shape=None,
                 block_id=None,
                 offset=0,
                 num_patches=None,
                 nr_types=None,
                 available_gpus_ids=None):

    if mask is not None:
        roi = tuple([slice(sp * b, sp * (b + 1), 1)
                     for sp, b in zip(scaled_patch_shape, block_id)])

        if np.sum(mask[roi]) == 0:
            H, W = patch.shape[-2:]
            h = H - offset
            w = W - offset
            return np.zeros((h, w, 3 if nr_types is None else 4))

    net_idx = (block_id[0] * num_patches[1] + block_id[1]) % len(available_gpus_ids)
    with torch.no_grad():
        patch_img = torch.from_numpy(patch[:, :, 0]).contiguous()
        # Send the patch to the corresponding GPU (id any)
        pred_map = net[net_idx](
            patch_img.to(
                'cuda:%i' % available_gpus_ids[net_idx] if torch.cuda.is_available() else 'cpu'))

        pred_map = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()]
             for k, v in pred_map.items()]
        )
        pred_map["np"] = F.softmax(pred_map["np"], dim=-1)[..., 1:]
        if "tp" in pred_map:
            type_map = F.softmax(pred_map["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_map["tp"] = type_map

        pred_map = torch.cat(list(pred_map.values()), -1).cpu().numpy()
        pred_map = pred_map[0]

    return pred_map


def _process_instance(pred_map, pred_inst, inst_id, tl_pos, br_pos, merge_edges,
                      nr_types=None,
                      ambiguous_size=128,
                      check_sides_first=False):
    H, W = pred_inst.shape

    inst_map_full = pred_inst == inst_id

    rmin, rmax, cmin, cmax = get_bounding_box(inst_map_full)

    inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
    inst_map = inst_map_full[inst_bbox[0][0]:inst_bbox[1][0],
                             inst_bbox[0][1]:inst_bbox[1][1]]

    inst_map = inst_map.astype(np.uint8)
    inst_moment = cv2.moments(inst_map)
    inst_contour = cv2.findContours(
        inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # * opencv protocol format may break
    inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))

    # < 3 points dont make a contour, so skip, likely artifact too
    # as the contours obtained via approximation => too small or sthg
    if inst_contour.shape[0] < 3 or len(inst_contour.shape) != 2:
        return None, None

    inst_centroid = [
        (inst_moment["m10"] / inst_moment["m00"]),
        (inst_moment["m01"] / inst_moment["m00"]),
    ]

    inst_centroid = np.array(inst_centroid)
    inst_contour[:, 0] += inst_bbox[0][1] + tl_pos[1]  # X
    inst_contour[:, 1] += inst_bbox[0][0] + tl_pos[0]  # Y
    inst_centroid[0] += inst_bbox[0][1] + tl_pos[1]  # X
    inst_centroid[1] += inst_bbox[0][0] + tl_pos[0]  # Y

    if nr_types is not None:
        inst_type_crop = pred_map[inst_bbox[0][0]:inst_bbox[1][0],
                                  inst_bbox[0][1]:inst_bbox[1][1], 0]

        inst_type = inst_type_crop[np.nonzero(inst_map)]
        type_list, type_pixels = np.unique(inst_type,
                                           return_counts=True)

        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]

        if inst_type == 0:  # ! pick the 2nd most dominant if exist
            if len(type_list) > 1:
                inst_type = type_list[1][0]

        type_dict = {v[0]: v[1] for v in type_list}
        type_prob = type_dict[inst_type] / (np.sum(inst_map) + 1.0e-6)
        inst_type = int(inst_type)

    else:
        type_prob = None
        inst_type = None

    bbox_tl_x = inst_bbox[0][1]
    bbox_tl_y = inst_bbox[0][0]
    bbox_br_x = inst_bbox[1][1]
    bbox_br_y = inst_bbox[1][0]

    inst_bbox[0][1] += tl_pos[1] # X
    inst_bbox[0][0] += tl_pos[0] # Y
    inst_bbox[1][1] += tl_pos[1] # X
    inst_bbox[1][0] += tl_pos[0] # Y

    inst_key = '%i,%i,%i,%i,%i' % (*tl_pos, *br_pos, inst_id)

    info_dict_key = 'inner'
    inst_pred_map = None

    if ambiguous_size > 0:
        # For objects detected at boundaries, save its corresponding
        # prediction map. This cab be used in _fix_hor_boundaries and
        # _fix_ver_boundaries to resolve ambiguities at chunk
        # boundaries.
        inst_pred_map = np.copy(pred_map[bbox_tl_y:bbox_br_y,
                                         bbox_tl_x:bbox_br_x])
        inst_pred_map = np.concatenate((inst_pred_map,
                                        inst_map[..., np.newaxis]),
                                        axis=2)

        if check_sides_first:
            if (bbox_tl_x < ambiguous_size
                and not merge_edges["left"]):
                info_dict_key = 'left'
            elif (bbox_br_x >= W - ambiguous_size
                and not merge_edges["right"]):
                info_dict_key = 'right'
            elif (bbox_tl_y < ambiguous_size
                and not merge_edges["top"]):
                info_dict_key = 'top'
            elif (bbox_br_y >= H - ambiguous_size
                and not merge_edges["bottom"]):
                info_dict_key = 'bottom'
            else:
                inst_pred_map = None
        else:
            if (bbox_tl_y < ambiguous_size
                and not merge_edges["top"]):
                info_dict_key = 'top'
            elif (bbox_br_y >= H - ambiguous_size
                and not merge_edges["bottom"]):
                info_dict_key = 'bottom'
            elif (bbox_tl_x < ambiguous_size
                and not merge_edges["left"]):
                info_dict_key = 'left'
            elif (bbox_br_x >= W - ambiguous_size
                and not merge_edges["right"]):
                info_dict_key = 'right'
            else:
                inst_pred_map = None

    inst_info_dict = {
            "bbox": inst_bbox.tolist(),
            "centroid": inst_centroid.tolist(),
            "contour": inst_contour.tolist(),
            "type_prob": type_prob,
            "type": inst_type,
            "pred_map": inst_pred_map
            }

    return info_dict_key, {inst_key: inst_info_dict}


def _post_process(pred_map, mask=None, scaled_patch_shape=None,
                  block_id=None, nr_types=None, offset=None,
                  tile_shape=None,
                  num_tiles=None,
                  check_sides_first=True,
                  ambiguous_size=128,
                  merge_edges=True):

    H, W = pred_map.shape[:2]

    tl_pos = (block_id[0] * tile_shape[0] + offset[0],
              block_id[1] * tile_shape[1] + offset[1])
    br_pos = (tl_pos[0] + H, tl_pos[1] + W)

    merge_edges = dict(
        top=block_id[0] == 0 and merge_edges,
        bottom=block_id[0] == num_tiles[0] - 1 and merge_edges,
        left=block_id[1] == 0 and merge_edges,
        right=block_id[1] == num_tiles[1] - 1 and merge_edges)

    # Add a correction offset to the ambiguous prediction maps according to the
    # priority (sides first or not)
    corr_ver = 0 if check_sides_first else ambiguous_size
    corr_hor = ambiguous_size if check_sides_first else 0

    info_dicts = dict(
        tl_pos=tl_pos,
        br_pos=br_pos,
        inner={},
        left={},
        right={},
        top={},
        bottom={},
        bound_pred_maps=dict(
            left=None,
            left_tl_pos=(tl_pos[0] + corr_ver, tl_pos[1]),
            left_br_pos=(br_pos[0] - corr_ver, tl_pos[1] + ambiguous_size),
            right=None,
            right_tl_pos=(tl_pos[0] + corr_ver, br_pos[1] - ambiguous_size),
            right_br_pos=(br_pos[0] - corr_ver, br_pos[1]),
            top=None,
            top_tl_pos=(tl_pos[0], tl_pos[1] + corr_hor),
            top_br_pos=(tl_pos[0] + ambiguous_size, br_pos[1] - corr_hor),
            bottom=None,
            bottom_tl_pos=(br_pos[0] - ambiguous_size, tl_pos[1] + corr_hor),
            bottom_br_pos=(br_pos[0], br_pos[1] - corr_hor)))

    if mask is not None:
        roi = tuple([slice(sp * b, sp * (b + 1), 1)
                     for sp, b in zip(scaled_patch_shape, block_id)])
        if np.sum(mask[roi]) == 0:
            return np.array([[info_dicts]])

    if nr_types is not None:
        pred_inst = pred_map[..., 1:]

    else:
        pred_inst = pred_map

    # If the prediction map is empty/blank return a zero array
    blb_raw = pred_inst[..., 0]
    h_dir_raw = pred_inst[..., 1]
    v_dir_raw = pred_inst[..., 2]

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.uint8)

    num_cc, blb = cv2.connectedComponents(blb)
    if num_cc > 1:
        blb = morphology.remove_small_objects(blb, min_size=10)

    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    num_cc, marker = cv2.connectedComponents(marker)
    if num_cc > 1:
        marker = morphology.remove_small_objects(marker, min_size=10)

    pred_inst = segmentation.watershed(dist, markers=marker, mask=blb)
    pred_inst = pred_inst.astype(np.int32)

    inst_id_list = np.unique(pred_inst)[1:]
    for inst_id in inst_id_list:
        (info_dict_key,
         info_dict_inst) = _process_instance(pred_map, pred_inst, inst_id,
                                             tl_pos,
                                             br_pos,
                                             merge_edges,
                                             nr_types,
                                             ambiguous_size,
                                             check_sides_first)

        if info_dict_key is not None:
            info_dicts[info_dict_key].update(info_dict_inst)

    # Delete the current instance from the prediction map
    pred_map[(*np.nonzero(pred_inst),
             slice(0, 2 if nr_types is None else 3, 1))] = 0

    # Move the remaining ambiguous prediction map for fixing chunk
    # boundaries in the next step.
    if ambiguous_size > 0 and merge_edges:
        if not merge_edges["left"]:
            info_dicts['bound_pred_maps']['left'] = np.copy(
                pred_map[corr_ver:H - corr_ver, :ambiguous_size])
        if not merge_edges["right"] and ambiguous_size > 0:
            info_dicts['bound_pred_maps']['right'] = np.copy(
                pred_map[corr_ver:H - corr_ver, -ambiguous_size:])
        if not merge_edges["top"] and ambiguous_size > 0:
            info_dicts['bound_pred_maps']['top'] = np.copy(
                pred_map[:ambiguous_size, corr_hor:W - corr_hor])
        if not merge_edges["bottom"] and ambiguous_size > 0:
            info_dicts['bound_pred_maps']['bottom'] = np.copy(
                pred_map[-ambiguous_size:, corr_hor:W - corr_hor])

    inst_info_arr = np.array([[info_dicts]])

    return inst_info_arr


def _fix_hor_boundaries(inst_info_arr, block_id=None, nr_types=None):
    info_dicts_A = inst_info_arr[0, 0]

    if inst_info_arr.shape[1] == 1:
        # There is no chunk next to chunk A because it is the right edge of the 
        # image.
        info_dicts_B = dict(
            left={},
            bound_pred_maps=dict(
                left=None,
                left_br_pos=info_dicts_A['bound_pred_maps']['right_br_pos']))
    else:
        info_dicts_B = inst_info_arr[0, 1]

    # Check if any of the two chunks is empty before continue
    if (info_dicts_A['bound_pred_maps']['right'] is not None
      and info_dicts_B['bound_pred_maps']['left'] is not None):
        pred_map = np.hstack(
            (info_dicts_A['bound_pred_maps']['right'],
             info_dicts_B['bound_pred_maps']['left']))

        pred_map_tl = info_dicts_A['bound_pred_maps']['right_tl_pos']
        pred_map_br = info_dicts_B['bound_pred_maps']['left_br_pos']

    elif info_dicts_A['bound_pred_maps']['right'] is not None:
        pred_map = np.copy(info_dicts_A['bound_pred_maps']['right'])

        pred_map_tl = info_dicts_A['bound_pred_maps']['right_tl_pos']
        pred_map_br = info_dicts_A['bound_pred_maps']['right_br_pos']

    elif info_dicts_B['bound_pred_maps']['left'] is not None:
        pred_map = np.copy(info_dicts_B['bound_pred_maps']['left'])

        pred_map_tl = info_dicts_B['bound_pred_maps']['left_tl_pos']
        pred_map_br = info_dicts_B['bound_pred_maps']['left_br_pos']

    else:
        # If both chunks are empty, return chunk A as it is
        return inst_info_arr[:, :1]

    min_y = min(chain(map(lambda d: d['bbox'][0][0],
                          info_dicts_A['right'].values()),
                      map(lambda d: d['bbox'][0][0],
                          info_dicts_B['left'].values()),
                      [pred_map_tl[0]]))

    min_x = min(chain(map(lambda d: d['bbox'][0][1],
                          info_dicts_A['right'].values()),
                      map(lambda d: d['bbox'][0][1],
                          info_dicts_B['left'].values()),
                      [pred_map_tl[1]]))

    max_y = max(chain(map(lambda d: d['bbox'][1][0],
                          info_dicts_A['right'].values()),
                      map(lambda d: d['bbox'][1][0],
                          info_dicts_B['left'].values()),
                      [pred_map_br[0]]))

    max_x = max(chain(map(lambda d: d['bbox'][1][1],
                          info_dicts_A['right'].values()),
                      map(lambda d: d['bbox'][1][1],
                          info_dicts_B['left'].values()),
                      [pred_map_br[1]]))

    pad_y_A = pred_map_tl[0] - min_y
    pad_x_A = pred_map_tl[1] - min_x
    pad_y_B = max_y - pred_map_br[0]
    pad_x_B = max_x - pred_map_br[1]

    pred_map = np.pad(pred_map, ((pad_y_A, pad_y_B),
                                 (pad_x_A, pad_x_B),
                                 (0, 0)))

    for k, d in info_dicts_A['right'].items():
        inst_y, inst_x = np.nonzero(d['pred_map'][..., -1].astype(bool))
        pred_map[(inst_y + d['bbox'][0][0] - min_y,
                  inst_x + d['bbox'][0][1] - min_x)] = \
                d['pred_map'][(inst_y, inst_x,
                               slice(0, 3 if nr_types is None else 4, 1))]

    for k, d in info_dicts_B['left'].items():
        inst_y, inst_x = np.nonzero(d['pred_map'][..., -1].astype(bool))
        pred_map[(inst_y + d['bbox'][0][0] - min_y,
                  inst_x + d['bbox'][0][1] - min_x)] = \
                d['pred_map'][(inst_y, inst_x,
                               slice(0, 3 if nr_types is None else 4, 1))]

    # Run the _post_process and get the dictionary for the individual
    # instances at the chunk boundaries
    bound_inst_info_arr = _post_process(pred_map, block_id=(0, 0), 
                                        nr_types=nr_types,
                                        offset=(min_y, min_x),
                                        tile_shape=(max_y - min_y,
                                                    max_x - min_x),
                                        num_tiles=(1, 1),
                                        check_sides_first=False,
                                        ambiguous_size=-1,
                                        merge_edges=False)

    info_dicts_A['inner'].update(bound_inst_info_arr[0, 0]['inner'])

    info_dicts_A['right'] = {}
    info_dicts_A['bound_pred_maps']['right'] = None

    # Remove the prediction maps form the inner list
    for pred in info_dicts_A['inner'].values():
        del pred['pred_map']

    inst_info_arr = np.array([[info_dicts_A]])

    return inst_info_arr


def _fix_ver_boundaries(inst_info_arr, block_id=None, nr_types=None,
                        num_tiles=None,
                        ambiguous_size=128):
    info_dicts_A = inst_info_arr[0, 0]

    if inst_info_arr.shape[0] == 1:
        # There is no chunk under chunk A because it is the bottom edge of the 
        # image.
        info_dicts_B = dict(
            top={},
            bound_pred_maps=dict(
                top=None,
                top_br_pos=info_dicts_A['bound_pred_maps']['bottom_br_pos']))
    else:
        info_dicts_B = inst_info_arr[1, 0]

    # Check if any of the two chunks is empty before continue
    if (info_dicts_A['bound_pred_maps']['bottom'] is not None
      and info_dicts_B['bound_pred_maps']['top'] is not None):
        pred_map = np.vstack((info_dicts_A['bound_pred_maps']['bottom'],
                              info_dicts_B['bound_pred_maps']['top']))

        pred_map_tl = info_dicts_A['bound_pred_maps']['bottom_tl_pos']
        pred_map_br = info_dicts_B['bound_pred_maps']['top_br_pos']

    elif info_dicts_A['bound_pred_maps']['bottom'] is not None:
        pred_map = np.copy(info_dicts_A['bound_pred_maps']['bottom'])

        pred_map_tl = info_dicts_A['bound_pred_maps']['bottom_tl_pos']
        pred_map_br = info_dicts_A['bound_pred_maps']['bottom_br_pos']

    elif info_dicts_B['bound_pred_maps']['top'] is not None:
        pred_map = np.copy(info_dicts_B['bound_pred_maps']['top'])

        pred_map_tl = info_dicts_B['bound_pred_maps']['top_tl_pos']
        pred_map_br = info_dicts_B['bound_pred_maps']['top_br_pos']

    else:
        # If the current chunk A is at the left or right edge of the image, 
        # send the objects detected at the left and right boundaries to the
        # inner detection list.
        if block_id[1] == 0:
            info_dicts_A['inner'].update(info_dicts_A['left'])

            info_dicts_A['left'] = {}
            info_dicts_A['bound_pred_maps']['left'] = None
            info_dicts_A['bound_pred_maps']['left_br_pos'] = info_dicts_A['tl_pos']

        if block_id[1] == num_tiles[1] - 1:
            info_dicts_A['inner'].update(info_dicts_A['right'])

            info_dicts_A['right'] = {}
            info_dicts_A['bound_pred_maps']['right'] = None
            info_dicts_A['bound_pred_maps']['right_tl_pos'] = info_dicts_A['br_pos']

        inst_info_arr = np.array([[info_dicts_A]])

        return inst_info_arr

    min_y = min(chain(map(lambda d: d['bbox'][0][0],
                          info_dicts_A['bottom'].values()),
                      map(lambda d: d['bbox'][0][0],
                          info_dicts_B['top'].values()),
                      [pred_map_tl[0]]))

    min_x = min(chain(map(lambda d: d['bbox'][0][1],
                          info_dicts_A['bottom'].values()),
                      map(lambda d: d['bbox'][0][1],
                          info_dicts_B['top'].values()),
                      [pred_map_tl[1]]))

    max_y = max(chain(map(lambda d: d['bbox'][1][0],
                          info_dicts_A['bottom'].values()),
                      map(lambda d: d['bbox'][1][0],
                          info_dicts_B['top'].values()),
                      [pred_map_br[0]]))

    max_x = max(chain(map(lambda d: d['bbox'][1][1],
                          info_dicts_A['bottom'].values()),
                      map(lambda d: d['bbox'][1][1],
                          info_dicts_B['top'].values()),
                      [pred_map_br[1]]))

    pad_y_A = pred_map_tl[0] - min_y
    pad_x_A = pred_map_tl[1] - min_x
    pad_y_B = max_y - pred_map_br[0]
    pad_x_B = max_x - pred_map_br[1]

    pred_map = np.pad(pred_map, ((pad_y_A, pad_y_B), (pad_x_A, pad_x_B),
                                 (0, 0)))

    for k, d in info_dicts_A['bottom'].items():
        inst_y, inst_x = np.nonzero(d['pred_map'][..., -1].astype(bool))
        pred_map[(inst_y + d['bbox'][0][0] - min_y,
                    inst_x + d['bbox'][0][1] - min_x)] = \
                    d['pred_map'][(inst_y, inst_x,
                                   slice(0, 3 if nr_types is None else 4, 1))]

    for k, d in info_dicts_B['top'].items():
        inst_y, inst_x = np.nonzero(d['pred_map'][..., -1].astype(bool))
        pred_map[(inst_y + d['bbox'][0][0] - min_y,
                    inst_x + d['bbox'][0][1] - min_x)] = \
                    d['pred_map'][(inst_y, inst_x,
                                   slice(0, 3 if nr_types is None else 4, 1))]

    # Run the _post_process and get the dictionary for the individual
    # instances at the chunk boundaries
    bound_inst_info_arr = _post_process(pred_map, block_id=(0, 0),
                                        nr_types=nr_types,
                                        offset=(min_y, min_x),
                                        tile_shape=(max_y - min_y,
                                                    max_x - min_x),
                                        num_tiles=(1, 1),
                                        check_sides_first=True,
                                        ambiguous_size=ambiguous_size,
                                        merge_edges=False)

    info_dicts_A['inner'].update(bound_inst_info_arr[0, 0]['inner'])
    info_dicts_A['inner'].update(bound_inst_info_arr[0, 0]['top'])
    info_dicts_A['inner'].update(bound_inst_info_arr[0, 0]['bottom'])

    # If the current chunk A is at the left or right edge of the image, send
    # the objects detected at the left and right boundaries to the inner
    # detection list. Otherwise, move those to the corresponding list to be
    # fixed on the next step.
    if block_id[1] == 0:
        info_dicts_A['inner'].update(bound_inst_info_arr[0, 0]['left'])
    else:
        info_dicts_A['left'].update(bound_inst_info_arr[0, 0]['left'])

    if block_id[1] == num_tiles[1] - 1:
        info_dicts_A['inner'].update(bound_inst_info_arr[0, 0]['right'])
    else:
        info_dicts_A['right'].update(bound_inst_info_arr[0, 0]['right'])

    info_dicts_A['bottom'] = {}
    info_dicts_A['bound_pred_maps']['bottom'] = None

    inst_info_arr = np.array([[info_dicts_A]])

    return inst_info_arr


class InferManagerDask(base.InferManager):
    def _save_json(self, path, info_dict, mag=None):
        json_dict = {"mag": mag, "nuc": info_dict}  # to sync the format protocol
        with open(path, "w") as handle:
            json.dump(json_dict, handle)
        return info_dict

    def _parse_args(self, run_args):
        """Parse command line arguments and set as instance variables."""
        for variable, value in run_args.items():
            self.__setattr__(variable, value)

        # to tuple
        self.tile_shape = [self.tile_shape, self.tile_shape]
        self.patch_input_shape = [self.patch_input_shape, self.patch_input_shape]
        self.patch_output_shape = [self.patch_output_shape, self.patch_output_shape]
        return

    def process_single_file(self, wsi_path, msk_path, output_dir):
        """Process a single whole-slide image and save the results.

        Args:
            wsi_path: path to input whole-slide image
            msk_path: path to input mask. If not supplied, mask will be automatically generated.
            output_dir: path where output will be saved

        """
        # TODO: customize universal file handler to sync the protocol
        scaled_wsi_mag = 1.25  # ! hard coded
        ambiguous_size = self.ambiguous_size
        tile_shape = (np.array(self.tile_shape)).astype(np.int64)
        patch_output_shape = np.array(self.patch_output_shape)

        path_obj = pathlib.Path(wsi_path)
        wsi_ext = path_obj.suffix
        wsi_name = path_obj.stem

        self.wsi_handler = get_file_handler(wsi_path, backend=wsi_ext,
                                            data_group=self.method['data_group'])

        if msk_path is not None and os.path.isfile(msk_path):
            self.wsi_mask = cv2.imread(msk_path)
            self.wsi_mask = cv2.cvtColor(self.wsi_mask, cv2.COLOR_BGR2GRAY)
            self.wsi_mask[self.wsi_mask > 0] = 1

        else:
            log_info(
                "WARNING: No mask found, generating mask via thresholding at 1.25x!"
            )

            # simple method to extract tissue regions using intensity thresholding and morphological operations
            def simple_get_mask():
                wsi_thumb_rgb = self.wsi_handler.get_full_img(read_mag=scaled_wsi_mag)
                gray = cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
                mask = morphology.remove_small_objects(
                    mask == 0, min_size=16 * 16, connectivity=2
                )
                mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
                mask = morphology.binary_dilation(mask, morphology.disk(16))
                return mask

            self.wsi_mask = np.array(simple_get_mask() > 0, dtype=np.uint8)

            if self.save_mask:
                cv2.imwrite("%s/mask/%s.png" % (output_dir, wsi_name), self.wsi_mask * 255)

            if self.save_thumb:
                wsi_thumb_rgb = self.wsi_handler.get_full_img(read_mag=1.25)
                cv2.imwrite("%s/thumb/%s.png" % (output_dir, wsi_name),
                            cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2BGR))

        self.wsi_inst_info = {}

        # Process chunks/tiles/patches according to the mask generated above.
        # If the mask is blank at that position, return a zero array instead of
        # the network inference.
        offset = (self.patch_input_shape[0] - self.patch_output_shape[0]) // 2

        # dask.config.set(scheduler='single-threaded')

        z_wsi = da.from_zarr(self.wsi_handler._file_path,
                             component=self.method['data_group'] + '/0')

        self.wsi_proc_shape = self.wsi_handler.get_dimensions(self.proc_mag)
        self.wsi_proc_shape = self.wsi_proc_shape[::-1]

        num_patches = [int(math.ceil(s / p))
                       for s, p in zip(self.wsi_proc_shape,
                                       patch_output_shape)]

        num_tiles = [int(math.ceil(o_p / t_p * n_p))
                     for t_p, o_p, n_p in zip(tile_shape,
                                              patch_output_shape,
                                              num_patches)]

        scaled_patch_shape = [int(math.ceil(p * scaled_wsi_mag
                                            / self.wsi_handler.metadata["base_mag"]))
                              for p in self.patch_output_shape]
        scaled_tile_shape = [int(math.ceil(p * scaled_wsi_mag
                                           / self.wsi_handler.metadata["base_mag"]))
                             for p in tile_shape]

        paddings = [(0, 0), (0, 0), (0, 0)]
        paddings += [(0, n_p * p - s)
                     for s, p, n_p in zip(self.wsi_proc_shape,
                                          patch_output_shape,
                                          num_patches)]

        padded_z_wsi = da.pad(z_wsi, paddings, mode='constant')

        padded_z_wsi = padded_z_wsi.rechunk(
            (1, 3, 1, patch_output_shape[0], patch_output_shape[1]))

        pred_z_wsi = padded_z_wsi.map_overlap(
            _infer_patch,
            net=self.run_step,
            mask=self.wsi_mask,
            scaled_patch_shape=scaled_patch_shape,
            offset=2 * offset,
            num_patches=num_patches,
            nr_types=self.nr_types,
            available_gpus_ids=self.available_gpus_ids,
            depth=(0, 0, 0, offset, offset),
            boundary=0,
            trim=False,
            dtype=np.float32,
            drop_axis=(0, 1, 2),
            new_axis=2,
            chunks=(patch_output_shape[0], patch_output_shape[1], 4),
            meta=np.empty((0), dtype=np.float32))

        pred_z_wsi = pred_z_wsi.rechunk((tile_shape[0], tile_shape[1], 4))

        post_z_wsi = pred_z_wsi.map_blocks(
            _post_process,
            mask=self.wsi_mask,
            scaled_patch_shape=scaled_tile_shape,
            nr_types=self.nr_types,
            offset=(0, 0),
            tile_shape=tile_shape,
            num_tiles=num_tiles,
            check_sides_first=False,
            ambiguous_size=ambiguous_size,
            merge_edges=True,
            dtype=np.object,
            drop_axis=2,
            chunks=(1, 1),
            meta=np.empty((0), dtype=np.object)
        )

        with ProgressBar():
            post_z_wsi = post_z_wsi.compute()

        post_z_wsi = da.from_array(post_z_wsi, chunks=(1, 1))

        # Fix cells detected at chunk boundaries
        dict_ver_fix = post_z_wsi.map_overlap(
            _fix_ver_boundaries,
            nr_types=self.nr_types,                        
            num_tiles=num_tiles,
            ambiguous_size=ambiguous_size,
            depth=((0, 1), (0, 0)),
            boundary='none',
            trim=False,
            dtype=np.object,
            meta=np.empty((0), dtype=np.object)
        )

        with ProgressBar():
            dict_ver_fix = dict_ver_fix.compute()

        dict_ver_fix = da.from_array(dict_ver_fix, chunks=(1, 1))
        dict_hor_fix = dict_ver_fix.map_overlap(
            _fix_hor_boundaries,
            nr_types=self.nr_types,
            depth=((0, 0), (0, 1)),
            boundary='none',
            trim=False,
            dtype=np.object,
            meta=np.empty((0), dtype=np.object)
        )

        with ProgressBar():
            dicts = dict_hor_fix.compute()

        start = time.perf_counter()
        self.wsi_inst_info = reduce(lambda d1, d2: {**d1, **d2['inner']},
                                    list(dicts.flatten()),
                                    {})

        all_keys = list(self.wsi_inst_info.keys())
        for i, k in enumerate(all_keys):
            self.wsi_inst_info[i+1] = self.wsi_inst_info.pop(k)
            if 'pred_map' in self.wsi_inst_info[i+1]:
                del(self.wsi_inst_info[i+1]['pred_map'])

        # ! cant possibly save the inst map at high res, too large

        if self.save_mask or self.save_thumb:
            json_path = "%s/json/%s.json" % (output_dir, wsi_name)
        else:
            json_path = "%s/%s.json" % (output_dir, wsi_name)

        self._save_json(json_path, self.wsi_inst_info, mag=self.proc_mag)
        end = time.perf_counter()
        log_info("Save Time: {0}".format(end - start))

    def process_wsi_list(self, run_args):
        """Process a list of whole-slide images.

        Args:
            run_args: arguments as defined in run_infer.py
        
        """
        self._parse_args(run_args)

        if not os.path.exists(self.output_dir + "/json/"):
            rm_n_mkdir(self.output_dir + "/json/")
        if self.save_thumb:
            if not os.path.exists(self.output_dir + "/thumb/"):
                rm_n_mkdir(self.output_dir + "/thumb/")
        if self.save_mask:
            if not os.path.exists(self.output_dir + "/mask/"):
                rm_n_mkdir(self.output_dir + "/mask/")

        wsi_path_list = glob.glob(self.input_dir + "/*")
        wsi_path_list.sort()  # ensure ordering
        for wsi_path in wsi_path_list[:]:
            wsi_base_name = pathlib.Path(wsi_path).stem
            msk_path = "%s/%s.png" % (self.input_mask_dir, wsi_base_name)
            if self.save_thumb or self.save_mask:
                output_file = "%s/json/%s.json" % (self.output_dir, wsi_base_name)
            else:
                output_file = "%s/%s.json" % (self.output_dir, wsi_base_name)
            if os.path.exists(output_file):
                log_info("Skip: %s" % wsi_base_name)
                continue

            log_info("Processing file %s" % wsi_base_name)
            self.process_single_file(wsi_path, msk_path, self.output_dir)

        return
