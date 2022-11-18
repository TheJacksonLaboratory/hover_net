import multiprocessing as mp
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait
from multiprocessing import Lock, Pool

mp.set_start_method("spawn", True)  # ! must be at top for VScode debugging

import argparse
import glob
import json
import logging
import math
import os
import pathlib
import re
import shutil
import sys
import time
from functools import reduce
from importlib import import_module

import cv2
import numpy as np
import zarr
from numcodecs import Blosc

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.ndimage import measurements, binary_fill_holes
from skimage import morphology, segmentation

import psutil
import scipy.io as sio
import torch
from skimage import morphology
import torch.utils.data as data
from torchvision import transforms
import tqdm
from dataloader.infer_loader import SerializeArray, SerializeZarrArray, SerializeFileList
from dataloader import ZarrDataset
from docopt import docopt
from misc.utils import (
    cropping_center,
    get_bounding_box,
    log_debug,
    log_info,
    rm_n_mkdir,
)
from misc.wsi_handler import get_file_handler

from . import base

thread_lock = Lock()


####
def _init_worker_child(lock_):
    global lock
    lock = lock_


####
def _remove_inst(inst_map, remove_id_list):
    """Remove instances with id in remove_id_list.
    
    Args:
        inst_map: map of instances
        remove_id_list: list of ids to remove from inst_map
    """
    for inst_id in remove_id_list:
        inst_map[inst_map == inst_id] = 0
    return inst_map


####
def _get_patch_top_left_info(img_shape, input_size, output_size):
    """Get top left coordinate information of patches from original image.

    Args:
        img_shape: input image shape
        input_size: patch input shape
        output_size: patch output shape

    """
    in_out_diff = input_size - output_size
    nr_step = np.floor((img_shape - in_out_diff) / output_size) + 1
    last_output_coord = (in_out_diff // 2) + (nr_step) * output_size
    # generating subpatches index from orginal
    output_tl_y_list = np.arange(
        in_out_diff[0] // 2, last_output_coord[0], output_size[0], dtype=np.int32
    )
    output_tl_x_list = np.arange(
        in_out_diff[1] // 2, last_output_coord[1], output_size[1], dtype=np.int32
    )
    output_tl_y_list, output_tl_x_list = np.meshgrid(output_tl_y_list, output_tl_x_list)
    output_tl = np.stack(
        [output_tl_y_list.flatten(), output_tl_x_list.flatten()], axis=-1
    )
    input_tl = output_tl - in_out_diff // 2
    return input_tl, output_tl


#### all must be np.array
def _get_tile_info(img_shape, tile_shape, ambiguous_size=128):
    """Get information of tiles used for post processing.

    Args:
        img_shape: input image shape
        tile_shape: tile shape used for post processing
        ambiguous_size: used to define area at tile boundaries
    
    """
    # * get normal tiling set
    tile_grid_top_left, _ = _get_patch_top_left_info(img_shape, tile_shape, tile_shape)
    tile_grid_bot_right = []
    for idx in list(range(tile_grid_top_left.shape[0])):
        tile_tl = tile_grid_top_left[idx][:2]
        tile_br = tile_tl + tile_shape
        axis_sel = tile_br > img_shape
        tile_br[axis_sel] = img_shape[axis_sel]
        tile_grid_bot_right.append(tile_br)
    tile_grid_bot_right = np.array(tile_grid_bot_right)
    tile_grid = np.stack([tile_grid_top_left, tile_grid_bot_right], axis=1)
    tile_grid_x = np.unique(tile_grid_top_left[:, 1])
    tile_grid_y = np.unique(tile_grid_top_left[:, 0])
    # * get tiling set to fix vertical and horizontal boundary between tiles
    # for sanity, expand at boundary `ambiguous_size` to both side vertical and horizontal
    stack_coord = lambda x: np.stack([x[0].flatten(), x[1].flatten()], axis=-1)
    tile_boundary_x_top_left = np.meshgrid(
        tile_grid_y, tile_grid_x[1:] - ambiguous_size
    )
    tile_boundary_x_bot_right = np.meshgrid(
        tile_grid_y + tile_shape[0], tile_grid_x[1:] + ambiguous_size
    )
    tile_boundary_x_top_left = stack_coord(tile_boundary_x_top_left)
    tile_boundary_x_bot_right = stack_coord(tile_boundary_x_bot_right)
    tile_boundary_x = np.stack(
        [tile_boundary_x_top_left, tile_boundary_x_bot_right], axis=1
    )
    #
    tile_boundary_y_top_left = np.meshgrid(
        tile_grid_y[1:] - ambiguous_size, tile_grid_x
    )
    tile_boundary_y_bot_right = np.meshgrid(
        tile_grid_y[1:] + ambiguous_size, tile_grid_x + tile_shape[1]
    )
    tile_boundary_y_top_left = stack_coord(tile_boundary_y_top_left)
    tile_boundary_y_bot_right = stack_coord(tile_boundary_y_bot_right)
    tile_boundary_y = np.stack(
        [tile_boundary_y_top_left, tile_boundary_y_bot_right], axis=1
    )
    tile_boundary = np.concatenate([tile_boundary_x, tile_boundary_y], axis=0)
    # * get tiling set to fix the intersection of 4 tiles
    tile_cross_top_left = np.meshgrid(
        tile_grid_y[1:] - 2 * ambiguous_size, tile_grid_x[1:] - 2 * ambiguous_size
    )
    tile_cross_bot_right = np.meshgrid(
        tile_grid_y[1:] + 2 * ambiguous_size, tile_grid_x[1:] + 2 * ambiguous_size
    )
    tile_cross_top_left = stack_coord(tile_cross_top_left)
    tile_cross_bot_right = stack_coord(tile_cross_bot_right)
    tile_cross = np.stack([tile_cross_top_left, tile_cross_bot_right], axis=1)
    return tile_grid, tile_boundary, tile_cross


####
def _get_chunk_patch_info(
    img_shape, chunk_input_shape, patch_input_shape, patch_output_shape
):
    """Get chunk patch info. Here, chunk refers to tiles used during inference.

    Args:
        img_shape: input image shape
        chunk_input_shape: shape of tiles used for post processing
        patch_input_shape: input patch shape
        patch_output_shape: output patch shape

    """
    # If the input image is smaller than the patch size, consider as if that
    # was of the patch size instead
    img_shape = list(map(max, img_shape, patch_input_shape))

    round_to_multiple = lambda x, y: np.floor(x / y) * y
    patch_diff_shape = patch_input_shape - patch_output_shape

    chunk_output_shape = chunk_input_shape - patch_diff_shape
    chunk_output_shape = round_to_multiple(
        chunk_output_shape, patch_output_shape
    ).astype(np.int64)
    chunk_input_shape = (chunk_output_shape + patch_diff_shape).astype(np.int64)

    patch_input_tl_list, _ = _get_patch_top_left_info(
        img_shape, patch_input_shape, patch_output_shape
    )
    patch_input_br_list = patch_input_tl_list + patch_input_shape
    patch_output_tl_list = patch_input_tl_list + patch_diff_shape
    patch_output_br_list = patch_output_tl_list + patch_output_shape
    patch_info_list = np.stack(
        [
            np.stack([patch_input_tl_list, patch_input_br_list], axis=1),
            np.stack([patch_output_tl_list, patch_output_br_list], axis=1),
        ],
        axis=1,
    )

    chunk_input_tl_list, _ = _get_patch_top_left_info(
        img_shape, chunk_input_shape, chunk_output_shape
    )
    chunk_input_br_list = chunk_input_tl_list + chunk_input_shape
    # * correct the coord so it stay within source image
    y_sel = np.nonzero(chunk_input_br_list[:, 0] > img_shape[0])[0]
    x_sel = np.nonzero(chunk_input_br_list[:, 1] > img_shape[1])[0]
    chunk_input_br_list[y_sel, 0] = (
        img_shape[0] - patch_diff_shape[0]
    ) - chunk_input_tl_list[y_sel, 0]
    chunk_input_br_list[x_sel, 1] = (
        img_shape[1] - patch_diff_shape[1]
    ) - chunk_input_tl_list[x_sel, 1]
    chunk_input_br_list[y_sel, 0] = round_to_multiple(
        chunk_input_br_list[y_sel, 0], patch_output_shape[0]
    )
    chunk_input_br_list[x_sel, 1] = round_to_multiple(
        chunk_input_br_list[x_sel, 1], patch_output_shape[1]
    )
    chunk_input_br_list[y_sel, 0] += chunk_input_tl_list[y_sel, 0] + patch_diff_shape[0]
    chunk_input_br_list[x_sel, 1] += chunk_input_tl_list[x_sel, 1] + patch_diff_shape[1]
    chunk_output_tl_list = chunk_input_tl_list + patch_diff_shape // 2
    chunk_output_br_list = chunk_input_br_list - patch_diff_shape // 2  # may off pixels
    chunk_info_list = np.stack(
        [
            np.stack([chunk_input_tl_list, chunk_input_br_list], axis=1),
            np.stack([chunk_output_tl_list, chunk_output_br_list], axis=1),
        ],
        axis=1,
    )

    return chunk_info_list, patch_info_list


####
def _post_proc_para_wrapper(pred_map_mmap_path, tile_info, func, func_kwargs):
    """Wrapper for parallel post processing."""
    idx, tile_tl, tile_br = tile_info
    wsi_pred_map_ptr = np.load(pred_map_mmap_path, mmap_mode="r")
    tile_pred_map = wsi_pred_map_ptr[tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]]
    tile_pred_map = np.array(tile_pred_map)  # from mmap to ram
    return func(tile_pred_map, **func_kwargs), tile_info


####
def _post_proc_para_wrapper_zarr(pred_map_mmap_path, tile_info, func, func_kwargs):
    """Wrapper for parallel post processing."""
    idx, tile_tl, tile_br = tile_info
    wsi_pred_map_ptr = zarr.open(pred_map_mmap_path, "r")
    tile_pred_map = wsi_pred_map_ptr[0, :, 0, tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]].transpose(1, 2, 0)

    return func(tile_pred_map, **func_kwargs), tile_info
    # return da.map_blocks(func, tile_pred_map, **func_kwargs, dtype=np.float32, meta=np.empty((), dtype=np.float32), chunks=(4, 226, 610))


####
def _assemble_and_flush(wsi_pred_map_mmap_path, chunk_info, patch_output_list):
    """Assemble the results. Write to newly created holder for this wsi"""
    wsi_pred_map_ptr = np.load(wsi_pred_map_mmap_path, mmap_mode="r+")
    chunk_pred_map = wsi_pred_map_ptr[
        chunk_info[1][0][0] : chunk_info[1][1][0],
        chunk_info[1][0][1] : chunk_info[1][1][1],
    ]
    if patch_output_list is None:
        # chunk_pred_map[:] = 0 # zero flush when there is no-results
        # print(chunk_info.flatten(), 'flush 0')
        return

    for pinfo in patch_output_list:
        pcoord, pdata = pinfo
        pdata = np.squeeze(pdata)
        pcoord = np.squeeze(pcoord)[:2]
        chunk_pred_map[
            pcoord[0] : pcoord[0] + pdata.shape[0],
            pcoord[1] : pcoord[1] + pdata.shape[1],
        ] = pdata
    # print(chunk_info.flatten(), 'pass')
    return


def _assemble_and_flush_zarr(wsi_pred_map_mmap_path, chunk_info, patch_output_list):
    """Assemble the results. Write to newly created holder for this wsi"""
    if patch_output_list is None:
        # chunk_pred_map[:] = 0 # zero flush when there is no-results
        # print(chunk_info.flatten(), 'flush 0')
        return
    wsi_pred_map_ptr = zarr.open(wsi_pred_map_mmap_path, mode="r+")

    for pinfo in patch_output_list:
        pcoord, pdata = pinfo
        pdata = np.squeeze(pdata).transpose(2, 0, 1)
        pcoord = np.squeeze(pcoord)[:2]
        if pcoord[0] + chunk_info[1][0][0] + pdata.shape[1] > wsi_pred_map_ptr.shape[-2] \
           or pcoord[1] + chunk_info[1][0][1] + pdata.shape[2] > wsi_pred_map_ptr.shape[-1]:
           continue
        wsi_pred_map_ptr[0, :, 0,
                         pcoord[0] + chunk_info[1][0][0] : pcoord[0] + chunk_info[1][0][0] + pdata.shape[1],
                         pcoord[1] + chunk_info[1][0][1] : pcoord[1] + chunk_info[1][0][1] + pdata.shape[2]] = pdata
    # print(chunk_info.flatten(), 'pass')
    return


def _proc_np_hv(pred_inst):
    blb_raw = pred_inst[..., 0]
    h_dir_raw = pred_inst[..., 1]
    v_dir_raw = pred_inst[..., 2]

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
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
    marker = measurements.label(marker)[0]
    marker = morphology.remove_small_objects(marker, min_size=10)

    pred_inst = segmentation.watershed(dist, markers=marker, mask=blb)

    return pred_inst


def _infer_patch(patch, net, wsi_mag=40.0, scaled_wsi_mag=1.25):
    scale_factor = scaled_wsi_mag / wsi_mag
    # now rescale then return
    if scale_factor > 1.0:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR
    wsi_thumb_rgb = cv2.resize(
        np.moveaxis(patch[0, :, 0], 0, -1), (0, 0), fx=scale_factor,
        fy=scale_factor,
        interpolation=interp)

    gray = cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    mask = morphology.remove_small_objects(
        mask == 0, min_size=16 * 16, connectivity=2
    )
    mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
    mask = morphology.binary_dilation(mask, morphology.disk(16))

    if mask.sum() == 0:
        pred_map = np.zeros((*patch.shape[-2:], 4))

    else:
        with torch.no_grad():
            pred_map = net(
                torch.from_numpy(np.moveaxis(patch[:, :, 0, ...], 1, -1)))
            pred_map = pred_map[0]

    return pred_map


def _post_process(pred_map, nr_types=None, tl_pos=None, br_pos=None):
    if tl_pos is None:
        tl_pos = (0, 0)
    if br_pos is None:
        br_pos = (164, 164)

    inner_info_dict = {}
    left_info_dict = {}
    right_info_dict = {}
    top_info_dict = {}
    bottom_info_dict = {}

    if pred_map[..., 0].sum() > 0:
        H, W = pred_map.shape[:2]

        if nr_types is not None:
            pred_type = pred_map[..., 0]
            pred_inst = pred_map[..., 1:]
            pred_type = pred_type.astype(np.int32)
        else:
            pred_inst = pred_map

        pred_inst = np.squeeze(pred_inst)
        pred_inst = _proc_np_hv(pred_inst)

        inst_id_list = np.unique(pred_inst)
        for inst_id in inst_id_list:
            if inst_id == 0:
                # Ignore background
                continue

            inst_map = pred_inst == inst_id
            # TODO: chane format of bbox output
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1] + tl_pos[1] # X
            inst_contour[:, 1] += inst_bbox[0][0] + tl_pos[0]  # Y
            inst_centroid[0] += inst_bbox[0][1] + tl_pos[1]  # X
            inst_centroid[1] += inst_bbox[0][0] + tl_pos[0]  # Y

            if nr_types is not None:
                inst_type_crop = pred_type[inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]]
                inst_type = inst_type_crop[np.nonzero(inst_map)]
                type_list, type_pixels = np.unique(inst_type, return_counts=True)
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
            inst_bbox[0][0] += tl_pos[0]  # Y
            inst_bbox[1][1] += tl_pos[1] # X
            inst_bbox[1][0] += tl_pos[0]  # Y

            inst_info_dict = {
                    "bbox": inst_bbox.tolist(),
                    "centroid": inst_centroid.tolist(),
                    "contour": inst_contour.tolist(),
                    "type_prob": type_prob,
                    "type": inst_type,
                }

            inst_key = '%i,%i,%i,%i,%i' % (*tl_pos, *br_pos, inst_id)
            inst_pred_map = pred_map[bbox_tl_y:bbox_br_y,
                                     bbox_tl_x:bbox_br_x,
                                     1:]
            inst_pred_map = inst_pred_map * inst_map[..., np.newaxis]

            if bbox_tl_x == 0:
                left_info_dict[inst_key] = inst_info_dict
                left_info_dict[inst_key]['pred_map'] = inst_pred_map
            elif bbox_br_x == W - 1:
                right_info_dict[inst_key] = inst_info_dict
                right_info_dict[inst_key]['pred_map'] = inst_pred_map
            elif bbox_tl_y == 0:
                top_info_dict[inst_key] = inst_info_dict
                top_info_dict[inst_key]['pred_map'] = inst_pred_map
            elif bbox_br_y == H - 1:
                bottom_info_dict[inst_key] = inst_info_dict
                bottom_info_dict[inst_key]['pred_map'] = inst_pred_map
            else:
                inner_info_dict[inst_key] = inst_info_dict

    inst_info_arr = np.array([[[inner_info_dict]], [[left_info_dict]],
                              [[right_info_dict]],
                              [[top_info_dict]],
                              [[bottom_info_dict]]])

    return inst_info_arr


def _fix_hor_boundaries(inst_info_arr):
    (inner_info_dict_A,
     left_info_dict_A,
     right_info_dict_A,
     top_info_dict_A,
     bottom_info_dict_A) = inst_info_arr[:, 0, 0]

    left_info_dict_B = inst_info_arr[1, 0, 1]

    if len(left_info_dict_B) == 0:
        # If there are no objects detected on the left edge of chunk B, move
        # all objects at the right edge of chunk A, except those at its top or
        # bottom edge, to the inner object dictionary.
        for k, d in right_info_dict_A.items():
            tl_pos = list(map(int, k.split(',')[:2]))
            br_pos = list(map(int, k.split(',')[2:4]))

            if d['bbox'][0][0] == tl_pos[0]:
                top_info_dict_A[k] = d
            elif d['bbox'][1][1] == br_pos[0]:
                bottom_info_dict_A[k] = d
            else:
                inner_info_dict_A[k] = d

    elif len(right_info_dict_A) > 0:
        chunk_info = list(map(int,
                              list(right_info_dict_A.keys())[0].split(',')))
        tl_pos = chunk_info[:2]
        br_pos = chunk_info[2:4]
        min_x_A = min(map(lambda d: d[1]['bbox'][0][1],
                          right_info_dict_A.items()))
        max_x_B = max(map(lambda d: d[1]['bbox'][1][1],
                          left_info_dict_B.items()))

        min_y_A = min(map(lambda d: d[1]['bbox'][0][0],
                          right_info_dict_A.items()))
        min_y_B = min(map(lambda d: d[1]['bbox'][0][0],
                          left_info_dict_B.items()))

        max_y_A = max(map(lambda d: d[1]['bbox'][1][0],
                          right_info_dict_A.items()))
        max_y_B = max(map(lambda d: d[1]['bbox'][1][0],
                          left_info_dict_B.items()))

        min_y = min(min_y_A, min_y_B)
        max_y = max(max_y_A, max_y_B)

        pred_map = np.zeros((max_y - min_y + 1,
                             max_x_B - min_x_A + 1,
                             3),
                            dtype=np.float32)

        for k, d in right_info_dict_A.items():
            pred_map[d['bbox'][0][0] - min_y:d['bbox'][1][0] - min_y,
                     d['bbox'][0][1] - min_x_A:d['bbox'][1][1] - min_x_A,
                     :] = d['pred_map']
        for k, d in left_info_dict_B.items():
            pred_map[d['bbox'][0][0] - min_y:d['bbox'][1][0] - min_y,
                     d['bbox'][0][1] - min_x_A:d['bbox'][1][1] - min_x_A,
                     :] = d['pred_map']

        # TODO: Run the _proc_np_hv function to the the individual instances

    inst_info_arr = np.array(
        [[[inner_info_dict_A]], [[left_info_dict_A]], [[{}]],
         [[top_info_dict_A]],
         [[bottom_info_dict_A]]])

    return inst_info_arr


####
class InferManager(base.InferManager):
    def __run_model(self, patch_top_left_list, pbar_desc):
        # TODO: the cost of creating dataloader may not be cheap ?
        if (hasattr(self.wsi_handler.file_ptr, 'store')
           and '.zarr' in self.wsi_handler.file_ptr.store.path):
            dataset = SerializeZarrArray(
                self.wsi_handler.file_ptr[self.method['data_group'] + '/0'],
                self.cached_pos,
                patch_top_left_list,
                self.patch_input_shape,
            )
        else:
            dataset = SerializeArray(
                "%s/cache_chunk.npy" % self.cache_path,
                patch_top_left_list,
                self.patch_input_shape,
            )

        dataloader = data.DataLoader(
            dataset,
            num_workers=self.nr_inference_workers,
            batch_size=self.batch_size,
            drop_last=False,
        )

        pbar = tqdm.tqdm(
            desc=pbar_desc,
            leave=True,
            total=int(len(dataloader)),
            ncols=80,
            ascii=True,
            position=0,
        )

        # run inference on input patches
        accumulated_patch_output = []
        for batch_idx, batch_data in enumerate(dataloader):
            sample_data_list, sample_info_list = batch_data
            sample_output_list = self.run_step(sample_data_list)
            sample_info_list = sample_info_list.numpy()
            curr_batch_size = sample_output_list.shape[0]
            sample_output_list = np.split(sample_output_list, curr_batch_size, axis=0)
            sample_info_list = np.split(sample_info_list, curr_batch_size, axis=0)

            if isinstance(self.wsi_pred_map, zarr.Array):
                # If the input image is already in zarr format, store the
                # prediction map in a Zarr file instead of saving it for
                # constructing the map later
                offset = (self.patch_input_shape[0] - self.patch_output_shape[0]) // 2
                for pos, pred in zip(sample_info_list, sample_output_list):
                    tl_x_dst = self.cached_pos[1] + pos[0, 1] + offset
                    tl_y_dst = self.cached_pos[0] + pos[0, 0] + offset
                    br_x_dst = min(self.wsi_pred_map.shape[4], self.cached_pos[1] + pos[0, 1] + offset + pred.shape[2])
                    br_y_dst = min(self.wsi_pred_map.shape[3], self.cached_pos[0] + pos[0, 0] + offset + pred.shape[1])

                    br_x_src = br_x_dst - tl_x_dst
                    br_y_src = br_y_dst - tl_y_dst

                    self.wsi_pred_map[0, :, 0, tl_y_dst:br_y_dst, tl_x_dst:br_x_dst] = \
                        pred[0, :br_y_src, :br_x_src, :].transpose(2, 0, 1)
            else:
                sample_output_list = list(zip(sample_info_list, sample_output_list))
                accumulated_patch_output.extend(sample_output_list)

            pbar.update()
        pbar.close()
        return accumulated_patch_output

    def __select_valid_patches(self, patch_info_list, has_output_info=True):
        """Select valid patches from the list of input patch information.

        Args:
            patch_info_list: patch input coordinate information
            has_output_info: whether output information is given
        
        """
        down_sample_ratio = self.wsi_mask.shape[0] / self.wsi_proc_shape[0]
        selected_indices = []
        for idx in range(patch_info_list.shape[0]):
            patch_info = patch_info_list[idx]
            patch_info = np.squeeze(patch_info)
            # get the box at corresponding mag of the mask
            if has_output_info:
                output_bbox = patch_info[1] * down_sample_ratio
            else:
                output_bbox = patch_info * down_sample_ratio
            output_bbox = np.rint(output_bbox).astype(np.int64)
            # coord of the output of the patch (i.e center regions)
            output_roi = self.wsi_mask[
                output_bbox[0][0] : output_bbox[1][0],
                output_bbox[0][1] : output_bbox[1][1],
            ]
            if np.sum(output_roi) > 0:
                selected_indices.append(idx)
        sub_patch_info_list = patch_info_list[selected_indices]
        return sub_patch_info_list

    def __get_raw_prediction(self, chunk_info_list, patch_info_list):
        """Process input tiles (called chunks for inference) with HoVer-Net.

        Args:
            chunk_info_list: list of inference tile coordinate information
            patch_info_list: list of patch coordinate information
        
        """
        # 1 dedicated thread just to write results back to disk
        proc_pool = Pool(processes=1)
        masking = lambda x, a, b: (a <= x) & (x <= b)
        for idx in range(0, chunk_info_list.shape[0]):
            chunk_info = chunk_info_list[idx]
            # select patch basing on top left coordinate of input
            start_coord = chunk_info[0, 0]
            end_coord = chunk_info[0, 1] - self.patch_input_shape
            selection = masking(
                patch_info_list[:, 0, 0, 0], start_coord[0], end_coord[0]
            ) & masking(patch_info_list[:, 0, 0, 1], start_coord[1], end_coord[1])
            chunk_patch_info_list = np.array(
                patch_info_list[selection]
            )  # * do we need copy ?

            # further select only the patches within the provided mask
            chunk_patch_info_list = self.__select_valid_patches(chunk_patch_info_list)

            # there no valid patches, so flush 0 and skip
            if chunk_patch_info_list.shape[0] == 0:
                if not (hasattr(self.wsi_handler.file_ptr, 'store')
                        and '.zarr' in self.wsi_handler.file_ptr.store.path):
                    proc_pool.apply_async(
                        _assemble_and_flush,
                        args=("%s/pred_map.npy" % self.cache_path,
                            chunk_info,
                            None)
                    )
                continue

            chunk_patch_info_list -= chunk_info[:, 0]
            # shift the coordinare from wrt slide to wrt chunk
            if (hasattr(self.wsi_handler.file_ptr, 'store')
               and '.zarr' in self.wsi_handler.file_ptr.store.path):
                self.cached_pos = chunk_info[0][0]
                self.cached_size = chunk_info[0][1] - chunk_info[0][0]
            else:
                chunk_data = self.wsi_handler.read_region(
                    chunk_info[0][0][::-1], (chunk_info[0][1] - chunk_info[0][0])[::-1]
                )
                chunk_data = np.array(chunk_data)[..., :3]
                np.save("%s/cache_chunk.npy" % self.cache_path, chunk_data)

            pbar_desc = "Process Chunk %d/%d" % (idx, chunk_info_list.shape[0])
            patch_output_list = self.__run_model(
                chunk_patch_info_list[:, 0, 0], pbar_desc
            )

            # When using Zarr to store the prediction map, it is not needed to
            # assemble it since the map is generated at inference time
            if not (hasattr(self.wsi_handler.file_ptr, 'store')
                   and '.zarr' in self.wsi_handler.file_ptr.store.path):
                proc_pool.apply_async(
                    _assemble_and_flush,
                    args=("%s/pred_map.npy" % self.cache_path,
                          chunk_info,
                          patch_output_list),
                )
        proc_pool.close()
        proc_pool.join()
        return

    def __dispatch_post_processing(self, tile_info_list, callback):
        """Post processing initialisation."""
        proc_pool = None
        if self.nr_post_proc_workers > 0:
            proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

        future_list = []
        if (hasattr(self.wsi_handler.file_ptr, 'store')
           and '.zarr' in self.wsi_handler.file_ptr.store.path):
            wsi_pred_map_mmap_path = "%s/pred_map.zarr" % self.cache_path
            post_proc_wrapper_fun = _post_proc_para_wrapper_zarr
        else:
            wsi_pred_map_mmap_path = "%s/pred_map.npy" % self.cache_path
            post_proc_wrapper_fun = _post_proc_para_wrapper
        for idx in list(range(tile_info_list.shape[0])):
            tile_tl = tile_info_list[idx][0]
            tile_br = tile_info_list[idx][1]

            tile_info = (idx, tile_tl, tile_br)
            func_kwargs = {
                "nr_types": self.method["model_args"]["nr_types"],
                "return_centroids": True,
            }

            # TODO: standarize protocol
            if proc_pool is not None:
                proc_future = proc_pool.submit(
                    post_proc_wrapper_fun,
                    wsi_pred_map_mmap_path,
                    tile_info,
                    self.post_proc_func,
                    func_kwargs,
                )

                # ! manually poll future and call callback later as there is no guarantee
                # ! that the callback is called from main thread
                future_list.append(proc_future)
            else:
                results = post_proc_wrapper_fun(
                    wsi_pred_map_mmap_path, tile_info, self.post_proc_func, func_kwargs
                )
                callback(results)
        if proc_pool is not None:
            silent_crash = False
            # loop over all to check state a.k.a polling
            for future in as_completed(future_list):
                # ! silent crash, cancel all and raise error
                if future.exception() is not None:
                    silent_crash = True
                    # ! cancel somehow leads to cascade error later
                    # ! so just poll it then crash once all future
                    # ! acquired for now
                    # for future in future_list:
                    #     future.cancel()
                    # break
                else:
                    callback(future.result())
            assert not silent_crash
        return

    def _parse_args(self, run_args):
        """Parse command line arguments and set as instance variables."""
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        # to tuple
        self.chunk_shape = [self.chunk_shape, self.chunk_shape]
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
        ambiguous_size = self.ambiguous_size
        tile_shape = (np.array(self.tile_shape)).astype(np.int64)
        chunk_input_shape = np.array(self.chunk_shape)
        patch_input_shape = np.array(self.patch_input_shape)
        patch_output_shape = np.array(self.patch_output_shape)

        path_obj = pathlib.Path(wsi_path)
        wsi_ext = path_obj.suffix
        wsi_name = path_obj.stem

        start = time.perf_counter()
        self.wsi_handler = get_file_handler(wsi_path, backend=wsi_ext, data_group=self.method['data_group'])
        self.wsi_proc_shape = self.wsi_handler.get_dimensions(self.proc_mag)

        # If the input image is smaller than the patch size, consider as if
        # that was of the patch size instead
        self.wsi_proc_shape = list(map(max, self.wsi_proc_shape, patch_input_shape))
        self.wsi_handler.prepare_reading(
            read_mag=self.proc_mag, cache_path="%s/src_wsi.npy" % self.cache_path
        )
        self.wsi_proc_shape = np.array(self.wsi_proc_shape[::-1])  # to Y, X

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
                scaled_wsi_mag = 1.25  # ! hard coded
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

        if np.sum(self.wsi_mask) == 0:
            log_info("Skip due to empty mask!")
            return
        if self.save_mask:
            cv2.imwrite("%s/mask/%s.png" % (output_dir, wsi_name), self.wsi_mask * 255)
        if self.save_thumb:
            wsi_thumb_rgb = self.wsi_handler.get_full_img(read_mag=1.25)
            cv2.imwrite(
                "%s/thumb/%s.png" % (output_dir, wsi_name),
                cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2BGR),
            )

        # * declare holder for output
        # create a memory-mapped .npy file with the predefined dimensions and dtype
        # TODO: dynamicalize this, retrieve from model?
        out_ch = 3 if self.method["model_args"]["nr_types"] is None else 4
        self.wsi_inst_info = {}
        # TODO: option to use entire RAM if users have too much available, would be faster than mmap
        if ((hasattr(self.wsi_handler.file_ptr, 'store')
            and '.zarr' in self.wsi_handler.file_ptr.store.path)):
            self.wsi_inst_map = zarr.open_array("%s/pred_inst.zarr" % self.cache_path, mode='w',
                                                shape=tuple([1, 1, 1] + list(self.wsi_proc_shape)),
                                                dtype=np.int32,
                                                compressor=Blosc(clevel=5))
            self.wsi_pred_map = zarr.open_array("%s/pred_map.zarr" % self.cache_path, mode='w',
                                                shape=tuple([1, out_ch, 1] + list(self.wsi_proc_shape)),
                                                dtype=np.float32,
                                                compressor=Blosc(clevel=5))
        else:
            # warning, the value within this is uninitialized
            self.wsi_inst_map = np.lib.format.open_memmap(
                "%s/pred_inst.npy" % self.cache_path,
                mode="w+",
                shape=tuple(self.wsi_proc_shape),
                dtype=np.int32,
            )

            self.wsi_pred_map = np.lib.format.open_memmap(
                "%s/pred_map.npy" % self.cache_path,
                mode="w+",
                shape=tuple(self.wsi_proc_shape) + (out_ch,),
                dtype=np.float32,
            )
            # self.wsi_inst_map[:] = 0 # flush fill

        # ! for debug
        # self.wsi_pred_map = np.load('%s/pred_map.npy' % self.cache_path, mmap_mode='r')
        end = time.perf_counter()
        log_info("Preparing Input Output Placement: {0}".format(end - start))

        # * raw prediction
        start = time.perf_counter()
        chunk_info_list, patch_info_list = _get_chunk_patch_info(
            self.wsi_proc_shape,
            chunk_input_shape,
            patch_input_shape,
            patch_output_shape,
        )

        # get the raw prediction of HoVer-Net, given info of inference tiles and patches
        self.__get_raw_prediction(chunk_info_list, patch_info_list)
        end = time.perf_counter()
        log_info("Inference Time: {0}".format(end - start))

        # TODO: deal with error banding
        ##### * post processing
        ##### * done in 3 stages to ensure that nuclei at the boundaries are dealt with accordingly
        start = time.perf_counter()
        tile_coord_set = _get_tile_info(self.wsi_proc_shape, tile_shape, ambiguous_size)
        # 3 sets of patches are extracted and are dealt with differently
        # tile_grid_info: central region of post processing tiles
        # tile_boundary_info: boundary region of post processing tiles
        # tile_cross_info: region at corners of post processing tiles
        tile_grid_info, tile_boundary_info, tile_cross_info = tile_coord_set
        tile_grid_info = self.__select_valid_patches(tile_grid_info, False)
        tile_boundary_info = self.__select_valid_patches(tile_boundary_info, False)
        tile_cross_info = self.__select_valid_patches(tile_cross_info, False)

        ####################### * Callback can only receive 1 arg
        def post_proc_normal_tile_callback(args):
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                pbar.update()  # external
                return  # when there is nothing to do

            top_left = pos_args[1][::-1]

            # ! WARNING:
            # ! inst ID may not be contiguous,
            # ! hence must use max as safeguard

            wsi_max_id = 0
            if len(self.wsi_inst_info) > 0:
                wsi_max_id = max(self.wsi_inst_info.keys())
            for inst_id, inst_info in inst_info_dict.items():
                # now correct the coordinate wrt to wsi
                inst_info["bbox"] += top_left
                inst_info["contour"] += top_left
                inst_info["centroid"] += top_left
                self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
            pred_inst[pred_inst > 0] += wsi_max_id
            if self.wsi_inst_map.ndim > 2:
                self.wsi_inst_map[0, 0, 0,
                    tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
                ] = pred_inst
            else:
                self.wsi_inst_map[
                    tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
                ] = pred_inst

            pbar.update()  # external
            return

        ####################### * Callback can only receive 1 arg
        def post_proc_fixing_tile_callback(args):
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                pbar.update()  # external
                return  # when there is nothing to do

            top_left = pos_args[1][::-1]

            # for fixing the boundary, keep all nuclei split at boundary (i.e within unambigous region)
            # of the existing prediction map, and replace all nuclei within the region with newly predicted

            # ! WARNING:
            # ! inst ID may not be contiguous,
            # ! hence must use max as safeguard

            # ! must get before the removal happened
            wsi_max_id = 0
            if len(self.wsi_inst_info) > 0:
                wsi_max_id = max(self.wsi_inst_info.keys())

            # * exclude ambiguous out from old prediction map
            # check 1 pix of 4 edges to find nuclei split at boundary
            if self.wsi_inst_map.ndim > 2:
                roi_inst = self.wsi_inst_map[0, 0, 0,
                    tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
                ]
            else:
                roi_inst = self.wsi_inst_map[
                    tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
                ]

            roi_inst = np.copy(roi_inst)
            roi_edge = np.concatenate(
                [roi_inst[[0, -1], :].flatten(), roi_inst[:, [0, -1]].flatten()]
            )
            roi_boundary_inst_list = np.unique(roi_edge)[1:]  # exclude background
            roi_inner_inst_list = np.unique(roi_inst)[1:]
            roi_inner_inst_list = np.setdiff1d(
                roi_inner_inst_list, roi_boundary_inst_list, assume_unique=True
            )
            roi_inst = _remove_inst(roi_inst, roi_inner_inst_list)
            if self.wsi_inst_map.ndim > 2:
                self.wsi_inst_map[0, 0, 0,
                    tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
                ] = roi_inst
            else:
                self.wsi_inst_map[
                    tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
                ] = roi_inst
            for inst_id in roi_inner_inst_list:
                self.wsi_inst_info.pop(inst_id, None)

            # * exclude unambiguous out from new prediction map
            # check 1 pix of 4 edges to find nuclei split at boundary
            roi_edge = pred_inst[roi_inst > 0]  # remove all overlap
            boundary_inst_list = np.unique(roi_edge)  # no background to exclude
            inner_inst_list = np.unique(pred_inst)[1:]
            inner_inst_list = np.setdiff1d(
                inner_inst_list, boundary_inst_list, assume_unique=True
            )
            pred_inst = _remove_inst(pred_inst, boundary_inst_list)

            # * proceed to overwrite
            for inst_id in inner_inst_list:
                # ! happen because we alrd skip thoses with wrong
                # ! contour (<3 points) within the postproc, so
                # ! sanity gate here
                if inst_id not in inst_info_dict:
                    log_info("Nuclei id=%d not in saved dict WRN1." % inst_id)
                    continue
                inst_info = inst_info_dict[inst_id]
                # now correct the coordinate wrt to wsi
                inst_info["bbox"] += top_left
                inst_info["contour"] += top_left
                inst_info["centroid"] += top_left
                self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
            pred_inst[pred_inst > 0] += wsi_max_id
            pred_inst = roi_inst + pred_inst
            if self.wsi_inst_map.ndim > 2:
                self.wsi_inst_map[0, 0, 0,
                    tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
                ] = pred_inst
            else:
                self.wsi_inst_map[
                    tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
                ] = pred_inst

            pbar.update()  # external
            return

        #######################
        pbar_creator = lambda x, y: tqdm.tqdm(
            desc=y, leave=True, total=int(len(x)), ncols=80, ascii=True, position=0
        )
        pbar = pbar_creator(tile_grid_info, "Post Proc Phase 1")
        # * must be in sequential ordering
        self.__dispatch_post_processing(tile_grid_info, post_proc_normal_tile_callback)
        pbar.close()

        pbar = pbar_creator(tile_boundary_info, "Post Proc Phase 2")
        self.__dispatch_post_processing(tile_boundary_info, post_proc_fixing_tile_callback)
        pbar.close()

        pbar = pbar_creator(tile_cross_info, "Post Proc Phase 3")
        self.__dispatch_post_processing(tile_cross_info, post_proc_fixing_tile_callback)
        pbar.close()

        end = time.perf_counter()
        log_info("Total Post Proc Time: {0}".format(end - start))

        # ! cant possibly save the inst map at high res, too large
        start = time.perf_counter()
        if self.save_mask or self.save_thumb:
            json_path = "%s/json/%s.json" % (output_dir, wsi_name)
        else:
            json_path = "%s/%s.json" % (output_dir, wsi_name)
        self.__save_json(json_path, self.wsi_inst_info, mag=self.proc_mag)
        end = time.perf_counter()
        log_info("Save Time: {0}".format(end - start))

    def process_wsi_list(self, run_args):
        """Process a list of whole-slide images.

        Args:
            run_args: arguments as defined in run_infer.py
        
        """
        self._parse_args(run_args)

        if not os.path.exists(self.cache_path):
            rm_n_mkdir(self.cache_path)

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
            try:
                log_info("Process: %s" % wsi_base_name)
                self.process_single_file(wsi_path, msk_path, self.output_dir)
                log_info("Finish")
            except:
               logging.exception("Crash")

            if (hasattr(self.wsi_handler.file_ptr, 'store')
              and '.zarr' in self.wsi_handler.file_ptr.store.path):
                wsi_pred_map_mmap_path = "%s/pred_map.zarr" % self.cache_path
                if self.keep_maps and os.path.isdir(wsi_pred_map_mmap_path):
                    source = zarr.open(wsi_pred_map_mmap_path, mode="r")
                    dest_group = zarr.open("%s/%s_pred_map.zarr" % (self.cache_path, wsi_base_name), mode='w')
                    zarr.copy(source, dest_group, '0')

        if self.cache_path != self.output_dir:
            rm_n_mkdir(self.cache_path)  # clean up all cache
        else:
            if os.path.isdir("%s/pred_map.zarr" % self.cache_path):
                shutil.rmtree("%s/pred_map.zarr" % self.cache_path, True)
            if os.path.isdir("%s/pred_inst.zarr" % self.cache_path):
                shutil.rmtree("%s/pred_inst.zarr" % self.cache_path, True)
            if os.path.isfile("%s/pred_map.npy" % self.cache_path):
                os.remove("%s/pred_map.npy" % self.cache_path)
            if os.path.isfile("%s/pred_inst.npy" % self.cache_path):
                os.remove("%s/pred_inst.npy" % self.cache_path)
        return




####
class InferManagerDask(base.InferManager):
    def __save_json(self, path, info_dict, mag=None):
        json_dict = {"mag": mag, "nuc": info_dict}  # to sync the format protocol
        with open(path, "w") as handle:
            json.dump(json_dict, handle)
        return info_dict

    def __select_valid_patches(self, patch_info_list, has_output_info=True):
        """Select valid patches from the list of input patch information.

        Args:
            patch_info_list: patch input coordinate information
            has_output_info: whether output information is given
        
        """
        down_sample_ratio = self.wsi_mask.shape[0] / self.wsi_proc_shape[0]
        selected_indices = []
        for idx in range(patch_info_list.shape[0]):
            patch_info = patch_info_list[idx]
            patch_info = np.squeeze(patch_info)
            # get the box at corresponding mag of the mask
            if has_output_info:
                output_bbox = patch_info[1] * down_sample_ratio
            else:
                output_bbox = patch_info * down_sample_ratio
            output_bbox = np.rint(output_bbox).astype(np.int64)
            # coord of the output of the patch (i.e center regions)
            output_roi = self.wsi_mask[
                output_bbox[0][0] : output_bbox[1][0],
                output_bbox[0][1] : output_bbox[1][1],
            ]
            if np.sum(output_roi) > 0:
                selected_indices.append(idx)
        sub_patch_info_list = patch_info_list[selected_indices]
        return sub_patch_info_list

    def __dispatch_post_processing(self, tile_info_list, callback):
        """Post processing initialisation."""
        proc_pool = None
        if self.nr_post_proc_workers > 0:
            proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

        future_list = []
        if (hasattr(self.wsi_handler.file_ptr, 'store')
           and '.zarr' in self.wsi_handler.file_ptr.store.path):
            wsi_pred_map_mmap_path = "%s/pred_map.zarr" % self.cache_path
            post_proc_wrapper_fun = _post_proc_para_wrapper_zarr
        else:
            wsi_pred_map_mmap_path = "%s/pred_map.npy" % self.cache_path
            post_proc_wrapper_fun = _post_proc_para_wrapper
        for idx in list(range(tile_info_list.shape[0])):
            tile_tl = tile_info_list[idx][0]
            tile_br = tile_info_list[idx][1]

            tile_info = (idx, tile_tl, tile_br)
            func_kwargs = {
                "nr_types": self.method["model_args"]["nr_types"],
                "return_centroids": True,
            }

            # TODO: standarize protocol
            if proc_pool is not None:
                proc_future = proc_pool.submit(
                    post_proc_wrapper_fun,
                    wsi_pred_map_mmap_path,
                    tile_info,
                    self.post_proc_func,
                    func_kwargs,
                )

                # ! manually poll future and call callback later as there is no guarantee
                # ! that the callback is called from main thread
                future_list.append(proc_future)
            else:
                results = post_proc_wrapper_fun(
                    wsi_pred_map_mmap_path, tile_info, self.post_proc_func, func_kwargs
                )
                callback(results)
        if proc_pool is not None:
            silent_crash = False
            # loop over all to check state a.k.a polling
            for future in as_completed(future_list):
                # ! silent crash, cancel all and raise error
                if future.exception() is not None:
                    silent_crash = True
                    # ! cancel somehow leads to cascade error later
                    # ! so just poll it then crash once all future
                    # ! acquired for now
                    # for future in future_list:
                    #     future.cancel()
                    # break
                else:
                    callback(future.result())
            assert not silent_crash
        return

    def _parse_args(self, run_args):
        """Parse command line arguments and set as instance variables."""
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        # to tuple
        self.chunk_shape = [self.chunk_shape, self.chunk_shape]
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
        ambiguous_size = self.ambiguous_size
        tile_shape = (np.array(self.tile_shape)).astype(np.int64)
        chunk_input_shape = np.array(self.chunk_shape)
        patch_input_shape = np.array(self.patch_input_shape)
        patch_output_shape = np.array(self.patch_output_shape)

        path_obj = pathlib.Path(wsi_path)
        wsi_ext = path_obj.suffix
        wsi_name = path_obj.stem

        start = time.perf_counter()
        self.wsi_handler = get_file_handler(wsi_path, backend=wsi_ext, data_group=self.method['data_group'])

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
                scaled_wsi_mag = 1.25  # ! hard coded
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

        if np.sum(self.wsi_mask) == 0:
            log_info("Skip due to empty mask!")
            return
        if self.save_mask:
            cv2.imwrite("%s/mask/%s.png" % (output_dir, wsi_name), self.wsi_mask * 255)
        if self.save_thumb:
            wsi_thumb_rgb = self.wsi_handler.get_full_img(read_mag=1.25)
            cv2.imwrite(
                "%s/thumb/%s.png" % (output_dir, wsi_name),
                cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2BGR),
            )

        self.wsi_inst_info = {}

        # TODO: Process chunks/tiles/patches according to the mask generated
        # above. If the mask is blank at that position, return a zero array
        # instead of the network inference
        offset = (self.patch_input_shape[0] - self.patch_output_shape[0]) // 2
        z_wsi = da.from_zarr(self.wsi_handler._file_path,
                             component=self.method['data_group'] + '/0')

        self.wsi_proc_shape = self.wsi_handler.get_dimensions(self.proc_mag)
        self.wsi_proc_shape = self.wsi_proc_shape[::-1]
        num_patches = [int(math.ceil(s / p))
                       for s, p in zip(self.wsi_proc_shape,
                                       patch_output_shape)]

        paddings = [(0, 0), (0, 0), (0, 0)]
        paddings += [(0, n_p * p - s + 2 * offset)
                     for s, p, n_p in zip(self.wsi_proc_shape,
                                          patch_output_shape,
                                          num_patches)]

        padded_z_wsi = da.pad(z_wsi, paddings, mode='constant')
        padded_z_wsi = padded_z_wsi.rechunk((1, 3, 1, patch_output_shape[0],
                                             patch_output_shape[1]))
        pred_z_wsi = []
        for i in range(num_patches[0]):
            pred_z_wsi.append([])
            row_offset = i * patch_output_shape[0]
            for j in range(num_patches[1]):
                col_offset = j * patch_output_shape[1]
                res = dask.delayed(_infer_patch)(
                    padded_z_wsi[...,
                                 row_offset:row_offset + patch_input_shape[0],
                                 col_offset:col_offset + patch_input_shape[1]],
                    net=self.run_step,
                    wsi_mag=self.wsi_handler.metadata["base_mag"],
                    scaled_wsi_mag=1.25)

                res = dask.delayed(_post_process)(
                    res,
                    nr_types=self.nr_types,
                    tl_pos=(row_offset + offset, col_offset + offset),
                    br_pos=(row_offset + patch_output_shape[0] + offset - 1,
                            col_offset + patch_output_shape[1] + offset - 1))

                res = da.from_delayed(res, shape=(5, 1, 1),
                                      dtype=np.object,
                                      meta=np.empty((0), dtype=np.object))
                pred_z_wsi[-1].append(res)

        pred_z_wsi = da.block(pred_z_wsi)

        dict_pred_z_wsi = pred_z_wsi.map_overlap(
            _fix_hor_boundaries,
            depth=((0, 0), (0, 0), (0, 1)),
            boundary=None,
            trim=False,
            dtype=np.object
        )

        dask.config.set(scheduler='single-threaded')
        with ProgressBar():
            dicts = dict_pred_z_wsi.compute()

        self.wsi_inst_info = reduce(lambda d1, d2: {**d1, **d2},
                                    list(dicts.flatten()),
                                    {})

        all_keys = list(self.wsi_inst_info.keys())
        for i, k in enumerate(all_keys):
            self.wsi_inst_info[i+1] = self.wsi_inst_info.pop(k)

        # ! cant possibly save the inst map at high res, too large
        start = time.perf_counter()
        if self.save_mask or self.save_thumb:
            json_path = "%s/json/%s.json" % (output_dir, wsi_name)
        else:
            json_path = "%s/%s.json" % (output_dir, wsi_name)
        self.__save_json(json_path, self.wsi_inst_info, mag=self.proc_mag)
        end = time.perf_counter()
        log_info("Save Time: {0}".format(end - start))

    def process_wsi_list(self, run_args):
        """Process a list of whole-slide images.

        Args:
            run_args: arguments as defined in run_infer.py
        
        """
        self._parse_args(run_args)

        if not os.path.exists(self.cache_path):
            rm_n_mkdir(self.cache_path)

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

            self.process_single_file(wsi_path, msk_path, self.output_dir)
            # try:
            #     log_info("Process: %s" % wsi_base_name)
            #     self.process_single_file(wsi_path, msk_path, self.output_dir)
            #     log_info("Finish")
            # except:
            #    logging.exception("Crash")

            if (hasattr(self.wsi_handler.file_ptr, 'store')
              and '.zarr' in self.wsi_handler.file_ptr.store.path):
                wsi_pred_map_mmap_path = "%s/pred_map.zarr" % self.cache_path
                if self.keep_maps and os.path.isdir(wsi_pred_map_mmap_path):
                    source = zarr.open(wsi_pred_map_mmap_path, mode="r")
                    dest_group = zarr.open("%s/%s_pred_map.zarr" % (self.cache_path, wsi_base_name), mode='w')
                    zarr.copy(source, dest_group, '0')

        if self.cache_path != self.output_dir:
            rm_n_mkdir(self.cache_path)  # clean up all cache
        else:
            if os.path.isdir("%s/pred_map.zarr" % self.cache_path):
                shutil.rmtree("%s/pred_map.zarr" % self.cache_path, True)
            if os.path.isdir("%s/pred_inst.zarr" % self.cache_path):
                shutil.rmtree("%s/pred_inst.zarr" % self.cache_path, True)
            if os.path.isfile("%s/pred_map.npy" % self.cache_path):
                os.remove("%s/pred_map.npy" % self.cache_path)
            if os.path.isfile("%s/pred_inst.npy" % self.cache_path):
                os.remove("%s/pred_inst.npy" % self.cache_path)
        return
