import math
import sys
import zarr
import json
import numpy as np
from PIL import Image

from skimage import draw


def plotpred(pred, shape, color=None, dst_img=None):
    if color is None:
        color = tuple([np.rint(np.random.rand() * 255 / 4).astype(dtype=np.uint8)] * 3)

    if not isinstance(color, np.ndarray):
        color = np.array(color).reshape(1, 3)

    if dst_img is None:
        dst_img = np.zeros((*shape, 3), dtype=np.uint8)

    for k in pred['nuc'].keys():
        contour = np.array(pred['nuc'][k]['contour'])
        rr, cc = draw.polygon(contour[:, 1], contour[:, 0], shape=shape)

        dst_img[rr, cc, :] = dst_img[rr, cc, :] + color

    return dst_img


def checkpred(pred, shape):
    pred_keys = list(pred.keys())
    n_preds = len(pred_keys)

    overlapped_preds = []
    for i in range(n_preds - 1):
        pred_i = pred[pred_keys[i]]
        pred_area_i = np.zeros(shape, dtype=np.uint8)
        pred_i_contour = np.array(pred_i['contour'])
        pred_area_i[draw.polygon(pred_i_contour[:, 1], pred_i_contour[:, 0], shape=shape)] = 1

        for j in range(i + 1, n_preds):
            pred_j = pred[pred_keys[j]]
            pred_area_j = np.zeros(shape, dtype=np.uint8)
            pred_j_contour = np.array(pred_j['contour'])
            pred_area_j[draw.polygon(pred_j_contour[:, 1], pred_j_contour[:, 0], shape=shape)] = 1
            
            if np.max(pred_area_i + pred_area_j) >= 2:
                overlapped_preds.append((pred_keys[i], pred_keys[j]))

    return overlapped_preds


if __name__ == "__main__":
    log_dir = sys.argv[1]
    img_dir = sys.argv[2]
    img = sys.argv[3]
    patch_size = int(sys.argv[4])
    ambiguous_size = int(sys.argv[5])

    pred_src = json.load(open("%s/%s_zarr.json" % (log_dir, img), "r"))
    pred_ref = json.load(open("%s/%s.json" % (log_dir, img), "r"))

    z = np.moveaxis(zarr.open("%s/%s.zarr" % (img_dir, img), "r")['0/0'][0, :, 0], 0, -1)

    patches_src = plotpred(pred_src, (z.shape[0], z.shape[1]), color=(0, 127, 0), dst_img=z//2)
    patches_ref = plotpred(pred_ref, (z.shape[0], z.shape[1]), color=(0, 127, 0), dst_img=z//2)

    n_chunks_h = z.shape[0] // patch_size
    n_chunks_w = z.shape[1] // patch_size

    diff = patches_src - patches_ref

    diff = Image.fromarray(diff)
    diff.save('%s/patches_diff.jpeg' % log_dir)

    im_ref = Image.fromarray(patches_ref)
    im_ref.save('%s/patches_ref_overlay.jpeg' % log_dir)

    im_src = Image.fromarray(patches_src)
    im_src.save('%s/patches_src_overlay.jpeg' % log_dir)

    patches_src = plotpred(pred_src, (z.shape[0], z.shape[1]), color=(0, 127, 0))
    patches_ref = plotpred(pred_ref, (z.shape[0], z.shape[1]), color=(0, 0, 127))

    for i in range(1, n_chunks_h):
        patches_ref[patch_size * i, :, 0] = 127
        patches_ref[patch_size * i + ambiguous_size, :, 0] = 127
        patches_ref[patch_size * i + ambiguous_size, :, 1] = 127
        patches_ref[patch_size * i - ambiguous_size, :, 0] = 127
        patches_ref[patch_size * i - ambiguous_size, :, 1] = 127

    for i in range(1, n_chunks_w):
        patches_ref[:, patch_size * i, 0] = 127
        patches_ref[:, patch_size * i + ambiguous_size, 0] = 127
        patches_ref[:, patch_size * i + ambiguous_size, 1] = 127
        patches_ref[:, patch_size * i - ambiguous_size, 0] = 127
        patches_ref[:, patch_size * i - ambiguous_size, 1] = 127

    im_ref = Image.fromarray(patches_ref)
    im_ref.save('%s/patches_ref.jpeg' % log_dir)

    im_src = Image.fromarray(patches_src)
    im_src.save('%s/patches_src.jpeg' % log_dir)

    # overlapped_preds_src = checkpred(pred_src['nuc'], (z.shape[0], z.shape[1]))
    # print('Overlapped predictions zarr-based', len(overlapped_preds_src))

    # overlapped_preds_ref = checkpred(pred_ref['nuc'], (z.shape[0], z.shape[1]))
    # print('Overlapped predictions dask-based', len(overlapped_preds_ref))

    # for ovp in overlapped_preds_ref:
    #    print(ovp)
