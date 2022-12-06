import math
import sys
import zarr
import json
import numpy as np
import matplotlib.pyplot as plt

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
    pred_src = json.load(open("%s/%s_zarr.json" % (log_dir, img), "r"))
    pred_ref = json.load(open("%s/%s.json" % (log_dir, img), "r"))

    z = np.moveaxis(zarr.open("%s/%s.zarr" % (img_dir, img), "r")['0/0'][0, :, 0], 0, -1)

    patches_src = plotpred(pred_src, (z.shape[0], z.shape[1]), color=(127, 127, 127), dst_img=z//2)
    patches_ref = plotpred(pred_ref, (z.shape[0], z.shape[1]), color=(127, 0, 0), dst_img=z//2)

    n_chunks_h = int(math.ceil(z.shape[0] / 328))
    n_chunks_w = int(math.ceil(z.shape[1] / 328))

    plt.imshow(patches_src - patches_ref)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Reference')
    ax1.imshow(patches_src)
    ax2.set_title('Dask')
    ax2.imshow(patches_ref)

    for i in range(1, n_chunks_h):
        ax2.plot([0, z.shape[1]], [328 * i, 328 * i], 'b-')
        ax2.plot([0, z.shape[1]], [328 * i + 64, 328 * i + 64], 'c:')
        ax2.plot([0, z.shape[1]], [328 * i - 64, 328 * i - 64], 'c:')

    for i in range(1, n_chunks_w):
        ax2.plot([328 * i, 328 * i], [0, z.shape[0]], 'b-')
        ax2.plot([328 * i + 64, 328 * i + 64], [0, z.shape[0]], 'c:')
        ax2.plot([328 * i - 64, 328 * i - 64], [0, z.shape[0]], 'c:')

    plt.savefig("%s/patches_comp_overlay.png" % log_dir)
    plt.show()

    patches_src = plotpred(pred_src, (z.shape[0], z.shape[1]), color=(127, 127, 127))
    patches_ref = plotpred(pred_ref, (z.shape[0], z.shape[1]), color=(127, 0, 0))

    plt.imshow(patches_src - patches_ref)

    for i in range(1, n_chunks_h):
        plt.plot([0, z.shape[1]], [328 * i, 328 * i], 'b-')
        plt.plot([0, z.shape[1]], [328 * i + 64, 328 * i + 64], 'c:')
        plt.plot([0, z.shape[1]], [328 * i - 64, 328 * i - 64], 'c:')

    for i in range(1, n_chunks_w):
        plt.plot([328 * i, 328 * i], [0, z.shape[0]], 'b-')
        plt.plot([328 * i + 64, 328 * i + 64], [0, z.shape[0]], 'c:')
        plt.plot([328 * i - 64, 328 * i - 64], [0, z.shape[0]], 'c:')

    plt.savefig("%s/patches_diff.png" % log_dir)
    plt.show()

    plt.subplot(1, 2, 1)
    plt.imshow(patches_src)
    plt.subplot(1, 2, 2)
    plt.imshow(patches_ref)

    for i in range(1, n_chunks_h):
        plt.plot([0, z.shape[1]], [328 * i, 328 * i], 'b-')
        plt.plot([0, z.shape[1]], [328 * i + 64, 328 * i + 64], 'c:')
        plt.plot([0, z.shape[1]], [328 * i - 64, 328 * i - 64], 'c:')

    for i in range(1, n_chunks_w):
        plt.plot([328 * i, 328 * i], [0, z.shape[0]], 'b-')
        plt.plot([328 * i + 64, 328 * i + 64], [0, z.shape[0]], 'c:')
        plt.plot([328 * i - 64, 328 * i - 64], [0, z.shape[0]], 'c:')

    plt.savefig("%s/patches_comp.png" % log_dir)
    plt.show()

    overlapped_preds_src = checkpred(pred_src['nuc'], (z.shape[0], z.shape[1]))
    print('Overlapped predictions zarr-based', len(overlapped_preds_src))

    overlapped_preds_ref = checkpred(pred_ref['nuc'], (z.shape[0], z.shape[1]))
    print('Overlapped predictions dask-based', len(overlapped_preds_ref))

    for ovp in overlapped_preds_ref:
        print(ovp)
