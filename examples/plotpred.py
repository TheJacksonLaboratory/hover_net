import os
import zarr
import json
import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt


def plotpred(pred):
    patches = []
    for k in pred['nuc'].keys():
        contour = np.array(pred['nuc'][k]['contour'])
        patches.append(Polygon(contour, True))

    return patches

pred_scr = json.load(open(r"C:\Users\cervaf\Documents\Logging\hover_net\result\TCGA-F9-A97G-01Z-00-DX1_1.json", "r"))
pred_ref = json.load(open(r"C:\Users\cervaf\Documents\Logging\hover_net\result\TCGA-F9-A97G-01Z-00-DX1_1_wsi.json", "r"))
pred_cmp_prj = json.load(open(r"C:\Users\cervaf\Documents\Logging\hover_net\result\TCGA-F9-A97G-01Z-00-DX1_1_comp.json", "r"))
pred_cmp_res = json.load(open(r"C:\Users\cervaf\Documents\Logging\hover_net\result\TCGA-F9-A97G-01Z-00-DX1_1_comp_res.json", "r"))

z = zarr.open(r"C:\Users\cervaf\Documents\Logging\hover_net\zarr\TCGA-F9-A97G-01Z-00-DX1_1.zarr")

patches_scr = plotpred(pred_scr)
patches_ref = plotpred(pred_ref)
patches_cmp_prj = plotpred(pred_cmp_prj)
patches_cmp_res = plotpred(pred_cmp_res)

patches_scr = PatchCollection(patches_scr, alpha=0.25)
patches_ref = PatchCollection(patches_ref, alpha=0.25)
patches_cmp_prj = PatchCollection(patches_cmp_prj, alpha=0.25)
patches_cmp_res = PatchCollection(patches_cmp_res, alpha=0.25)

patches_scr.set_color([1, 0, 0])
patches_ref.set_color([1, 1, 0])
patches_cmp_prj.set_color([0, 1, 1])
patches_cmp_res.set_color([1, 0, 1])

fig, ax = plt.subplots()

ax.imshow(np.moveaxis(z['0/0'][0, :, 0], 0, -1))

ax.add_collection(patches_scr)
ax.add_collection(patches_ref)
ax.add_collection(patches_cmp_prj)
ax.add_collection(patches_cmp_res)

plt.show()
