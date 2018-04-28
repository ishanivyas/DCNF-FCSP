import numpy as np
from skimage.measure import regionprops

def gen_sp_centroid(superpixels):
    """Generate the centroid of the superpixels."""
    img_shape     = superpixels.labelled_img.shape
    centroid_mask = np.zeros(img_shape, dtype=int)

    for sp_offs in superpixels.superpixel_offsets:
        sp_mask          = np.zeros(img_shape, dtype=int).reshape((-1,1))
        sp_mask[sp_offs] = 1
        sp_mask          = sp_mask.reshape(img_shape)
        cx,cy            = np.round_(regionprops(sp_mask)[0].centroid, 0).astype(int)

        assert cx < img_shape[1]
        assert cy < img_shape[0]

        centroid_mask[cy, cx] = 1

    return centroid_mask
