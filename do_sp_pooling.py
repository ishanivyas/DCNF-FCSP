import numpy as np
#-from scipy.misc import imresize # deprecated
from skimage.transform import resize  # Replaces scipy.misc.imresize

def do_sp_pooling(feat_image, superpixels):
    """Pool values from the pixels in the superpixels, returning the feature image weighted..."""
    print("do_sp_pooling: feat_image.shape = " + str(feat_image.shape))
    h,w,dim,_         = feat_image.shape
    n_feat_pixels     = h*w

    feat_image        = feat_image.reshape((n_feat_pixels, dim))  # Flatten the spatial dimensions.
    orig_img_size     = superpixels.image.shape

    # Create a map of pixels to the corresponding flat-offset in feat_image
    pix2feat_offs     = resize(np.arange(n_feat_pixels).reshape((h, w)),
                               superpixels.labelled_img.shape,
                               mode='constant').flatten()
    n_sps             = superpixels.n_superpixels
    weight_pool       = np.zeros((n_sps, n_feat_pixels))  # Matlab code uses same sparseness as feat_image but different dimensions

    for sp_idx in range(n_sps):
        sp_offs_in_img          = superpixels.superpixel_offsets[sp_idx] # Get the superpixel's offsets in the original image.
        sp_feat_offs            = pix2feat_offs[sp_offs_in_img]          # Convert the offsets to offsets in the feature image.

        # Count the number of times each feature-pixel is part of the superpixel.
        [uniq_offs, uniq_off_counts] = np.unique(sp_feat_offs, return_counts=True)
        uniq_offs               = uniq_offs.astype(int)
        frequencies             = uniq_off_counts.astype(np.float32) / sp_feat_offs.size
        sp_frequency            = np.zeros((1, n_feat_pixels), dtype=float)
        sp_frequency[uniq_offs] = frequencies.reshape(-1,1)

        # Store their frequencies for weighting the feature image.
        weight_pool[sp_idx,:]   = sp_frequency

    return [weight_pool.dot(feat_image), weight_pool]
