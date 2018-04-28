import numpy as np
from skimage.util    import pad
from skimage.feature import local_binary_pattern
from skimage.color   import rgb2luv

from superpixel      import SuperPixels
from gen_box_img_pad import gen_box_img_pad,gen_box_img_nopad
from color_hist      import color_hist
from util            import im2single,im2uint8,rgb2gray,sparse

_N_LBP_ANGLES = 32

def gen_pairwise_feature_diffs(superpixels):
    """Compute several features for the superpixels and then compare them
    using these features.  The features are: mean-color, color-histogram,
    and local-binary-pattern ("LBP", a feature descriptor for textures)."""
    assert type(superpixels) == SuperPixels

    # NOTE: the order should match the pairwise params order in the trained model
    features = [
        gen_feature_diffs(lambda superpixels:
                              gen_sp_texture_descriptors(superpixels, cell_size=8),
                          superpixels, gammas=[50]),
        gen_feature_diffs(lambda superpixels:
                              gen_sp_color_histograms(superpixels, padding=None),
                          superpixels, gammas=[25]),
        gen_feature_diffs(lambda superpixels:
                              gen_sps_mean_color(superpixels, color_space='luv', box_size=0),
                          superpixels, gammas=[.1])
    ]

    pairwise = np.full((1,3),None)
    for k in range(len(features)):
        Fk            = features[k]
        pairwise[0,k] = np.sum(Fk)*np.eye(Fk.shape[0]) - Fk   # TODO is this right??

    return pairwise


def gen_feature_diffs(generate_features_fn, superpixels, gammas):
    """Compare each SuperPixel to its neighbors using a particular kind of feature."""
    gammas             = np.array(gammas)
    n_superpixels      = superpixels.n_superpixels
    img_data           = im2single(superpixels.image)
    feat_data_sps      = generate_features_fn(superpixels) #@
    #-assert gammas.size == 1
    col_indices        = []
    row_indices        = []
    adj_features       = []

    # For each SuperPixel
    for sp_idx in range(n_superpixels):
        adjacent_sp_offs = superpixels.adjacent_superpixel_offsets[sp_idx]
        n_adj            = len(adjacent_sp_offs)
        adj_feat_data    = np.zeros((n_adj))

        # For each adjacent SuperPixel
        for adj_sp_idx_idx in range(n_adj):
            # Compute the norm of the difference of this SPs feature and its adjacent SP features
            adj_sp_idx    = adjacent_sp_offs[adj_sp_idx_idx]
            one_feat_diff = feat_data_sps[sp_idx,:] - feat_data_sps[adj_sp_idx,:]
            one_dist      = np.linalg.norm(one_feat_diff) / np.sqrt(feat_data_sps.shape[1])
            adj_feat_data[adj_sp_idx_idx] = np.exp(-1. * gammas * one_dist)

        # Append to the sparse-array constructor lists
        row_indices  += [sp_idx]*n_adj
        col_indices  += adjacent_sp_offs.tolist()
        adj_features += adj_feat_data.tolist()

        assert np.amin(adj_feat_data) >= 0

    return sparse(row_indices, col_indices, adj_features, n_superpixels, n_superpixels)


def img_in_color_space(image, color_space):
    if color_space == 'rgb':
        return image
    if color_space == 'luv':
        new_cs_img = rgb2luv(image)
        new_cs_img[np.isnan(new_cs_img)] = 0
        return new_cs_img
    raise Exception('Unknown color space: ' + str(color_space))
    return  None


def gen_sps_mean_color(superpixels, color_space='luv', box_size=None):
    """Calculate the average color for each SuperPixel."""
    image = img_in_color_space(superpixels.image, color_space)
    n_sps          = superpixels.n_superpixels
    n_channels     = image.shape[2]
    sp_offsets     = superpixels.superpixel_offsets
    sp_mean_colors = np.zeros((n_sps, n_channels))

    if box_size is None or box_size <= 0:
        for chan_i in range(n_channels):
            chan_img = image[:,:,chan_i].ravel()
            for sp_idx in range(n_sps):
                sp_mean_colors[sp_idx,chan_i] = np.mean(chan_img[sp_offsets[sp_idx]])
    else:
        padding = [box_size, box_size]
        for i in range(n_sps):
            sp_mean_colors[i,:] = np.mean(gen_box_img_nopad(image,
                                                            superpixels.mask(i),
                                                            padding),
                                          axis=(0,1))
    return sp_mean_colors


def gen_sp_texture_descriptors(superpixels, cell_size):
    box_size     = np.array([cell_size, cell_size])
    gray         = rgb2gray(im2single(superpixels.image))
    img_data_pad = pad(gray,
                       cell_size, 'constant', constant_values=0)
    n_sps        = superpixels.n_superpixels

    features = None

    for sp_idx in range(n_sps):
        one_sp_mask       = superpixels.mask(sp_idx)
        image_under_sp, _ = gen_box_img_pad(img_data_pad, one_sp_mask, box_size)
        one_feat          = local_binary_pattern(image_under_sp, _N_LBP_ANGLES,
                                                 cell_size/2, method='uniform').flatten().T
        if features is None:
            features = np.zeros((n_sps, one_feat.shape[0]))

        features[sp_idx,:] = one_feat[:]

    return features


def gen_sp_color_histograms(superpixels, padding=None):
    img     = im2uint8(superpixels.image)
    n_sps   = superpixels.n_superpixels

    features = None
    for sp_idx in range(n_sps):
        sp_mask = superpixels.mask(sp_idx)

        if padding is None:
            feature      = color_hist(img, sp_mask)
        else:
            sp_img       = gen_box_img_nopad(img, sp_mask, np.array([padding, padding]))
            tmp_sp_mask  = np.ones((sp_img.shape[0], sp_img.shape[1]))
            feature      = color_hist(sp_img, tmp_sp_mask)

        if features is None:
            features = np.zeros((n_sps, feature.shape[0]))

        features[sp_idx,:] = feature

    return features
