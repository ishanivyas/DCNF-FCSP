import pathlib
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
#-from scipy.misc import imsave as imwrite  # for SciPy <1.2
#-from scipy.imageio import imwrite         # for SciPy >=1.2

def save(depths, filepath, result_dir, model_norm_settings, linear_scale=False):
    depths_ = np.copy(depths)
    mns     = model_norm_settings

    # Use the minimum and maximum value for the norm to take the image out of
    # log-space.
    if mns.norm_type == 2:
        max_d = np.power(np.full_like(mns.max_d, 2), mns.max_d)
        min_d = np.power(np.full_like(mns.min_d, 2), mns.min_d)
    elif mns.norm_type == 3:
        max_d = np.power(np.full_like(mns.max_d, 10), mns.max_d)
        min_d = np.power(np.full_like(mns.min_d, 10), mns.min_d)

    # Decide if we are going to save the images in linear or log space.
    if linear_scale:
        offset  = min_d
        scaling = max_d - min_d
    else:
        offset  = np.log10(min_d)
        scaling = np.log10(max_d) - np.log10(min_d)
        depths_ = np.log10(np.abs(depths_))

    depths_ = (depths_ - offset)/scaling

    # Save the image with a grayscale palette.
    plt.imshow(depths_)
    plt.savefig(result_dir / (filepath + '_gray.png'),
                bbox_inches='tight')

    # Save the image with a good palette.
    depths_ = (depths_ - np.amin(depths_)) / (np.amax(depths_) - np.amin(depths_))
    depths_ = depths_*(64-1) + 1
    depths_ = np.round(depths_, 5)
    plt.imshow(depths_, cmap='viridis')
    plt.savefig(result_dir / (filepath + '_viridis.png'),
                bbox_inches='tight')

    # Save the depth data itself.  TODO
    with h5.File(result_dir / (filepath + '.mat'), 'w') as hdf5_out:
        hdf5_out.create_dataset('depths', data=depths.astype(np.float32))
