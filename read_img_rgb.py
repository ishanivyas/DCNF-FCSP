import pathlib
from scipy.misc import imread
from util import im2double,im2uint8,ind2rgb,gray2ind

def read_imagefile_into_rgb(filepath, use_uint8=False):
    image = imread(str(filepath), mode='RGB')

    if image.ndim > 3 and image.shape[3] > 1:
        image = image[:,:,:,1]

    if use_uint8:
        image = im2uint8(image)
    else:
        image = im2double(image)

    assert image.shape[2] == 3
    return image
