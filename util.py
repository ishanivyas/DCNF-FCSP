import numpy as np
from scipy.sparse import coo_matrix
from time import time
from skimage.transform import resize,rescale  # for SciPy 1.2 and above

#-from scipy.io import {save,load}mat
#-from scipy.misc import imresize,imsave,imread,imshow
#-from scipy.imageio import imresize,imwrite,imread,imshow  # for SciPy 1.2 and above

#-from skimage.io import imshow,imsave,imread
#-from skimage.util import img_as_{float{,32,64},{,u}int,ubyte,bool},pad,crop  (some of these are in skimage.util (*_as_*))
from skimage.util import img_as_ubyte
#-from skimage.feature import local_binary_pattern
#-from skimage.segmentation import slic
#-from skimage.restoration import inpaint_biharmonic

# http://scikit-image.org/docs/dev/api/skimage.color.html#module-skimage.util
#-from skimage.color import convert_colorspace,{hsv,xyz,rgbcie,gray,lab,hed,yuv}2rgb,rgb2{hsv,xyz,rgbcie,gray,lab,hed,yuv},...

def tic():
    return time()

def toc(since):
    return tic() - since

def obj(**kwargs):
    return namedtuple('obj', kwargs)(**kwargs)

def h5str(h5u8obj):
    """Read a string from a HDF5 object."""
    return ''.join(chr(i) for i in h5u8obj[:])

def h5vecu16(h5vobj):
    """Read an (N,1,...1) array as (N,) and convert to uint16."""
    return h5vobj[()].squeeze().astype(np.uint16)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [.299, .587, .114])

def immaxedge(image, max_edge):
    """Resize an image so that its longest edge is equal to max_edge."""
    height, width = image.shape[0:2]
    scale = 1.0
    if height > width:
        if height > max_edge:
            scale = (1. * max_edge) / height
    elif width > max_edge:
            scale = (1. * max_edge) / width

    if scale == 1.0:
        return image

    return rescale(image, scale, mode='constant') #-, multichannel=True, anti_aliasing=True

def im2single(im):
    """Rescale image so pixels are in the range [0,1], single precision."""
    min_px = np.amin(im)
    max_px = np.amax(im)
    return (im.astype(np.float32) - min_px) / max(1, max_px - min_px)

def im2double(im):
    """Rescale the image so the pixels are in the range [0,1]"""
    min_px = np.amin(im)
    max_px = np.amax(im)
    return (im.astype(np.float64) - min_px) / max(1, max_px - min_px)

def im2uint8(im):
    return img_as_ubyte(im)

def ind2rgb(img):
    raise "Unimplemented"  # TODO

def gray2ind(img):
    raise "Unimplemented"  # TODO

def clamp01(m):
    """Clamp the values of an array so they lie in [0..1]."""
    s = m.shape
    return np.minimum(np.maximum(np.zeros(s), m), np.ones(s))

def dense(m):
    return m.toarray()

def sparse(i, j, v, w=None, h=None):
    """
    Produce a sparse array using a parallel set of coordinates and values.  If a
    coordinate is repeated, its value should be the sum of the observed values
    for that coordinate.

    i: x coordinates
    j: y coordinates
    v: values to put at the x,y coords
    w: the width of the matrix
    h: the height of the matrix
    """
    if w and h:
        return coo_matrix((v,(i,j)),
                          shape=(w,h))
    return coo_matrix((v,(i,j)))

class literal:
    """A class used for easily storing and updating somewhat-arbitrary name-value pairs.

    Example:
      class foo(literal):
        def __init__(self, **kwargs):
          self.sna   = 'bar'
          self.fnord = None
          ...
          self.set(kwargs)

      foo(fnord='baz', ...)  # Make a foo initialized as default, but with fnord='baz'
    """
    def __init__(self, **kwargs):
        pass

    def set(self, kvargs):
        #-for k in ['res','sync','conserveMemory','disableDropout',
        #-          'freezeDropout','keep_layer_output','ds_info']:
        #-    if kwargs.get(k) is not None:
        #-        setattr(self, k, kwargs[k])
        self.__dict__.update(kvargs)


class options(literal):
    """Options (unused)."""
    def __init__(self, **kwargs):
        self.sync = False
        self.set(kwargs)


################################################################################
# Tests
################################################################################
if __name__ == "__main__":
    from test import fail
    try:
        assert options(sync=True).sync == True
        print(__file__ + ": PASS")
    except AssertionError as err:
        fail(err)
