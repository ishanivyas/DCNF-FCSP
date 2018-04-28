import numpy as np
import numpy.linalg as la

from pathlib      import Path
from h5py         import File as MatFile
from skimage.util import pad

from superpixel                 import SuperPixels
from util                       import im2double,immaxedge,tic,toc,h5str
from do_sp_pooling              import do_sp_pooling
from gen_feature_info_pairwise  import gen_pairwise_feature_diffs
from gen_sp_centroid            import gen_sp_centroid
from my_fill_depth_colorization import my_fill_depth_colorization
from project_back_label         import project_back_label
from read_img_rgb               import read_imagefile_into_rgb
from nnactivelayer              import ActiveLayer

class DCNF_FCSP_Model:
    def __init__(self, model_path, max_img_edge=0, avg_sp_size=16):
        """Load a neural network from a .mat file.
           The structure of the file (in pseudo-JSON format) is: {
            data_obj: {
                image_norm_info: [1],
                label_norm_info: /* a LabelNorm */,
                net: {
                    layers: [...*Lazy* HDF5 Object Reference ...]
                },
            }
           }
        """
        if not model_path:
            raise Exception('No model provided to load.')

        if type(model_path) == str:
            model_path = Path(model_path)

        if not model_path.exists():
            raise Exception('Could not load the model "%s": file does not exist.'
                            % (str(model_path)))

        print('\nLoading the trained model...\n')
        hdf_file             = MatFile(str(model_path), 'r')
        model                = hdf_file['data_obj']

        lzy_lyrs,inorm,lnorm = [model[k] for k in ['net/layers', 'image_norm_info', 'label_norm_info']]
        lyr_obj_refs         = lzy_lyrs[()]  # Reify the lazy HDF5 layers.
        raw_data_for_layers  = [hdf_file[lyr_obj_ref[0]] for lyr_obj_ref in lyr_obj_refs]

        self.layers          = [self.h5layer(lyr) for lyr in raw_data_for_layers]
        self.image_norm      = inorm[0]
        self.label_norm      = LabelNorm(lnorm)

        # TODO move max_img_edge and avg_sp_size into on-disk parameters for the model
        self.max_img_edge    = max_img_edge
        self.avg_sp_size     = avg_sp_size

    def h5layer(self, h5obj):
        """Read a neural network layer from an HDF5 file."""
        typ = h5str(h5obj['type'])
        if typ == 'conv':
            return conv(h5obj)
        elif typ == 'relu':
            return relu(h5obj)
        elif typ == 'pool':
            return pool(h5obj)
        elif typ == 'custom':
            return sp_pooling(h5obj)
        elif typ == 'logistic':
            return logistic(h5obj)
        elif typ == 'structured_loss':
            return structured_loss(h5obj)
        else:
            raise Exception('unknown layer type: ' + typ)
            return None

    def deepen(self, image):
        """Given a depth model and an image, produce an corresponding depth-image."""
        if isinstance(image, Path):
            print('Processing image %s\n' % (str(image)))
            image = read_imagefile_into_rgb(image)

        if self.max_img_edge > 0:
            image = immaxedge(image, self.max_img_edge)

        superpixels = SuperPixels(image, self.avg_sp_size)
        superpixels.find_adjacencies()

        pairwise    = gen_pairwise_feature_diffs(superpixels)
        sp_depths   = self.evaluate(image, superpixels, pairwise)
        return my_fill_depth_colorization(im2double(image),
                                          sp_depths.astype(float),
                                          gen_sp_centroid(superpixels)).astype(float)

    def evaluate(self, image, superpixels, pairwise):
        image      = image.astype(np.float32) - np.full_like(image, self.image_norm, dtype=np.float32)
        actvns     = self.propagate(image, superpixels, pairwise)
        labels     = actvns[-1].labels
        depths_sps = project_back_label(labels.flatten(), self.label_norm)
        return depths_sps[superpixels.labelled_img]

    def propagate(self, image, superpixels, pairwise):
        elapsed  = tic()
        nLayers  = 1 + len(self.layers)
        act      = [ActiveLayer() for i in range(nLayers)]
        act[0].x = image.reshape(image.shape + tuple([1]))
        mi       = ModelInputs(superpixels, pairwise)

        for i in range(nLayers-1):
            lyr = self.layers[i]
            when = tic()
            if lyr.type in ['conv', 'logistic', 'pool', 'relu', 'custom']:
                act[i+1].x = lyr.propagate(act[i].x, mi)
            elif lyr.type == 'structured_loss':
                act[i+1].labels = self.structured_loss(act[i].x, lyr, mi)
            else:
                error('Unknown layer type %s', lyr.type)

            act[i].x = None          # Recover memory.
            act[i].time = toc(when)  # Save time elapsed processing the layer.
            if act[i+1].x is not None:
                print("L[%d]: %s layer '%s'\t produced a %s\t output in %f seconds."
                      % (i, str(lyr.type), str(lyr.name), str(act[i+1].x.shape), act[i].time))
            else:
                print("L[%d]: %s layer '%s'\t produced a %s\t output in %f seconds."
                      % (i, str(lyr.type), str(lyr.name), str(act[i+1].labels.shape), act[i].time))

        print("NN forward propagation completed in %f seconds." % (toc(elapsed)))
        return act

    def structured_loss(self, x, layer, mi):
        n      = mi.superpixels.n_superpixels
        y      = np.squeeze(x[:,:,:,0:n])
        scales = layer.struct_model_w.flatten()
        pws    = mi.pairwise[0,:]

        # Scale the pair-wise feature differences by the 3 elements in scales,
        # add them together, with I.
        A = np.identity(n)
        for i in range(scales.size):
            A += scales[i] * pws[i]

        labels,resids,rnk,s = la.lstsq(A, y)
        return labels.reshape((1,-1))

def my_sp_pooling_forward(_layer, X, mi):
    H,W,n_channels,n_imgs = X.shape
    assert n_imgs == 1
    n_superpixels   = mi.superpixels.n_superpixels
    Y               = np.zeros((1, 1, n_channels, n_superpixels))
    feat_sps_img,_  = do_sp_pooling(X, mi.superpixels)
    feat_sps_img    = feat_sps_img.T
    rows, cols      = feat_sps_img.shape
    assert cols == n_superpixels
    Y[:,:,:,0:n_superpixels] = feat_sps_img.reshape((1,1,rows,cols))
    return Y

# TODO cleanup the passing of these arguments
class ModelInputs:
    def __init__(self, superpixels, pairwise):
        self.superpixels = superpixels
        self.pairwise    = pairwise


class LabelNorm:
    def __init__(self, h5obj=None):
        self.max_d     = h5obj['max_d'][0]        # [1]
        self.min_d     = h5obj['min_d'][0]        # [1]
        self.norm_type = h5obj['norm_type'][0,0]  # 1x1

class nnlayer:
    """Network layer data object."""
    def __init__(self, h5obj=None):
        if h5obj is not None:
            self.type = h5str(h5obj['type'])
            self.name = h5str(h5obj.get('name')) if h5obj.get('name') else None
        else:
            self.type = type(self).__name__
            self.name = None

    def propagate(self, x, _):
        raise Exception('propagate method for ' + str(type(self)) + ' layer is not implemented.')

    def backpropagate(self, x, _):
        raise Exception('backpropagate method for ' + str(type(self)) + ' layer is not implemented.')

# Import layer implementations for convolution and pooling.
from conv2 import conv
from pool1 import pool

class relu(nnlayer):
    def __init__(self, h5obj=None):
        super().__init__(h5obj)

    def propagate(self, x, _):
        x[x<0] = 0
        return x  # OR np.maximum(x, 0, x) OR x*(x>0) OR np.max(x, np.zeros_like(x))

class sp_pooling(nnlayer):
    def __init__(self, h5obj=None):
        super().__init__(h5obj)

    def propagate(self, x, mi):
        return my_sp_pooling_forward(self, x, mi)

class logistic(nnlayer):
    def __init__(self, h5obj=None):
        super().__init__(h5obj)

    def propagate(self, x, _):
        return np.reciprocal(np.ones_like(x) + np.exp(-x))  # 1/(1+e^-x)

class structured_loss(nnlayer):
    def __init__(self, h5obj=None):
        super().__init__(h5obj)
        self.struct_model_w = h5obj['struct_model_w'][()] if h5obj else None  # 1x3

    def propagate(self, x, mi):
        return my_nn_structured_loss(x, self, mi)


################################################################################
# Tests
################################################################################
if __name__ == "__main__":
    from test import fail
    try:
        X = np.array([[ 1,  2,  3,  4,  5,  6,  7,  8],
                      [11, 12, 13, 14, 15, 16, 17, 18],
                      [21, 22, 23, 24, 25, 26, 27, 28],
                      [31, 32, 33, 34, 35, 36, 37, 38],
                      [41, 42, 43, 44, 45, 46, 47, 48],
                      [51, 52, 53, 54, 55, 56, 57, 58],
                      [61, 62, 63, 64, 65, 66, 67, 68],
                      [71, 72, 73, 74, 75, 76, 77, 78]])
        assert np.array_equal(block_reduce(X, (4,4), func=np.amax),
                              np.array([[34, 38],
                                        [74, 78]]))
        assert np.array_equal(block_reduce(X, (2,2), func=np.amax),
                              np.array([[12, 14, 16, 18],
                                        [32, 34, 36, 38],
                                        [52, 54, 56, 58],
                                        [72, 74, 76, 78]]))
        print(__file__ + ": PASS")
    except AssertionError as err:
        fail(err)
