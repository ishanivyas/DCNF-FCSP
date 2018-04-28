import numpy as np
from nnetwork import nnlayer

from im2col import *

class conv(nnlayer):
    def __init__(self, h5obj=None):
        super().__init__(h5obj)
        if h5obj is None:
            self.pad = self.stride = self.biases = self.filters = None
            return
        self.pad     = int(h5obj['pad'][0,0])     # 4x1     (ones|zeros)
        self.stride  = int(h5obj['stride'][0,0])  # 2x1,1x1 (ones|twos)
        self.biases  = h5obj['biases'][()]        # 64x1,...
        self.filters = h5obj['filters'][()]       # 64x3x3x3,...

    def propagate(self, x, _):
        """ In:
                X: XH,XW,C,NX
                F: NF,C,FH,FW
                b: NB,1
            Out:
                Y: YH,YW,M,N
        """
        print("conv F=%s, b=%s, p=%s, S=%s"
              % (str(self.filters.shape), str(self.biases.shape),
                 str(self.pad), str(self.stride)))
        x = np.transpose(x, (3,2,0,1))
        y = self.conv(x, self.filters, self.biases, self.pad, self.stride)
        return np.transpose(y, (2, 3, 1, 0))

    def conv(self, x, w, b, pad, stride):
        """
        A fast implementation of the forward pass for a convolutional layer
        based on im2col.
            In:
                X: NX,C,XH,XW
                F: NF,C,FH,FW
            Out:
                Y: ??

        See: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html
        """
        N, C, H, W = x.shape
        if len(w.shape) < 4:
            w = np.expand_dims(w, axis=-1)
        NF, _, FH, FW = w.shape

        # Check dimensions
        #-assert (W + 2 * pad - FW) % stride == 0, 'width does not work'
        #-assert (H + 2 * pad - FH) % stride == 0, 'height does not work'

        # Create output
        YH = int((H + 2 * pad - FH) / stride) + 1
        YW = int((W + 2 * pad - FW) / stride) + 1
        x_cols = im2col_indices(x, FH, FW, pad, stride)
        try:
            wdotx_b = w.reshape((NF, -1)).dot(x_cols) + b.reshape(-1, 1)
            return wdotx_b.reshape(NF, YH, YW, N).transpose(3, 0, 1, 2)
        except:
            # The last layer only has 3 dimensions in the result of the
            # convolution layer.
            wdotx_b = w.reshape((-1, NF)).dot(x_cols) + b.reshape(1, -1)
            return wdotx_b.reshape(1, N, YH, YW)

