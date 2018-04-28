import numpy as np
import nnetwork

from skimage.measure import block_reduce
from util import h5str

class pool(nnetwork.nnlayer):
    def __init__(self, h5obj=None):
        super().__init__(h5obj)
        if h5obj is None:
            self.pad    = 1
            self.stride = 1
            self.pool   = 2
            self.method = np.amax
        else:
            self.pad    = int(h5obj['pad'][0,0])     # 4x1 (ones)
            self.stride = int(h5obj['stride'][0,0])  # 2x1 (ones|twos)
            self.pool   = int(h5obj['pool'][0,0])    # 2x1 (twos)
            self.method = {
                'max': np.amax,
                'min': np.amin,
                'avg': np.mean,
            }[h5str(h5obj['method'])]

    @classmethod
    def new(klass, pool, method, pad, stride):
        o = klass(h5obj=None)
        o.pool   = pool
        o.method = method
        o.pad    = pad
        o.stride = stride
        return o

    @property
    def pad(self):
        return self.__pad

    @pad.setter
    def pad(self, p):
        self.__pad = p
        self.pad_ = [
            None,     # x has 0 dimensions
            p,        # x has 1 dimension
            p,        # x has 2 dimensions: pad all (they are all spatial dimensions)
            ((p, p),  # x has 3 dimensions (H,W,C): pad the spatial dimensions only
             (p, p),
             (0, 0)),
            ((p, p),  # x has 4 dimensions (H,W,C,N): pad the spatial dimensions only
             (p, p),
             (0, 0),
             (0, 0))
        ]

    def propagate(self, x, mi):
        x    = np.pad(x, self.pad_[len(x.shape)], mode='constant', constant_values=0)

        # Only the H,W dimensions should be pooled.
        pool = np.ones((len(x.shape),), dtype=int)
        pool[0:2] = 2
        pool = tuple(pool)
        #-pool = tuple(np.full((len(x.shape),), 2, dtype=int))

        if self.pool == self.stride:
            return block_reduce(x, pool, self.method)

        if self.pool == 2 and self.stride == 1:
            a = block_reduce(x,        pool, self.method)
            b = block_reduce(x[:,1:],  pool, self.method)
            c = block_reduce(x[1:,:],  pool, self.method)
            d = block_reduce(x[1:,1:], pool, self.method)

            e = np.empty(x.shape, dtype=x.dtype)
            e[0::2,0::2] = a
            e[0::2,1::2] = b
            e[1::2,0::2] = c
            e[1::2,1::2] = d
            return e[0:-1,0:-1]

        raise Exception('pool(pool=%d stride=%d, pad=%d) is not supported!'
                        % (self.pool, self.stride, self.pad))
