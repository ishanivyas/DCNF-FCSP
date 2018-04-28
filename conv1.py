import numpy as np
from nnetwork import nnlayer



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
                X: XH,XW,C,N
                F: FW,FH,C,M
            Out:
                Y: YH,YW,M,N
        """
        assert self.filters.shape[2] == x.shape[2]
        S = self.stride
        F = self.filters
        b = self.biases
        P = self.pad

        print("    conv(" + str(x.shape) + ", " + str(F.shape) + ")")
        FH,FW,FC,NF = F.shape
        XH,XW,XC,NX = x.shape
        X           = np.pad(x, P, mode='constant', constant_values=0)

        YH = int((XH - FH + 2*P) / S) + 1
        YW = int((XW - FW + 2*P) / S) + 1

        Y = np.zeros((YH, YW, NF, NX))
        for h in range(YH):
            for w in range(YW):
                for f in range(NF):
                    Y[h,w,f,:] = np.sum(X[h*S:h*S+FH,
                                          w*S:w*S+FW,
                                          0:3,
                                          0]
                                        * F[:,:,:,f],
                                        axis=(0,1,2)) + b[f]
        return Y
