import math
import numpy as np
import scipy
import matplotlib.image as mpimg
from util import rgb2gray,sparse

def my_fill_depth_colorization(color, depth, centroid_mask, penalty=1):
    """
    Preprocesses the kinect depth image using a gray scale version of the
    RGB image as a weighting for the smoothing. This code is a slight
    adaptation of Anat Levin's colorization code:

        http://www.cs.huji.ac.il/~yweiss/Colorization/

    Args:
      color - HxWx3 matrix; the RGB image, with each component in [0,1].
      depth - HxW matrix; a depth image in absolute (meters) space.
      penalty - a penalty value between 0 and 1 for the current depth values.

    Returns:
      A denoised depth image.
    """
    assert len(depth.shape) == 2
    maxImgAbsDepth = np.amax(depth[centroid_mask])
    depth       = depth / maxImgAbsDepth

    depth[depth > 1] = 1   #-depth[np.where(depth > 1)] = 1
    [H, W] = depth.shape
    numPix = H * W
    indsM  = np.arange(numPix, dtype=int).reshape((H, W))

    grayImg   = rgb2gray(color)
    winRad    = 1
    leng      = 0
    absImgNdx = 0
    cols  = np.zeros((numPix * (2*winRad+1)**2,))   # TODO: may need to remove the +1
    rows  = np.zeros((numPix * (2*winRad+1)**2,))   # TODO: may need to remove the +1
    vals  = np.zeros((numPix * (2*winRad+1)**2,))   # TODO: may need to remove the +1
    gvals = np.zeros((         (2*winRad+1)**2,))   # TODO: may need to remove the +1

    for j in range(W):
        for i in range(H):
            absImgNdx = absImgNdx + 1

            # Count the number of points in the current window.
            nWin = 0
            for ii in     range(max(0, i-winRad), min(i+winRad+1, H)):
                for jj in range(max(0, j-winRad), min(j+winRad+1, W)):
                    if ii == i and jj == j:
                        continue
                    leng += 1
                    nWin += 1
                    rows[leng]  = absImgNdx
                    cols[leng]  = indsM[ii,jj]
                    g           = grayImg[ii, jj]
                    gvals[nWin] = g

            curVal      = grayImg[i, j]
            gvals[nWin] = curVal
            c_var       = np.mean(np.square(gvals[0:nWin] - np.mean(gvals[0:nWin])))
            min_sq_gval = np.amin(np.square(gvals[0:nWin] - curVal))

            csig = max(c_var*0.6, 0.000002, -min_sq_gval/math.log(0.01))

            gvals[0:nWin] = np.exp(-np.square(gvals[0:nWin]-curVal)/csig)
            gvals[0:nWin] = gvals[0:nWin] / np.sum(gvals[0:nWin])
            vals[leng-nWin:leng] = -gvals[0:nWin]

            # Now the self-reference (along the diagonal).
            leng += 1
            rows[leng] = absImgNdx
            cols[leng] = absImgNdx
            vals[leng] = 1 #- sum(gvals[0:nWin])

    rows = rows[0:leng]
    cols = cols[0:leng]
    vals = vals[0:leng]
    A    = sparse(rows, cols, vals, numPix+1, numPix+1)
    print(A.shape)

    rows = range(centroid_mask.size)
    cols = range(centroid_mask.size)
    vals = penalty * centroid_mask.flatten()
    G    = sparse(rows, cols, vals, numPix+1, numPix+1)
    print(G.shape)

    y = (vals.flatten() * depth.flatten())
    y = np.vstack((y.reshape((-1,1)),
                   np.array([[1]])))
    new_vals = scipy.sparse.linalg.lsqr((A + G),
                                        y)[:1]
    new_vals = np.array(new_vals, dtype='float')
    new_vals = new_vals.T
    new_vals = new_vals[:-1]
    return np.reshape(new_vals,(H,W)) * maxImgAbsDepth
