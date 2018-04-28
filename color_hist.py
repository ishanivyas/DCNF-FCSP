import numpy as np

def color_hist(img, mask):
    """Return an array of histograms, one for each color component."""
    histograms = np.array([])
    for ch in range(3):
        channel        = img[:,:,ch]
        ch_histogram,_ = np.histogram(channel[np.where(mask>0)],
                                        np.arange(0, 255, 255/11)) / np.sum(mask.ravel())
        histograms     = np.hstack((histograms, ch_histogram))
    return histograms
