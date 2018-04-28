import numpy as np
from skimage.segmentation import slic,relabel_sequential
from util                 import im2single

# TODO consider using this to gather per-superpixel data instead of using a
# bunch of parallel arrays below.
class SuperPixel:
    def __init__(self, i):
        self.offsets = None
        self.adjacent_labels = None
        self.adjacent_offsets = None
        self.mean_color = None
        self.local_binary_pattern = None
        self.color_histogram = None

    def mask(self):
        mask = np.zeros((self.offsets.size,))
        mask[self.offsets] = 1
        return mask

    def pixels(self, image):
        """Get the pixels covered by the superpixel."""
        return image.reshape((-1,3))[self.offsets,:]

    def centroid(self):
        centroid_offset = np.mean(self.offsets)
        return centroid_offset

    def mean_color(self, image):
        return np.mean(self.pixels(image), axis=(0))

    def lbp(self, image):
        pass

    def color_hist(self, image):
        p = self.pixels(image)
        h = [None]*3
        for i in range(3):
            h[i] = np.histogram(p, np.arange(0, 255, 255/11)) / p.shape[0]
        return h # TODO return this as a column vector of sorts??


class SuperPixels:
    def __init__(self, image, start_size=16):
        """Segment an image into superpixels."""
        labelled_img, n_superpixels = self.segment(image)
        assert n_superpixels < 2**16
        labelled_img = labelled_img.astype(np.uint16)

        # Create a list if offsets belonging to each superpixel
        superpixel_offsets = [np.flatnonzero(labelled_img == i).astype(np.uint32)
                              for i in range(n_superpixels)]
        # Save the results
        self.image              = image
        self.labelled_img       = labelled_img
        self.n_superpixels      = n_superpixels
        self.superpixel_offsets = superpixel_offsets
        self.adjacent_superpixels = [None]*n_superpixels
        self.adjacent_superpixel_offsets = [None]*n_superpixels

    def mask(self, i):
        """Get a mask of zeros and ones for a specific superpixel."""
        img_size                   = self.labelled_img.shape
        pixel_offsets              = self.superpixel_offsets[i]
        one_sp_mask                = np.zeros(img_size).reshape((-1,1))
        one_sp_mask[pixel_offsets] = 1
        return one_sp_mask.reshape(img_size)

    def segment(self, image):
        """Return a segment-labelled image.
        Labels are from 1..N where N is the resulting number of superpixels found.  One-based
        labelling is done because the `skimage.regionprops' function ignores 0-labels.
        Return [labelled_pixels, N]
        """
        # SuperPixel segment and label from 1..N
        #-seg_map = vl_slic(img_data, /*regionSize:*/sp_size, /*Regularizer:*/0.1, 'MinRegionSize', 10) ;
        lbld_pxls, _, lbls = relabel_sequential(slic(im2single(image), #-compactness=0.1,
                                                     multichannel=True, convert2lab=True,
                                                     slic_zero=True))
        return lbld_pxls, len(lbls)

    def find_adjacencies(self):
        """Determine which superpixels are adjacent to which others."""
        labelled_img   = self.labelled_img
        n_superpixels  = self.n_superpixels

        # Shift the labels one pixel in each direction.
        below        = np.roll(labelled_img, [ 1,  0]) # Rotate entries downward.
        below[ 0, :] = labelled_img[ 0, :]             # Copy first row to avoid interpreting matrix space as torus.
        below        = below.flatten()                 # Flatten so the offsets from flatnonzero will work.
        above        = np.roll(labelled_img, [-1,  0]) # Rotate entries upward.
        above[-1, :] = labelled_img[-1, :]             # Copy last  row to avoid interpreting matrix space as torus.
        above        = above.flatten()                 # Flatten so the offsets from flatnonzero will work.
        left         = np.roll(labelled_img, [ 0,  1]) # Rotate entries rightward.
        left[ :, 0]  = labelled_img[ :, 0]             # Copy first col to avoid interpreting matrix space as torus.
        left         = left.flatten()                  # Flatten so the offsets from flatnonzero will work.
        right        = np.roll(labelled_img, [ 0, -1]) # Rotate entries leftward.
        right[ :,-1] = labelled_img[ :,-1]             # Copy last  col to avoid interpreting matrix space as torus.
        right        = right.flatten()                 # Flatten so the offsets from flatnonzero will work.

        # For each superpixel, find where the other pixels have intruded on its
        # borders due to a shift.
        adjacent_mat = np.full((n_superpixels, n_superpixels), False)
        for i in range(n_superpixels):
            offsets = self.superpixel_offsets[i]
            # Look at this superpixels offsets in the shifted label images and
            # see if any labels are no longer for itself.
            adj     = np.setdiff1d(
                np.array([below[offsets], above[offsets], right[offsets], left[offsets]]),
                np.array([i])
            )

            self.adjacent_superpixels[i] = adj
            adjacent_mat[i,  adj] = True  # i is adjacent to adj.
            adjacent_mat[adj,  i] = True  # If i is adjacent to adj, then adj are all adjacent to i.
            adjacent_mat[i,    i] = False # i is not adjacent to itself.

        # For each superpixel, get a list of the True offsets in the adjacency matrix.
        for i in range(n_superpixels):
            self.adjacent_superpixel_offsets[i] = np.flatnonzero(adjacent_mat[:, i]).astype(np.uint16)	 # Get their indices in the adjacency matrix and save them as an attribute of the list.
