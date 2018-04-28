import math
import numpy as np

def gen_box_img_pad(image, sp_mask, box_size):
    # Find the center of the superpixel's bounding box by finding the extrema of
    # the non-zero entries in the superpixel's mask and averaging.
    [rows, cols] = np.nonzero(sp_mask)
    sp_bbox      = [top,left,bottom,right] = [np.amin(rows),np.amin(cols),np.amax(rows),np.amax(cols)]
    sp_bbox_ctr  = [(top+bottom)/2, (left+right)/2]

    # Find the top-left of the box_size area co-centered with the superpixel.
    box_top_lft  = sp_bbox_ctr - box_size/2

    # Padding:
    bounds       = np.ceil([box_top_lft[0],
                            box_top_lft[1],
                            box_top_lft[0]+box_size[0]+1,
                            box_top_lft[1]+box_size[1]+1]
                           ).astype(np.uint16)

    # Return the boxed part of the image and its bounds in the original.
    box_img      = image[bounds[0]:bounds[2],
                         bounds[1]:bounds[3]]
    return [box_img, bounds]

def gen_box_img_nopad(image, sp_mask, box_size):
    # Find the superpixel's bounding box and extract that part of the image.
    [rows, cols] = np.nonzero(sp_mask)
    sp_bbox      = [top,left,bottom,right] = [np.amin(rows),np.amin(cols),np.amax(rows),np.amax(cols)]
    box_img      = image[sp_bbox[0]:(sp_bbox[2]+1),
                         sp_bbox[1]:(sp_bbox[3]+1),
                         :]
    return [box_img, sp_bbox]

################################################################################
# Tests
################################################################################
if __name__ == "__main__":
    import unittest
    class TestGenBoxImg(unittest.TestCase):
        def test_NoPad(self):
            image = np.full((9,9,1), 0)  # All pixels are 0...
            image[4,4,:] = 2             # ... except middle pixel is 2.

            mask = np.zeros((9,9), dtype=np.int8)	     # None of the pixels are masked...
            mask[3:6,3:6] = 1            # ... except indices 3..5 x 3..5.

            img, bbox = gen_box_img_nopad(image, mask, np.array([3, 3]))
            self.assertEqual(img.shape, (3,3,1))
            self.assertEqual(img[0,0],  0)
            self.assertEqual(img[1,1],  2)
            self.assertEqual(bbox,      [3,3,5,5])

        def test_Pad(self):
            image = np.full((9,9,1), 0)  # All pixels are 0...
            image[4,4,:] = 2             # ... except middle pixel is 2.

            mask = np.zeros((9,9), dtype=np.int8)	     # None of the pixels are masked...
            mask[3:6,3:6] = 1            # ... except indices 3..5 x 3..5.

            img, bbox = gen_box_img_pad(image, mask, np.array([3, 3]))
            self.assertEqual(img.shape, (3,3,1))
            self.assertEqual(img[0,0],  0)
            self.assertEqual(img[1,1],  0)  # TODO
            self.assertEqual(bbox,      [3,3,5,5])  # TODO

    unittest.main()
