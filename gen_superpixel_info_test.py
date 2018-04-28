import unittest
import numpy as np
from gen_superpixel_info import *

class TestGenBoxImg(unittest.TestCase):
    def setUp(self):
        np.set_printoptions(precision=0, threshold=2000, linewidth=240, edgeitems=50)
        pass

    def test_Medium_AllZeros(self):
        img  = np.zeros((20,20,3), dtype='uint8')
        sps  = SuperPixels(img, 6)
        lbls = sps.labelled_img
        self.assertTrue(np.array_equal(lbls[0:3,0:3], np.zeros((3,3))))

    def test_Medium_SevensOverOnes(self):
        img = np.zeros((20,20,3), dtype='uint8')
        one = np.ones((8,8), dtype='uint8')
        img[3:11,3:11,0] = 1 * one
        img[7:15,7:15,1] = 7 * one

        sps = SuperPixels(img, 6)
        lbls = sps.labelled_img
        self.assertTrue(
            np.array_equal(lbls[0:6,0:6],
                           np.array([[ 0, 0, 0, 1, 1, 2],
                                     [ 0, 0, 0, 1, 1, 2],
                                     [ 0, 0, 0, 1, 1, 2],
                                     [10,10,10,11,11,12],
                                     [10,10,10,11,11,12],
                                     [20,20,20,21,21,22]])))

    def test_Large_SevensOverOnes(self):
        img = np.zeros((40,40,3), dtype='uint8')
        one = np.ones((20,20), dtype='uint8')
        img[2:22,2:22,0]   = 1 * one
        img[15:35,15:35,1] = 7 * one

        sps = SuperPixels(img, 8)
        lbls = sps.labelled_img
        self.assertTrue(
            np.array_equal(lbls[0:6,0:6],
                           np.array([[ 0, 0, 0, 0, 0, 1],
                                     [ 0, 0, 0, 0, 0, 1],
                                     [ 0, 0, 8, 8, 8, 8],
                                     [ 0, 0, 8, 8, 8, 8],
                                     [ 0, 0, 8, 8, 8, 8],
                                     [12,12, 8, 8, 8, 8]])))

    def test_Adjacency(self):
        self.assertTrue(False)

    def tearDown(self):
        pass
unittest.main()
