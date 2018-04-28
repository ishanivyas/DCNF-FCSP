import numpy as np
from numpy.testing import *
from nnetwork import *
from pool1 import *
if __name__ == "__main__":
    import unittest

    class ArrayAssertions:
        def assertArraysAreEqual(self, expected, actual):
            assert_array_equal(expected, actual)

    class NumPyTest(unittest.TestCase):
        def setUp(self):
            super().setUp()
            # Allow printing integer arrays up to 115w by 70h:
            np.set_printoptions(precision=0, threshold=9999, linewidth=260, edgeitems=50)

    class TestCaseData(NumPyTest):
        def setUp(self):
            super().setUp()
            # Sample data.  Start sequences from 1 so that numbers from the
            # sequence are distinguishable from the 0s adding during padding.
            self.f1x1     = np.arange(1, 1 + 1*1).reshape((1,1))
            self.f2x2     = np.arange(1, 1 + 2*2).reshape((2,2))
            self.f8x8     = np.arange(1, 1 + 8*8).reshape((8,8))
            self.f9x9     = np.arange(1, 1 + 9*9).reshape((9,9))

            self.f8x8x1   = np.arange(1, 1 + 8*8).reshape((8,8,1))
            self.f9x9x1   = np.arange(1, 1 + 9*9).reshape((9,9,1))

            self.f8x8x3x1 = np.arange(1, 1 + 8*8*3).reshape((8,8,3,1))
            self.f9x9x3x1 = np.arange(1, 1 + 9*9*3).reshape((9,9,3,1))

    class TestPoolLayer(TestCaseData,ArrayAssertions):
        def test_MinPool_Pool2Stride1Pad1_2d1ImplicitChannel(self):
            p = pool.new(pool=2, method=np.amin, pad=1, stride=1) # 0 0 0      0 0

            a = p.propagate(self.f1x1, None)                      # 0 1 0 =p=> 0 0
            self.assertArraysAreEqual(np.zeros((2,2)), a)         # 0 0 0

            b = p.propagate(self.f2x2, None)               # 0 0 0 0      0 0 0
            self.assertArraysAreEqual(np.array([[0,0,0],   # 0 1 2 0 =p=> 0 1 0
                                                [0,1,0],   # 0 3 4 0      0 0 0
                                                [0,0,0]]), # 0 0 0 0
                                      b)

        def test_MaxPool_Pool2Stride1Pad1_2d1ImplicitChannel(self):
            p = pool.new(pool=2, method=np.amax, pad=1, stride=1)

            a = p.propagate(self.f1x1, None)
            self.assertArraysAreEqual(np.ones((2,2)), a)

            b = p.propagate(self.f2x2, None)
            self.assertArraysAreEqual(np.array([[1,2,2],   # a[-2] = a[-1]
                                                [3,4,4],   # a[-2] = a[-1]
                                                [3,4,4]]), # a[-2] = a[-1]
                                      b)       # ^-- a[i+1] = a[i] + 2; except: a[-2] = a[-1]

            x = p.propagate(self.f8x8, None)
            self.assertArraysAreEqual(np.array([1,2,3,4,5,6,7,8,8]),        # a[i+1] = a[i] + 1; except: a[-2] = a[-1]
                                      x[0,:])
            self.assertArraysAreEqual(np.array([1,9,17,25,33,41,49,57,57]), # a[i+1] = a[i]+8; except: a[-2] = a[-1]
                                      x[:,0].flatten())

            z = p.propagate(self.f9x9, None)
            self.assertArraysAreEqual(np.array([1,2,3,4,5,6,7,8,9,9]),          # a[i+1] = a[i] + 1; except: a[-2] = a[-1]
                                      z[0,:])
            self.assertArraysAreEqual(np.array([1,10,19,28,37,46,55,64,73,73]), # a[i+1] = a[i]+9; except: a[-2] = a[-1]
                                      z[:,0].flatten())

        def test_MaxPool_Pool2Stride2Pad1_2d1ImplicitChannel(self):
            p = pool.new(pool=2, method=np.amax, pad=1, stride=2)

            x = p.propagate(self.f8x8, None)
            z = p.propagate(self.f9x9, None)
            self.assertEqual((5,5), x.shape)
            self.assertEqual((6,6), z.shape)

        def test_MaxPool_Pool2Stride2Pad1_2d1ExplicitChannel(self):
            p = pool.new(pool=2, method=np.amax, pad=1, stride=2)

            x = p.propagate(self.f8x8x1, None)
            z = p.propagate(self.f9x9x1, None)
            self.assertEqual((5,5,1), x.shape)
            self.assertEqual((6,6,1), z.shape)

        def test_MaxPool_Pool2Stride1Pad1_2d3chan1img(self):
            p = pool.new(pool=2, method=np.amax, pad=1, stride=1)

            y = p.propagate(self.f8x8x3x1, None)
            w = p.propagate(self.f9x9x3x1, None)
            self.assertEqual(( 9, 9,3,1), y.shape)
            self.assertEqual((10,10,3,1), w.shape)

        def test_MaxPool_Pool2Stride2Pad1_2d3chan1img(self):
            p = pool.new(pool=2, method=np.amax, pad=1, stride=2)

            y = p.propagate(self.f8x8x3x1, None)
            w = p.propagate(self.f9x9x3x1, None)
            self.assertEqual((5,5,3,1), y.shape)
            self.assertEqual((6,6,3,1), w.shape)

        def tearDown(self):
            pass

    unittest.main()
