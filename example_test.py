if __name__ == "__main__":
    import unittest
    class TestCaseX(unittest.TestCase):
        def setUp(self):
            pass

        def test_X(self):
            #-self.assert{True,False,{,Not}{Equal,IsInstance},Is{,Not}{,None,In}}(..., msg=None)
            #-self.assetCountEqual(a,b, msg=None)
            #-self.assert{Set,List,Dict,Tuple,Sequence,MultiLine}Equal(a, b, msg=None)
            #-self.fail(msg=None)
            #-self.run(result=None)
            pass

        def tearDown(self):
            pass

    unittest.main()
