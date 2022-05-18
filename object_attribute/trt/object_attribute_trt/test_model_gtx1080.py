from .test_model import TestUtils
import unittest

class Gtx1080TestUtils(TestUtils):
    def test_model(self):
        model_name = 'debug_geforce_gtx_1080_tensorrt_8.2.1.8_cuda_11.2.67'
        self._test_model(model_name)

if __name__ == "__main__":
    unittest.main()
