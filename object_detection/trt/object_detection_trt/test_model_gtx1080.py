from . import DEFAULT_MODEL_DIR_PATH_DICT
from .test_model import TestUtils
import unittest

class Gtx1080TestUtils(TestUtils):
    def test_model(self):
        for key in DEFAULT_MODEL_DIR_PATH_DICT.keys():
            model_name = 'geforce_gtx_1080'
            if model_name in key:
                self._test_model(key)

if __name__ == "__main__":
    unittest.main()
