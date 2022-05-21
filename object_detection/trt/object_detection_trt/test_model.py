import unittest
import tempfile
import numpy as np

from ml_function_utils import utils

from . import model
from . import DEFAULT_MODEL_DIR_PATH_DICT


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _test_model(self, model_name):
        input_tensor = np.random.randint(0, 255, (24, 720, 1280, 3), dtype=np.uint8)
        model_dir_path = utils.s3_cp_unzip(DEFAULT_MODEL_DIR_PATH_DICT[model_name]['S3Path'], self.temp_dir.name,
                                           {'VersionId': DEFAULT_MODEL_DIR_PATH_DICT[model_name]['VersionId']})
        trt_model = model.TrtModel(model_dir_path)
        self.assertIsNotNone(trt_model)
        output = trt_model.predict(input_tensor)
        self.assertIsNotNone(output)

if __name__ == "__main__":
    unittest.main()
