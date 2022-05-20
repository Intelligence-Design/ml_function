import unittest
import tempfile
import numpy as np
import time

from ml_function_utils import utils

from . import model
from . import DEFAULT_MODEL_DIR_PATH_DICT


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _test_model(self, model_name):
        input_tensor = np.zeros((40, 640, 1280, 3), dtype=np.uint8)
        model_dir_path = utils.s3_cp_unzip(DEFAULT_MODEL_DIR_PATH_DICT[model_name]['S3Path'], self.temp_dir.name,
                                           {'VersionId': DEFAULT_MODEL_DIR_PATH_DICT[model_name]['VersionId']})
        trt_model = model.TrtModel(model_dir_path)
        self.assertIsNotNone(trt_model)
        output = trt_model.predict(input_tensor)
        self.assertIsNotNone(output)
        for _ in range(10):
            start_time = time.time()
            output = trt_model.predict(input_tensor)
            ent_time = time.time()
            print(f'{ent_time-start_time}[sec]')

if __name__ == "__main__":
    unittest.main()
