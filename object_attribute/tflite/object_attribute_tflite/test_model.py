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

    def test__load_model(self):
        for model_name in DEFAULT_MODEL_DIR_PATH_DICT.keys():
            model_dir_path = utils.s3_cp_unzip(DEFAULT_MODEL_DIR_PATH_DICT[model_name]['S3Path'], self.temp_dir.name,
                                               {'VersionId': DEFAULT_MODEL_DIR_PATH_DICT[model_name]['VersionId']})
            tflite_model = model.TfliteModel(model_dir_path)
            self.assertIsNotNone(tflite_model)

    def test__load_model_with_option(self):
        for model_name in DEFAULT_MODEL_DIR_PATH_DICT.keys():
            model_dir_path = utils.s3_cp_unzip(DEFAULT_MODEL_DIR_PATH_DICT[model_name]['S3Path'], self.temp_dir.name,
                                               {'VersionId': DEFAULT_MODEL_DIR_PATH_DICT[model_name]['VersionId']})
            tflite_model = model.TfliteModel(model_dir_path, options={'num_threads': 1})
            self.assertIsNotNone(tflite_model)


if __name__ == "__main__":
    unittest.main()
