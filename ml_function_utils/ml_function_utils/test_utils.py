import unittest
import tempfile

from . import utils

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
    def tearDown(self):
        self.temp_dir.cleanup()
    def test_s3_cp_unzip(self):
        s3_file_path = 's3://id-ai-function/id_object_attribute/v_0_13_0_id_object_attribute_object_feature_tflite.zip'
        extra_arg = {'VersionId': 'e4mfJtLlbJEQd0v48s0IU3r_xA53a_0P'}
        output_dir_path = utils.s3_cp_unzip(s3_file_path, self.temp_dir.name, extra_arg)
        self.assertIsNotNone(output_dir_path)

if __name__ == "__main__":
    unittest.main()