import unittest
import numpy as np

from . import model


class TestUtils(unittest.TestCase):
    def test_calculate_cosine_distance(self):
        query_feature_array = np.array([1.0, 0.0, 0.0])
        dst_feature_array = np.array([[0.0, 1.0, 0.0], [0.1, 0.2, 0.3]])
        distances = model.BaseModel.calculate_cosine_distance(query_feature_array, dst_feature_array)
        self.assertGreater(distances[0], distances[1])

    def test_calculate_euclidean_distance(self):
        query_feature_array = np.array([1.0, 2.0, 3.0])
        dst_feature_array = np.array([[10.0, 20.0, 30.0], [4.0, 5.0, 6.0]])
        distances = model.BaseModel.calculate_euclidean_distance(query_feature_array, dst_feature_array)
        self.assertGreater(distances[0], distances[1])

    def test__preprocess(self):
        input_tensor = np.zeros((8, 640, 1024, 3), dtype=np.uint8)
        resize_input_shape = (128, 128)
        output_tensor = model.BaseModel.preprocess(input_tensor, resize_input_shape)
        self.assertTrue((output_tensor.shape == np.zeros(
            (input_tensor.shape[0], *resize_input_shape, input_tensor.shape[3])).shape))

if __name__ == "__main__":
    unittest.main()
