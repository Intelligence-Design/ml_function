import unittest
import numpy as np

from . import model


class TestUtils(unittest.TestCase):
    def test_calculate_cosine_distance(self):
        query_feature_array = np.array([1.0, 0.0, 0.0])
        dst_feature_array = np.array([[0.0, 1.0, 0.0], [0.1, 0.2, 0.3]])
        distances = model.Base.calculate_cosine_distance(query_feature_array, dst_feature_array)
        self.assertGreater(distances[0], distances[1])

    def test_calculate_euclidean_distance(self):
        query_feature_array = np.array([1.0, 2.0, 3.0])
        dst_feature_array = np.array([[10.0, 20.0, 30.0], [4.0, 5.0, 6.0]])
        distances = model.Base.calculate_euclidean_distance(query_feature_array, dst_feature_array)
        self.assertGreater(distances[0], distances[1])


if __name__ == "__main__":
    unittest.main()
