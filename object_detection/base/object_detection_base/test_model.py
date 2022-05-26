import unittest
import numpy as np

from . import model


class TestUtils(unittest.TestCase):
    def test__preprocess(self):
        input_tensor = np.zeros((8, 640, 1024, 3), dtype=np.uint8)
        resize_input_shape = (320, 320)
        output_tensor = model.BaseModel.preprocess(input_tensor, resize_input_shape)
        self.assertTrue((output_tensor.shape == np.zeros(
            (input_tensor.shape[0], *resize_input_shape, input_tensor.shape[3])).shape))

if __name__ == "__main__":
    unittest.main()
