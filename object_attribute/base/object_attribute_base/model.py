from typing import List, Dict, Tuple
from abc import ABCMeta, abstractmethod
import numpy as np
import glob
import os
import json
from PIL import Image


class BaseModel(metaclass=ABCMeta):
    """Base class for object_attribute.

    Args:
        model_dir_path: Load model directory path
                        $ tree model_dir_path/
                            model_dir_path/
                            ├── *.{model:trt, tflite, ...}
                            └── model.json
                        $ cat model_dir_path/model.json
                            {
                              "output_dto": [
                                {
                                  "predicts": [
                                    [
                                      0.1,
                                      0.2,
                                      0.3,
                                      0.45678999,
                                      0.9
                                    ],
                                    [
                                      0.1,
                                      0.2,
                                      0.3,
                                      0.45678999,
                                      0.9
                                    ]
                                  ],
                                  "key": "feature",
                                  "type": "metric",
                                  "extra": {
                                    "metric": "cosine",
                                    "dims": 128
                                  }
                                },
                                {
                                  "predicts": [
                                    [
                                      0.1,
                                      0.2,
                                      0.3
                                    ],
                                    [
                                      0.1,
                                      0.2,
                                      0.3
                                    ]
                                  ],
                                  "key": "gender",
                                  "type": "classification",
                                  "extra": {
                                    "classes": [
                                      "male",
                                      "female",
                                      "unknown"
                                    ]
                                  }
                                },
                                {
                                  "predicts": [
                                    [
                                      0.1,
                                      0.2,
                                      0.3,
                                      0.4
                                    ],
                                    [
                                      0.1,
                                      0.2,
                                      0.3,
                                      0.4
                                    ]
                                  ],
                                  "key": "age",
                                  "type": "classification",
                                  "extra": {
                                    "classes": [
                                      "0_19",
                                      "20_70",
                                      "71_100",
                                      "unknown"
                                    ]
                                  }
                                }
                              ],
                              "input_size": [
                                1,
                                128,
                                128,
                                3
                              ],
                              "train_repository": "https://github.com/Intelligence-Design/id-object-attribute",
                              "commit_id": "203175f185730fe8c06e740039c56aa53f5cb5b1"
                            }
        options　: Load model options

    Attributes:
        __meta_dict: meta info for model
    """

    def __init__(self, model_dir_path: str = None, options: Dict = None):
        meta_json_path = glob.glob(os.path.join(model_dir_path, '**/model.json'), recursive=True)[0]
        with open(meta_json_path, 'r') as f:
            self.__meta_dict = json.load(f)
        self._load_model(model_dir_path, options)

    @abstractmethod
    def _load_model(self, model_dir_path: str, options: Dict):
        """Load model

        Args:
            model_dir_path: Load model directory path
            options　: Load model options
        """
        raise NotImplementedError()

    @classmethod
    def preprocess(cls, input_tensor: np.ndarray, resize_input_shape: Tuple[int, int]) -> np.ndarray:
        """Predict

        Args:
            input_tensor (numpy.ndarray) : A shape-(Batch, Height, Width, Channel) array
            resize_input_shape : Resize size (Height, Width)
        Returns:
            (numpy.ndarray) : A shape-(Batch, Height, Width, Channel) array
        Raises:
            ValueError: If dimension mismatch or dtype mismatch
        """

        if len(input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_tensor.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_tensor.dtype}')

        output_tensor = np.zeros((input_tensor.shape[0], *resize_input_shape, input_tensor.shape[3]),
                                 dtype=input_tensor.dtype)
        for index, image in enumerate(input_tensor):
            pil_image = Image.fromarray(image)
            x_ratio, y_ratio = resize_input_shape[1] / pil_image.width, resize_input_shape[0] / pil_image.height
            if x_ratio < y_ratio:
                resize_size = (resize_input_shape[1], round(pil_image.height * x_ratio))
            else:
                resize_size = (round(pil_image.width * y_ratio), resize_input_shape[0])
            resize_pil_image = pil_image.resize(resize_size)
            output_image = np.array(resize_pil_image)
            output_tensor[index, :output_image.shape[0], :output_image.shape[1], :] = output_image
        return output_tensor

    @abstractmethod
    def predict(self, input_tensor: np.ndarray) -> List[Dict]:
        """Predict

        Args:
            input_tensor (numpy.ndarray) : A shape-(Batch, Height, Width, Channel) array

        Returns:
            (list): ex. Base.DTO
        """
        raise NotImplementedError()

    @property
    def meta_dict(self):
        return self.__meta_dict

    @classmethod
    def calculate_euclidean_distance(cls, query_feature_array: np.ndarray, dst_feature_array: np.ndarray) -> np.ndarray:
        """Calculate euclid distance between src_feature_array and dst_feature_array.

        Args:
            query_feature_array (numpy.ndarray) : A shape-(Dimension, ) array
            dst_feature_array (numpy.ndarray) : A shape-(Batch,Dimension, ) array

        Returns:
            (numpy.ndarray): A shape-(Batch, ) array of the pairwise distances
        """
        distances = np.linalg.norm(dst_feature_array - query_feature_array, axis=1)
        return distances

    @classmethod
    def calculate_cosine_distance(cls, query_feature_array: np.ndarray, dst_feature_array: np.ndarray) -> np.ndarray:
        """Calculate cosine distance between src_feature_array and dst_feature_array.

        Args:
            query_feature_array (numpy.ndarray) : A shape-(Dimension, ) array
            dst_feature_array (numpy.ndarray) : A shape-(Batch, Dimension) array

        Returns:
            (numpy.ndarray): A shape-(Batch, ) array of the pairwise distances
        """

        distances = np.dot(dst_feature_array, query_feature_array) / (
                np.linalg.norm(dst_feature_array, axis=1) * np.linalg.norm(query_feature_array))
        return np.minimum(1., np.maximum(0., 1 - distances))
