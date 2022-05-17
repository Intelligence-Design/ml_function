from typing import List, Dict, Tuple
from abc import ABCMeta, abstractmethod
import numpy as np
import glob
import os
import json


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
                              "model": [
                                {
                                  "key": "feature",
                                  "type": "metric",
                                  "metric": "cosine",
                                  "dims": 128
                                },
                                {
                                  "key": "gender",
                                  "type": "classification",
                                  "classes": [
                                    "male",
                                    "female",
                                    "unknown"
                                  ]
                                },
                                {
                                  "key": "age",
                                  "type": "classification",
                                  "classes": [
                                    "0_19",
                                    "20_70",
                                    "71_100",
                                    "unknown"
                                  ]
                                }
                              ],
                              "train_repository": "https://github.com/Intelligence-Design/id-object-attribute",
                              "commit_id": "203175f185730fe8c06e740039c56aa53f5cb5b1"
        options　: Load model options

    Attributes:
        __meta_dict: meta info for model
    """

    DTO = [
        {'feature':
            {
                'array': np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
                'type': 'metric',
                'metric': 'cosine'
            }},
        {'gender':
            {
                'array': np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]),
                'type': 'classification',
                'classes': ['0_19', '20_70', '71_100', 'unknown']
            }},
        {'age':
            {
                'array': np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3, ]]),
                'type': 'classification',
                'classes': ['male', 'female', 'unknown']
            }}
    ]

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
    @abstractmethod
    def _preprocess(cls, input_tensor: np.ndarray, resize_input_shape: Tuple[int, int]) -> np.ndarray:
        """Predict

        Args:
            input_tensor (numpy.ndarray) : A shape-(Batch, Height, Width, Channel) array

        Returns:
            (numpy.ndarray) : A shape-(Batch, Height, Width, Channel) array
        """
        raise NotImplementedError()

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
