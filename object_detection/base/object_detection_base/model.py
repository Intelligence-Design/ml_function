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
                                [
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
                                ]
                              ],
                              "key": "detection",
                              "type": "box",
                              "extra": {
                                "details": [
                                  "y1_ratio",
                                  "x1_ratio",
                                  "y2_ratio",
                                  "x2_ratio"
                                ]
                              }
                            },
                            {
                              "predicts": [
                                [
                                  1,
                                  2,
                                  3
                                ],
                                [
                                  1,
                                  2,
                                  3
                                ]
                              ],
                              "key": "detection",
                              "type": "class",
                              "extra": {
                                "classes": [
                                  "car",
                                  "person",
                                  "unknown"
                                ],
                                "white_classes": [
                                  "person"
                                ]
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
                              "key": "detection",
                              "type": "score",
                              "extra": {
                              }
                            }
                          ],
                          "input_size": [
                            1,
                            320,
                            320,
                            3
                          ],
                          "train_repository": "shttps://github.com/Intelligence-Design/id-object-detection",
                          "commit_id": "1cf53ce0311be9fddf6199cbd3e4bfad8cb1f920"
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