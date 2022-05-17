from typing import List, Dict, Tuple
import numpy as np
import glob
import os
import multiprocessing
import time
import copy

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow.lite.python import interpreter as tflite

from object_attribute_base.model import BaseModel


class TfliteModel(BaseModel):
    """TfliteModel class for object_attribute.

    Args:
        model_dir_path: Load model directory path
                        $ tree model_dir_path/
                            model_dir_path/
                            ├── converted_model.tflite
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
                              "input_size": [
                                1,
                                128,
                                128,
                                3
                              ],
                              "train_repository": "https://github.com/Intelligence-Design/id-object-attribute",
                              "commit_id": "203175f185730fe8c06e740039c56aa53f5cb5b1"
        options　: Load model options
                     Example.
                        {'num_threads': 8}

    Attributes:
        __meta_dict: meta info for model
    """

    def _load_model(self, model_dir_path: str, options: Dict):
        if (options is None) or ('num_threads' in list(options.keys())):
            num_thread = multiprocessing.cpu_count()
        else:
            num_thread = options['num_threads']
        model_file_path = glob.glob(os.path.join(model_dir_path, '**/*.tflite'), recursive=True)[0]
        self.interpreter = tflite.Interpreter(model_path=model_file_path, num_threads=num_thread)
        self.interpreter.allocate_tensors()

    def predict(self, input_tensor: np.ndarray) -> List[Dict]:
        if len(input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_tensor.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_tensor.dtype}')

        model_input_shape = self.interpreter.get_input_details()[0]['shape']
        resize_input_tensor = self.preprocess(input_tensor, (model_input_shape[1], model_input_shape[2]))
        output_tensor_list = []
        for resize_input_image in resize_input_tensor:
            self.__set_input_tensor(resize_input_image)
            self.interpreter.invoke()
            output_tensor = self.__get_output_tensor()
            output_tensor_list.append(output_tensor)
        result_dto = self.__output_tensor_list2dto(output_tensor_list)

    def __set_input_tensor(self, image: np.ndarray):
        input_tensor = self.interpreter.tensor(self.interpreter.get_input_details()[0]['index'])()
        input_tensor.fill(0)
        input_image = image.astype(self.interpreter.get_input_details()[0]['dtype'])
        input_tensor[0, :input_image.shape[0], :input_image.shape[1], :input_image.shape[2]] = input_image

    def __get_output_tensor(self) -> List[np.ndarray]:
        output_details = self.interpreter.get_output_details()
        output_tensor = []
        for index in range(len(output_details)):
            output = self.interpreter.get_tensor(output_details[index]['index'])
            scale, zero_point = output_details[index]['quantization']
            output = scale * (output - zero_point)
            output_tensor.append(output)
        return output_tensor

    def __output_tensor_list2dto(self, output_tensor_list: List[List[np.ndarray]]) -> List[Dict]:
        example_dto = copy.deepcopy(self.EXAMPLE_DTO)
        for output_tensor in output_tensor_list:
            print()


