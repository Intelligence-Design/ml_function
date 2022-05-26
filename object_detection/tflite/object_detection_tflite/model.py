from typing import List, Dict
import numpy as np
import glob
import os
import multiprocessing
import copy

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow.lite.python import interpreter as tflite

from object_detection_base.model import BaseModel


class TfliteModel(BaseModel):
    """TfliteModel class for object_detection.

    Args:
        model_dir_path: Load model directory path. Ref. BaseModel
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
        return self.interpreter.get_input_details()[0]['shape']

    def _predict(self, resize_input_tensor: np.ndarray) -> List[Dict]:
        if len(resize_input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(resize_input_tensor.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {resize_input_tensor.dtype}')

        output_tensor_list = []
        for resize_input_image in resize_input_tensor:
            self.__set_input_tensor(resize_input_image)
            self.interpreter.invoke()
            output_tensor = self.__get_output_tensor()
            output_tensor_list.append(output_tensor)

        dto = self.__output_tensor_list2dto(output_tensor_list)
        return dto

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
            if scale > 1e-4:
                output = scale * (output - zero_point)
            output_tensor.append(output)
        return output_tensor

    def __output_tensor_list2dto(self, output_tensor_list: List[List[np.ndarray]]) -> List[Dict]:
        dto = copy.deepcopy(self.meta_dict['output_dto'])
        for dto_index, dto_elem in enumerate(dto):
            dto_elem['predicts'] = np.zeros((len(output_tensor_list), *output_tensor_list[0][dto_index].shape[1:]),
                                            dtype=output_tensor_list[0][dto_index].dtype)

        concatenate_output_tensor_list = []
        for dto_index in range(len(dto)):
            concatenate_output_tensor_list_elem = []
            for output_tensor in output_tensor_list:
                concatenate_output_tensor_list_elem.extend(output_tensor[dto_index])
            concatenate_output_tensor_list.append(np.array(concatenate_output_tensor_list_elem))

        for dto_index, dto_elem in enumerate(dto):
            dto_elem['predicts'] = concatenate_output_tensor_list[dto_index]
        return dto
