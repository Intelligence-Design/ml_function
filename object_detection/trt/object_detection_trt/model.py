from typing import List, Dict
import numpy as np
import glob
import os
import copy
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from object_detection_base.model import BaseModel


class TrtModel(BaseModel):
    """TrtModel class for object_detection.

    Args:
        model_dir_path: Load model directory path. Ref. BaseModel
        options　: Load model options

    Attributes:
        __meta_dict: meta info for model
    """

    def _load_model(self, model_dir_path: str, options: Dict):
        model_file_path = glob.glob(os.path.join(model_dir_path, '**/*.trt'), recursive=True)[0]
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(model_file_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def predict(self, input_tensor: np.ndarray) -> List[Dict]:
        if len(input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_tensor.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_tensor.dtype}')

        model_input_shape = self.inputs[0]['shape']
        model_input_dtype = self.inputs[0]['dtype']
        resize_input_tensor = self.preprocess(input_tensor, (model_input_shape[1], model_input_shape[2]))
        output_spec_list = self.__output_spec()
        output_tensor_list = [np.zeros((resize_input_tensor.shape[0], *output_spec[0][1:]), output_spec[1]) for output_spec in output_spec_list]
        for index in range(0, input_tensor.shape[0], model_input_shape[0]):
            batch = resize_input_tensor[index:index + model_input_shape[0], :, :, :]
            batch_pad = np.zeros(model_input_shape, model_input_dtype)
            batch_pad[:batch.shape[0], :, :, :] = batch.astype(model_input_dtype)
            output_tensor = self.__predict(batch_pad)
            for tensor_index, output_tensor_elem in enumerate(output_tensor):
                output_tensor_list[tensor_index][index:index+batch.shape[0]] = output_tensor_elem[:batch.shape[0]]
        dto = self.__output_tensor_list2dto(output_tensor_list)
        return dto

    def __predict(self, input_tensor: np.ndarray):
        # Prepare the output data
        outputs = []
        for shape, dtype in self.__output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(input_tensor))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])
        return outputs

    def __output_spec(self):
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def __output_tensor_list2dto(self, output_tensor_list: List[np.ndarray]) -> List[Dict]:
        dto = copy.deepcopy(self.meta_dict['output_dto'])
        for dto_index, dto_elem in enumerate(dto):
            if 'np_ops' in dto_elem['extra'].keys() and dto_elem['extra']['np_ops'] == 'squeeze':
                output_tensor_list[dto_index] = np.squeeze(output_tensor_list[dto_index], axis=-1)
            dto_elem['predicts'] = output_tensor_list[dto_index]
        return dto