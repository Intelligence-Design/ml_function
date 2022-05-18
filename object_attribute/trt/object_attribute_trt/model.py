from typing import List, Dict
import numpy as np
import glob
import os
import copy
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from object_attribute_base.model import BaseModel


class TrtModel(BaseModel):
    """TrtModel class for object_attribute.

    Args:
        model_dir_path: Load model directory path
                        $ tree model_dir_path/
                            model_dir_path/
                            ├── converted_model.trt
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
                                }
                              ],
                              "input_size": [
                                32,
                                192,
                                192,
                                3
                              ],
                              "train_repository": "https://github.com/Intelligence-Design/id-object-attribute",
                              "commit_id": "beb0734af7cff4d08ddf1725b963eca2f62aea73",
                              "jet-pack": "None",
                              "cuda": "11.2.67",
                              "cudnn": "8_8.2.1.32-1",
                              "tensorrt": "8.2.1.8",
                              "device": "GP104 [GeForce GTX 1080]"
                            }

        options　: Load model options

    Attributes:
        __meta_dict: meta info for model
    """

    def _load_model(self, model_dir_path: str, options: Dict):
        model_file_path = glob.glob(os.path.join(model_dir_path, '**/*.trt'), recursive=True)[0]
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
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
            dto_elem['predicts'] = output_tensor_list[dto_index]
        return dto