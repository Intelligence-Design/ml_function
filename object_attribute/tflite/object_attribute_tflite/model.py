from typing import List, Dict
import numpy as np
import glob
import os
import multiprocessing

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

    @classmethod
    def _preprocess(cls, input_tensor: np.ndarray) -> np.ndarray:
        pass

    def predict(self, input_tensor: np.ndarray) -> List[Dict]:
        pass
