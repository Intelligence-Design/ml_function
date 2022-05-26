from typing import Dict
import multiprocessing
import glob
import os
import platform

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow.lite.python import interpreter as tflite

from object_detection_tflite.model import TfliteModel


class CoralModel(TfliteModel):
    def _load_model(self, model_dir_path: str, options: Dict):
        _EDGETPU_SHARED_LIB = {
            'Linux': 'libedgetpu.so.1',
            'Darwin': 'libedgetpu.1.dylib',
            'Windows': 'edgetpu.dll'
        }[platform.system()]
        if (options is None) or ('num_threads' in list(options.keys())):
            num_thread = multiprocessing.cpu_count()
        else:
            num_thread = options['num_threads']
        if (options is None) or ('options' in list(options.keys())):
            tflite_options = None
        else:
            tflite_options = options['options']
        delegates = [tflite.load_delegate(_EDGETPU_SHARED_LIB, options=tflite_options)]
        model_file_path = glob.glob(os.path.join(model_dir_path, '**/*.tflite'), recursive=True)[0]
        self.interpreter = tflite.Interpreter(model_path=model_file_path, experimental_delegates=delegates, num_threads=num_thread)
        self.interpreter.allocate_tensors()

