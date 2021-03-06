import copy
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
                                ],
                                "nms_iou_th": 0.5,
                                "max_detection": 50
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
                            },
                            {
                              "predicts": [
                                [
                                  10,
                                  20
                                ]
                              ],
                              "key": "detection",
                              "type": "box_num",
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
        self.model_input_shape = self._load_model(model_dir_path, options)

    @abstractmethod
    def _load_model(self, model_dir_path: str, options: Dict) -> Tuple:
        """Load model

        Args:
            model_dir_path: Load model directory path
            options　: Load model options
        Returns:
            (tuple) : A model input shape-(Batch, Height, Width, Channel) array
        """
        raise NotImplementedError()

    @classmethod
    def preprocess(cls, input_tensor: np.ndarray, resize_input_shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """Preprocess

        Args:
            input_tensor (numpy.ndarray) : A shape-(Batch, Height, Width, Channel) array
            resize_input_shape : Resize size (Height, Width)
        Returns:
            (numpy.ndarray) : A shape-(Batch, Height, Width, Channel) array
            (float): A resized scale
        Raises:
            ValueError: If dimension mismatch or dtype mismatch
        """

        if len(input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_tensor.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_tensor.dtype}')

        output_tensor = np.zeros((input_tensor.shape[0], *resize_input_shape, input_tensor.shape[3]),
                                 dtype=input_tensor.dtype)
        resized_scale = min(resize_input_shape[1] / input_tensor.shape[2],
                            resize_input_shape[0] / input_tensor.shape[1])
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
        return output_tensor, (resize_input_shape[1] / resized_scale, resize_input_shape[0] / resized_scale)

    def predict(self, input_tensor: np.ndarray, score_th: float = 0.2, iou_th: float = 0.5, size_th: float = None,
                white_classes_filter: bool = True) -> List[Dict]:
        """Predict

        Args:
            input_tensor (numpy.ndarray) : A shape-(Batch, Height, Width, Channel) array
            score_th: Score threshold. if None, ignore
            iou_th: IoU for nms. if None, ignore
            size_th: Size threshold. if None, 1/max(input_tensor[1], input_tensor[2])
            white_classes_filter: white classes filter.  if None, ignore

        Returns:
            (list): ex. Base.DTO
        """
        resize_input_tensor, resized_scale = self.preprocess(input_tensor,
                                                             (self.model_input_shape[1], self.model_input_shape[2]))
        dto = self._predict(resize_input_tensor)
        dto = self.__filter_limit(dto)
        if score_th is not None:
            dto = self.__filter_by_score(dto, score_th)
        if white_classes_filter:
            dto = self.__filter_by_white_classes(dto)
        if iou_th is not None:
            dto = self.__nms(dto, iou_th)
        if size_th is None:
            size_th = 1 / max((self.model_input_shape[1], self.model_input_shape[2]))
        dto = self.__filter_resized_scale(dto, resized_scale, size_th)
        dto = self.__convert_resized_scale(dto, resized_scale, input_tensor.shape)
        dto = self.__filter_limit(dto)
        return dto

    @abstractmethod
    def _predict(self, resize_input_tensor: np.ndarray) -> List[Dict]:
        """Predict

        Args:
            resize_input_tensor (numpy.ndarray) : A shape-(Batch, Height, Width, Channel) array

        Returns:
            (list): ex. Base.DTO
        """
        raise NotImplementedError()

    @property
    def meta_dict(self):
        return self.__meta_dict

    @classmethod
    def __filter_resized_scale(cls, dto, resized_scale, size_th):
        def __filter(bboxes, scores, classes, box_nums, pixel_th=0):
            if int(box_nums) == 0:
                return bboxes, scores, classes, box_nums
            filter_bboxes = np.zeros(bboxes.shape, bboxes.dtype)
            filter_scores = np.zeros(scores.shape, scores.dtype)
            filter_classes = np.zeros(classes.shape, classes.dtype)
            mask = (np.abs(bboxes[:, 0]-bboxes[:, 2]) > size_th) & (np.abs(bboxes[:, 1]-bboxes[:, 3]) > size_th)
            mask_bboxes, mask_scores, mask_classes = bboxes[:int(box_nums)][mask[:int(box_nums)]], \
                                                     scores[:int(box_nums)][mask[:int(box_nums)]], \
                                                     classes[:int(box_nums)][mask[:int(box_nums)]]
            filter_bboxes[:mask_bboxes.shape[0], :mask_bboxes.shape[1]] = mask_bboxes.astype(filter_bboxes.dtype)
            filter_scores[:mask_scores.shape[0]] = mask_scores.astype(filter_scores.dtype)
            filter_classes[:mask_classes.shape[0]] = mask_classes.astype(filter_classes.dtype)
            filter_box_nums = len(mask_bboxes)
            return filter_bboxes, filter_scores, filter_classes, filter_box_nums

        boxes, scores, classes, box_nums, _ = cls.split_dto(dto)
        filter_boxes_list, filter_scores_list, filter_classes_list, filter_box_nums_list = [], [], [], []
        for index in range(boxes.shape[0]):
            filter_boxes, filter_scores, filter_classes, filter_box_nums = __filter(boxes[index], scores[index],
                                                                                    classes[index], box_nums[index],
                                                                                    size_th)
            filter_boxes_list.append(filter_boxes)
            filter_scores_list.append(filter_scores)
            filter_classes_list.append(filter_classes)
            filter_box_nums_list.append(filter_box_nums)

        dto = cls.__replace_dto(dto, np.asarray(filter_boxes_list), np.asarray(filter_scores_list),
                                np.asarray(filter_classes_list), np.asarray(filter_box_nums_list))
        return dto

    @classmethod
    def __convert_resized_scale(cls, dto, resized_scale, input_tensor_shape):
        boxes, scores, classes, box_nums, _ = cls.split_dto(dto)
        boxes[:, :, 0] = (boxes[:, :, 0] * resized_scale[0]) / input_tensor_shape[1]
        boxes[:, :, 1] = (boxes[:, :, 1] * resized_scale[1]) / input_tensor_shape[2]
        boxes[:, :, 2] = (boxes[:, :, 2] * resized_scale[0]) / input_tensor_shape[1]
        boxes[:, :, 3] = (boxes[:, :, 3] * resized_scale[1]) / input_tensor_shape[2]
        dto = cls.__replace_dto(dto, boxes, scores, classes, box_nums)
        return dto

    @classmethod
    def __filter_limit(cls, dto):
        boxes, scores, classes, box_nums, _ = cls.split_dto(dto)
        boxes = np.minimum(1, np.maximum(boxes, 0))
        dto = cls.__replace_dto(dto, boxes, scores, classes, box_nums)
        return dto

    @classmethod
    def __filter_by_score(cls, dto, score_th=0.2):
        def __filter(bboxes, scores, classes, box_nums, score_th=0.2):
            if int(box_nums) == 0:
                return bboxes, scores, classes, box_nums
            filter_bboxes = np.zeros(bboxes.shape, bboxes.dtype)
            filter_scores = np.zeros(scores.shape, scores.dtype)
            filter_classes = np.zeros(classes.shape, classes.dtype)
            mask = scores > score_th
            mask_bboxes, mask_scores, mask_classes = bboxes[:int(box_nums)][mask[:int(box_nums)]], \
                                                     scores[:int(box_nums)][mask[:int(box_nums)]], \
                                                     classes[:int(box_nums)][mask[:int(box_nums)]]
            filter_bboxes[:mask_bboxes.shape[0], :mask_bboxes.shape[1]] = mask_bboxes.astype(filter_bboxes.dtype)
            filter_scores[:mask_scores.shape[0]] = mask_scores.astype(filter_scores.dtype)
            filter_classes[:mask_classes.shape[0]] = mask_classes.astype(filter_classes.dtype)
            filter_box_nums = len(mask_bboxes)
            return filter_bboxes, filter_scores, filter_classes, filter_box_nums

        boxes, scores, classes, box_nums, _ = cls.split_dto(dto)
        filter_boxes_list, filter_scores_list, filter_classes_list, filter_box_nums_list = [], [], [], []
        for index in range(boxes.shape[0]):
            filter_boxes, filter_scores, filter_classes, filter_box_nums = __filter(boxes[index], scores[index],
                                                                                    classes[index], box_nums[index],
                                                                                    score_th)
            filter_boxes_list.append(filter_boxes)
            filter_scores_list.append(filter_scores)
            filter_classes_list.append(filter_classes)
            filter_box_nums_list.append(filter_box_nums)

        dto = cls.__replace_dto(dto, np.asarray(filter_boxes_list), np.asarray(filter_scores_list),
                                np.asarray(filter_classes_list), np.asarray(filter_box_nums_list))
        return dto

    def __filter_by_white_classes(self, dto):
        def __filter(bboxes, scores, classes, box_nums, white_classes_indexes):
            if int(box_nums) == 0:
                return bboxes, scores, classes, box_nums
            filter_bboxes = np.zeros(bboxes.shape, bboxes.dtype)
            filter_scores = np.zeros(scores.shape, scores.dtype)
            filter_classes = np.zeros(classes.shape, classes.dtype)
            mask = np.zeros(classes.shape, np.bool)
            for white_classes_index in white_classes_indexes:
                mask = mask | (classes.astype(np.int) == white_classes_index)
            mask_bboxes, mask_scores, mask_classes = bboxes[:int(box_nums)][[mask[:int(box_nums)]]], \
                                                     scores[:int(box_nums)][[mask[:int(box_nums)]]], \
                                                     classes[:int(box_nums)][[mask[:int(box_nums)]]]
            filter_bboxes[:mask_bboxes.shape[0], :mask_bboxes.shape[1]] = mask_bboxes.astype(filter_bboxes.dtype)
            filter_scores[:mask_scores.shape[0]] = mask_scores.astype(filter_scores.dtype)
            filter_classes[:mask_classes.shape[0]] = mask_classes.astype(filter_classes.dtype)
            filter_box_nums = len(mask_bboxes)
            return filter_bboxes, filter_scores, filter_classes, filter_box_nums

        boxes, scores, classes, box_nums, white_classes_indexes = self.split_dto(dto)
        filter_boxes_list, filter_scores_list, filter_classes_list, filter_box_nums_list = [], [], [], []
        for index in range(boxes.shape[0]):
            filter_boxes, filter_scores, filter_classes, filter_box_nums = __filter(boxes[index], scores[index],
                                                                                    classes[index], box_nums[index],
                                                                                    white_classes_indexes)
            filter_boxes_list.append(filter_boxes)
            filter_scores_list.append(filter_scores)
            filter_classes_list.append(filter_classes)
            filter_box_nums_list.append(filter_box_nums)

        dto = self.__replace_dto(dto, np.asarray(filter_boxes_list), np.asarray(filter_scores_list),
                                 np.asarray(filter_classes_list), np.asarray(filter_box_nums_list))
        return dto

    @classmethod
    def __nms(cls, dto, iou_th=0.5):
        def __filter(bboxes, scores, classes, box_nums, iou_th=0.5):
            def __iou_np(a, b, a_area, b_area):
                abx_mn = np.maximum(a[0], b[:, 0])
                aby_mn = np.maximum(a[1], b[:, 1])
                abx_mx = np.minimum(a[2], b[:, 2])
                aby_mx = np.minimum(a[3], b[:, 3])
                w = np.maximum(0, abx_mx - abx_mn + 1)
                h = np.maximum(0, aby_mx - aby_mn + 1)
                intersect = w * h

                iou_np = intersect / (a_area + b_area - intersect)
                return iou_np

            if int(box_nums) == 0:
                return bboxes, scores, classes, box_nums
            filter_bboxes = np.zeros(bboxes.shape, bboxes.dtype)
            filter_scores = np.zeros(scores.shape, scores.dtype)
            filter_classes = np.zeros(classes.shape, classes.dtype)
            mask_bboxes, mask_scores, mask_classes = bboxes[:int(box_nums)], scores[:int(box_nums)], classes[
                                                                                                     :int(box_nums)]

            areas = (mask_bboxes[:, 2] - mask_bboxes[:, 0] + 1) * (mask_bboxes[:, 3] - mask_bboxes[:, 1] + 1)

            sort_index = np.argsort(mask_scores)

            i = -1
            while (len(sort_index) >= 2 - i):
                max_scr_ind = sort_index[i]
                ind_list = sort_index[:i]
                iou = __iou_np(mask_bboxes[max_scr_ind], mask_bboxes[ind_list], areas[max_scr_ind], areas[ind_list])
                del_index = np.where(iou >= iou_th)
                sort_index = np.delete(sort_index, del_index)
                i -= 1

            mask_bboxes = mask_bboxes[sort_index][::-1]
            mask_scores = mask_scores[sort_index][::-1]
            mask_classes = mask_classes[sort_index][::-1]

            filter_bboxes[:mask_bboxes.shape[0], :mask_bboxes.shape[1]] = mask_bboxes.astype(filter_bboxes.dtype)
            filter_scores[:mask_scores.shape[0]] = mask_scores.astype(filter_scores.dtype)
            filter_classes[:mask_classes.shape[0]] = mask_classes.astype(filter_classes.dtype)
            filter_box_nums = len(mask_bboxes)
            return filter_bboxes, filter_scores, filter_classes, filter_box_nums

        boxes, scores, classes, box_nums, _ = cls.split_dto(dto)

        filter_boxes_list, filter_scores_list, filter_classes_list, filter_box_nums_list = [], [], [], []
        for index in range(boxes.shape[0]):
            filter_boxes, filter_scores, filter_classes, filter_box_nums = __filter(boxes[index], scores[index],
                                                                                    classes[index], box_nums[index],
                                                                                    iou_th)
            filter_boxes_list.append(filter_boxes)
            filter_scores_list.append(filter_scores)
            filter_classes_list.append(filter_classes)
            filter_box_nums_list.append(filter_box_nums)

        dto = cls.__replace_dto(dto, np.asarray(filter_boxes_list), np.asarray(filter_scores_list),
                                np.asarray(filter_classes_list), np.asarray(filter_box_nums_list))
        return dto

    @classmethod
    def split_dto(cls, dto):
        dto = copy.deepcopy(dto)
        boxes, scores, classes, box_nums, white_classes_indexes = None, None, None, None, None
        for dto_elem in dto:
            if dto_elem['type'] == 'box':
                boxes = dto_elem['predicts']
            elif dto_elem['type'] == 'score':
                scores = dto_elem['predicts']
            elif dto_elem['type'] == 'class':
                classes = dto_elem['predicts']
                white_classes_indexes = [dto_elem['extra']['classes'].index(whilte_class) for whilte_class in
                                         dto_elem['extra']['white_classes']]
            elif dto_elem['type'] == 'box_num':
                box_nums = dto_elem['predicts']
        return boxes, scores, classes, box_nums, white_classes_indexes

    @classmethod
    def __replace_dto(cls, dto, boxes, scores, classes, box_nums):
        dto = copy.deepcopy(dto)
        for dto_elem in dto:
            dto_elem['predicts'] = np.zeros(dto_elem['predicts'].shape, dto_elem['predicts'].dtype)
            if dto_elem['type'] == 'box':
                dto_elem['predicts'] = boxes.astype(dto_elem['predicts'].dtype)
            elif dto_elem['type'] == 'score':
                dto_elem['predicts'] = scores.astype(dto_elem['predicts'].dtype)
            elif dto_elem['type'] == 'class':
                dto_elem['predicts'] = classes.astype(dto_elem['predicts'].dtype)
            elif dto_elem['type'] == 'box_num':
                dto_elem['predicts'] = box_nums.astype(dto_elem['predicts'].dtype)
        return dto
