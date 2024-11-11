from collections import deque
from typing import Tuple
from nptyping import NDArray
from loguru import logger
import numpy as np


class VisionSensorSettings:
    def __init__(self,
                 preview_width:int = 800,
                 camera_center_position:int = 430,
                 stop_position:int = 250,
                 edge_detection_range: int = 12,
                 film_type_is_negative: bool = True,
                 slope_threshold: float = 30.0,
                 contrast_offset: float = 30.0,
                 tile_center_offset:int = 50,
                 tile_width:int = 300,
                 tile_height:int = 350,
                 contrast_pic_height:int = 20,
                 contrast_pic_edge_offset:int = 10
                 ):
        self._preview_width = preview_width
        self._camera_center_position = camera_center_position
        self.stop_position = stop_position
        self._edge_detection_range = edge_detection_range
        self._film_type_is_negative = film_type_is_negative
        self._slope_threshold = slope_threshold
        self._contrast_offset = contrast_offset
        self._contrast_pic_height = contrast_pic_height
        self._contrast_pic_offset = contrast_pic_edge_offset
        self._tile_center_offset = tile_center_offset
        self._tile_width = tile_width
        self._tile_height = tile_height

    @property
    def tile_center_offset(self):
        return self._tile_center_offset

    @tile_center_offset.setter
    def tile_center_offset(self, value:int):
        self._tile_center_offset = value
        if (self._tile_center_offset + self._tile_width) > (self._preview_width // 2):
            self._tile_width = (self._preview_width // 2) - self._tile_center_offset
        logger.debug(f"set tile_center_offset to: {self._tile_center_offset}")

    @property
    def tile_width(self):
        return self._tile_width

    @tile_width.setter
    def tile_width(self, value:int):
        self._tile_width = value
        if (self._tile_width + self._tile_center_offset) > (self._preview_width // 2):
            self._tile_center_offset = (self._preview_width // 2) - self._tile_width
        logger.debug(f"set tile_width to: {self._tile_width}")

    @property
    def tile_height(self):
        return self._tile_height

    @tile_height.setter
    def tile_height(self, value: int):
        self._tile_height = value
        logger.debug(f"set tile_height to: {self.tile_height}")

    @property
    def preview_width(self):
        return self._preview_width

    @preview_width.setter
    def preview_width(self, value:int):
        self._preview_width = value
        if (self._tile_width + self._tile_center_offset) > (self._preview_width // 2):
            self._tile_width = (self._preview_width // 2) - self._tile_center_offset
        logger.debug(f"set preview_width to: {self._preview_width}")

    @property
    def slope_threshold(self):
        return self._slope_threshold

    @slope_threshold.setter
    def slope_threshold(self, value):
        self._slope_threshold = value
        logger.debug(f"set slope_threshold to: {self._slope_threshold}")

    @property
    def contrast_offset(self):
        return self._contrast_offset

    @contrast_offset.setter
    def contrast_offset(self, value):
        self._contrast_offset = value
        logger.debug(f"set contrast_offset to: {self._contrast_offset}")

    @property
    def contrast_pic_height(self):
        return self._contrast_pic_height

    @contrast_pic_height.setter
    def contrast_pic_height(self, value):
        self._contrast_pic_height = value
        logger.debug(f"set contrast_pic_height to: {self._contrast_pic_height}")

    @property
    def contrast_pic_offset(self):
        return self._contrast_pic_offset

    @contrast_pic_offset.setter
    def contrast_pic_offset(self, value):
        self._contrast_pic_offset = value
        logger.debug(f"set contrast_pic_offset to: {self._contrast_pic_offset}")

    @property
    def camera_center_position(self):
        return self._camera_center_position

    @camera_center_position.setter
    def camera_center_position(self, value):
        self._camera_center_position = value
        logger.debug(f"set image_center_position to: {self._camera_center_position}")

    @property
    def edge_detection_range(self):
        return self._edge_detection_range

    @edge_detection_range.setter
    def edge_detection_range(self, value):
        self._edge_detection_range = value
        logger.debug(f"set edge_detection_range to: {self._edge_detection_range}")

    @property
    def film_type_is_negative(self):
        return self._film_type_is_negative

    @film_type_is_negative.setter
    def film_type_is_negative(self, value):
        self._film_type_is_negative = value
        logger.debug(f"set edge_detection_range to: {self._film_type_is_negative}")


class CalcEdgeSlopeParameter:
    def __init__(self):
        self.image_data = None
        self.slope_film_pos:float = 0.0
        self.slope_film_neg:float = 0.0
        self.pos_film_pos:int = 0
        self.pos_film_neg:int = 0
        self.slope_data = []

    def calculate(self):
        if self.image_data is not None:
            reduced = np.mean(self.image_data, axis=1)
            self.slope_data = [(reduced[i + 3] - reduced[i - 3]) for i in range(3, len(reduced) - 3, 1)]

            self.slope_film_pos = min(self.slope_data)
            self.slope_film_neg = max(self.slope_data)
            self.pos_film_pos = self.slope_data.index(self.slope_film_pos)
            self.pos_film_neg = self.slope_data.index(self.slope_film_neg)


class CalculateContrast:
    def __init__(self, vs_settings:VisionSensorSettings):
        self.in_pic_contr_tile_left:float = 0.0
        self.in_pic_contr_tile_right:float = 0.0
        self.in_pic_contr_total:float = 0.0
        self.out_pic_contr_tile_left: float = 0.0
        self.out_pic_contr_tile_right: float = 0.0
        self.out_pic_contr_total: float = 0.0
        self.vs_settings = vs_settings

    def calculate_contrast(self,
                           image_tile_left:NDArray,
                           image_tile_right:NDArray,
                           edge_position:int = 0):
        in_pic_max_pos = edge_position - self.vs_settings.contrast_pic_offset
        in_pic_min_pos = in_pic_max_pos - self.vs_settings.contrast_pic_height
        in_pic_image_roi_left = image_tile_left[in_pic_min_pos:in_pic_max_pos, 0:image_tile_left.shape[1]]
        in_pic_image_roi_right = image_tile_right[in_pic_min_pos:in_pic_max_pos, 0:image_tile_right.shape[1]]
        out_pic_min_pos = edge_position + self.vs_settings.contrast_pic_offset
        out_pic_max_pos = out_pic_min_pos + self.vs_settings.contrast_pic_height
        out_pic_image_roi_left = image_tile_left[out_pic_min_pos:out_pic_max_pos, 0:image_tile_left.shape[1]]
        out_pic_image_roi_right = image_tile_right[out_pic_min_pos:out_pic_max_pos, 0:image_tile_right.shape[1]]
        self.in_pic_contr_tile_left = np.median(in_pic_image_roi_left)
        self.in_pic_contr_tile_right = np.median(in_pic_image_roi_right)
        self.out_pic_contr_tile_left = np.median(out_pic_image_roi_left)
        self.out_pic_contr_tile_right = np.median(out_pic_image_roi_right)
        self.in_pic_contr_total = self.in_pic_contr_tile_left + self.in_pic_contr_tile_right
        self.out_pic_contr_total = self.out_pic_contr_tile_left + self.out_pic_contr_tile_right


class EdgeParameterObject:
    def __init__(self):
        self.edge_position:int = -1
        self.edge_slope:float = 0.0
        self.edge_state:int = 0


class ProcessImageEdgeParameters:
    def __init__(self, vs_settings:VisionSensorSettings):
        self.image_np = None
        self.image_width:int = 0
        self.image_height:int = 0
        self.result_mean = EdgeParameterObject()
        self.result_tile_left = EdgeParameterObject()
        self.result_tile_right = EdgeParameterObject()
        self.edge_slope: float = 0.0
        self._vs_settings = vs_settings
        self.left_tile_slope_data = CalcEdgeSlopeParameter()
        self.right_tile_slope_data = CalcEdgeSlopeParameter()
        self._edge_position_tile_diff:int = 0
        self._total_edge_slope:float = 0.0
        self._total_contrast_offset: float = 0.0
        self._arr_slope_total_mean = deque(maxlen=40)
        self._slope_total_mean: float = 0.0
        self._slope_diff_rising:float = 0.0
        self._slope_diff_falling:float = 0.0
        self._new_edge_detected:bool = False
        self._t_edge_position:int = 0
        self._contrast_filter = CalculateContrast(vs_settings=self._vs_settings)
        self._edge_detected: bool = False
        self._edge_in_position: bool = False

    def process_image(self, image_data:NDArray):
        if image_data is not None:
            self._t_edge_position = -1
            self.image_np = image_data
            self.image_height, self.image_width = self.image_np.shape[:2]

            self.left_tile_slope_data.image_data, self.right_tile_slope_data.image_data = self._get_image_tiles(
                image_data=self.image_np,
                vs_settings=self._vs_settings
            )

            self.left_tile_slope_data.calculate()
            self.right_tile_slope_data.calculate()

            if self._vs_settings.film_type_is_negative:
                self._edge_position_tile_diff = abs(self.left_tile_slope_data.pos_film_neg - self.right_tile_slope_data.pos_film_neg)
                self.result_mean.edge_slope = abs(self.left_tile_slope_data.slope_film_neg + self.right_tile_slope_data.slope_film_neg)
                self.result_tile_left.edge_slope = self.left_tile_slope_data.slope_film_neg
                self.result_tile_right.edge_slope = self.right_tile_slope_data.slope_film_neg
            else:
                self._edge_position_tile_diff = abs(self.left_tile_slope_data.pos_film_pos - self.right_tile_slope_data.pos_film_pos)
                self.result_mean.edge_slope = abs(self.left_tile_slope_data.slope_film_pos + self.right_tile_slope_data.slope_film_pos)
                self.result_tile_left.edge_slope = self.left_tile_slope_data.slope_film_neg
                self.result_tile_right.edge_slope = self.right_tile_slope_data.slope_film_neg

            self._arr_slope_total_mean.append(self.result_mean.edge_slope)
            self._slope_total_mean = sum(self._arr_slope_total_mean) // len(self._arr_slope_total_mean)
            self._slope_diff_rising = self.result_mean.edge_slope - min(self._arr_slope_total_mean)
            self._slope_diff_falling = self._slope_total_mean - max(self._arr_slope_total_mean)

            if (self.result_mean.edge_slope - self._slope_total_mean) > 40 and self._new_edge_detected == 0:
                self._new_edge_detected = 200
                self._arr_slope_total_mean.clear()
                self._arr_slope_total_mean.append(self.result_mean.edge_slope)

            if (self._slope_total_mean - self.result_mean.edge_slope) > 40 and self._new_edge_detected == 200:
                self._new_edge_detected = 0
                self._arr_slope_total_mean.clear()
                self._arr_slope_total_mean.append(self.result_mean.edge_slope)


            if self.result_mean.edge_slope > self._vs_settings.slope_threshold:
                if self._vs_settings.film_type_is_negative:
                    self._t_edge_position = (self.left_tile_slope_data.pos_film_neg + self.right_tile_slope_data.pos_film_neg) // 2
                else:
                    self._t_edge_position = (self.left_tile_slope_data.pos_film_pos + self.right_tile_slope_data.pos_film_pos) // 2

            if self._t_edge_position > (self._vs_settings.contrast_pic_height + self._vs_settings.contrast_pic_offset):
                self._contrast_filter.calculate_contrast(
                    image_tile_left=self.left_tile_slope_data.image_data,
                    image_tile_right=self.right_tile_slope_data.image_data,
                    edge_position=self._t_edge_position
                )

            if self._vs_settings.film_type_is_negative:
                if self._contrast_filter.in_pic_contr_total + self._vs_settings.contrast_offset < self._contrast_filter.out_pic_contr_total:
                    self.result_mean.edge_position = self._t_edge_position
                else:
                    self.result_mean.edge_position = -1
                self.result_tile_right.edge_position = self.right_tile_slope_data.pos_film_neg
                self.result_tile_left.edge_position = self.left_tile_slope_data.pos_film_neg
                self.result_mean.edge_state = self._get_edge_state(self.result_mean.edge_position)
                self.result_tile_left.edge_state = self._get_edge_state(edge_position=self.left_tile_slope_data.pos_film_neg)
                self.result_tile_right.edge_state = self._get_edge_state(edge_position=self.right_tile_slope_data.pos_film_neg)
            else:
                if self._contrast_filter.in_pic_contr_total + self._vs_settings.contrast_offset > self._contrast_filter.out_pic_contr_total:
                    self.result_mean.edge_position = self._t_edge_position
                else:
                    self.result_mean.edge_position = -1
                self.result_tile_right.edge_position = self.right_tile_slope_data.pos_film_pos
                self.result_tile_left.edge_position = self.left_tile_slope_data.pos_film_pos
                self.result_mean.edge_state = self._get_edge_state(self.result_mean.edge_position)
                self.result_tile_left.edge_state = self._get_edge_state(edge_position=self.left_tile_slope_data.pos_film_neg)
                self.result_tile_right.edge_state = self._get_edge_state(edge_position=self.right_tile_slope_data.pos_film_neg)


    def _get_edge_state(self, edge_position):
        edge_state = 0
        edge_min_pos = self._vs_settings.stop_position - (self._vs_settings.edge_detection_range // 2)
        edge_max_pos = self._vs_settings.stop_position + (self._vs_settings.edge_detection_range // 2)

        if edge_position > 0:
            edge_state = 1

        if edge_min_pos < edge_position < edge_max_pos:
            edge_state = 2

        return edge_state

    @staticmethod
    def _get_image_tiles(
            image_data:NDArray,
            vs_settings:VisionSensorSettings,
    ) -> Tuple[NDArray, NDArray]:
        img_height, img_width = image_data.shape[:2]
        np_image_tile_left = image_data[
                             0 : vs_settings.tile_height,
                             (img_width // 2) - vs_settings.tile_width - vs_settings.tile_center_offset : (img_width // 2) - vs_settings.tile_center_offset]

        np_image_tile_right = image_data[
                              0:vs_settings.tile_height,
                              (img_width // 2) + vs_settings.tile_center_offset : (img_width // 2) + vs_settings.tile_width + vs_settings.tile_center_offset]
        return np_image_tile_left, np_image_tile_right