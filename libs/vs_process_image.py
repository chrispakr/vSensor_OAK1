from collections import deque
from typing import Tuple
from numpy.typing import NDArray
from dataclasses import dataclass
import logging
import numpy as np
import traceback


class VisionSensorSettings:
    PREVIEW_WIDTH = 800
    def __init__(self,
                 camera_center_position:int = 430,
                 stop_position:int = 250,
                 stop_offset:int = 10,
                 edge_detection_range: int = 12,
                 is_film_type_negative: bool = True,
                 slope_threshold: float = 30.0,
                 contrast_offset: float = 30.0,
                 tile_center_offset:int = 50,
                 tile_width:int = 300,
                 tile_height:int = 350,
                 contrast_pic_height:int = 20,
                 contrast_pic_edge_offset:int = 10,
                 exposure_time = 0,
                 lens_position = 0
                 ):
        # self.preview_width = 800
        self.logger = logging.getLogger("base." + self.__class__.__name__)
        self._camera_center_position = camera_center_position
        self.stop_position = stop_position
        self.stop_offset = stop_offset
        self._edge_detection_range = edge_detection_range
        self._is_film_type_negative = is_film_type_negative
        self._slope_threshold = slope_threshold
        self._contrast_offset = contrast_offset
        self._contrast_pic_height = contrast_pic_height
        self._contrast_pic_offset = contrast_pic_edge_offset
        self._tile_center_offset = tile_center_offset
        self._tile_width = tile_width
        self._tile_height = tile_height
        self.exposure_time = exposure_time
        self.lens_position = lens_position

    @property
    def tile_center_offset(self):
        return self._tile_center_offset

    @tile_center_offset.setter
    def tile_center_offset(self, value:int):
        self._tile_center_offset = value
        if (self._tile_center_offset + self._tile_width) > (self.PREVIEW_WIDTH // 2):
            self._tile_width = (self.PREVIEW_WIDTH // 2) - self._tile_center_offset
        self.logger.debug(f"set tile_center_offset to: {self._tile_center_offset}")

    @property
    def tile_width(self):
        return self._tile_width

    @tile_width.setter
    def tile_width(self, value:int):
        self._tile_width = value
        if (self._tile_width + self._tile_center_offset) > (self.PREVIEW_WIDTH // 2):
            self._tile_width = (self.PREVIEW_WIDTH // 2) - self._tile_center_offset
        self.logger.debug(f"set tile_width to: {self._tile_width}")

    @property
    def tile_height(self):
        return self._tile_height

    @tile_height.setter
    def tile_height(self, value: int):
        self._tile_height = value
        self.logger.debug(f"set tile_height to: {self.tile_height}")

    @property
    def slope_threshold(self):
        return self._slope_threshold

    @slope_threshold.setter
    def slope_threshold(self, value):
        self._slope_threshold = value
        self.logger.debug(f"set slope_threshold to: {self._slope_threshold}")

    @property
    def contrast_offset(self):
        return self._contrast_offset

    @contrast_offset.setter
    def contrast_offset(self, value):
        self._contrast_offset = value
        self.logger.debug(f"set contrast_offset to: {self._contrast_offset}")

    @property
    def contrast_pic_height(self):
        return self._contrast_pic_height

    @contrast_pic_height.setter
    def contrast_pic_height(self, value):
        self._contrast_pic_height = value
        self.logger.debug(f"set contrast_pic_height to: {self._contrast_pic_height}")

    @property
    def contrast_pic_offset(self):
        return self._contrast_pic_offset

    @contrast_pic_offset.setter
    def contrast_pic_offset(self, value):
        self._contrast_pic_offset = value
        self.logger.debug(f"set contrast_pic_offset to: {self._contrast_pic_offset}")

    @property
    def camera_center_position(self):
        return self._camera_center_position

    @camera_center_position.setter
    def camera_center_position(self, value):
        self._camera_center_position = value
        self.logger.debug(f"set camera_center_position to: {self._camera_center_position}")

    @property
    def edge_detection_range(self):
        return self._edge_detection_range

    @edge_detection_range.setter
    def edge_detection_range(self, value):
        self._edge_detection_range = value
        self.logger.debug(f"set edge_detection_range to: {self._edge_detection_range}")

    @property
    def is_film_type_negative(self):
        return self._is_film_type_negative

    @is_film_type_negative.setter
    def is_film_type_negative(self, value):
        self._is_film_type_negative = value
        self.logger.debug(f"set is_film_type_negative to: {self._is_film_type_negative}")


class CalcEdgeSlopeParameter:
    def __init__(self):
        self.logger = logging.getLogger("base." + self.__class__.__name__)
        self.image_data = None
        self.slope_film_pos:float = 0.0
        self.slope_film_neg:float = 0.0
        self.pos_film_pos:int = 0
        self.pos_film_neg:int = 0
        self.slope_data = []

    def calculate(self):
        if self.image_data is not None and self.image_data.size > 0:
            try:
                reduced = np.mean(self.image_data, axis=1)
                self.slope_data = [(reduced[i + 3] - reduced[i - 3]) for i in range(3, len(reduced) - 3, 1)]

                self.slope_film_pos = min(self.slope_data)
                self.slope_film_neg = max(self.slope_data)
                self.pos_film_pos = self.slope_data.index(self.slope_film_pos)
                self.pos_film_neg = self.slope_data.index(self.slope_film_neg)
            except ValueError as e:
                self.logger.error(f"calculate edge slope error: {e} // img_size: {self.image_data.shape}")
                traceback.print_exc()


@dataclass
class ContrastMeasurements:
    """Stores contrast measurements for image tiles"""
    inner_left: float = 0.0
    inner_right: float = 0.0
    inner_total: float = 0.0
    outer_left: float = 0.0
    outer_right: float = 0.0
    outer_total: float = 0.0


class CalculateContrast:
    def __init__(self, vs_settings: VisionSensorSettings):
        self.vs_settings = vs_settings
        self.measurements = ContrastMeasurements()

    def calculate_contrast(self,
                           image_tile_left: NDArray,
                           image_tile_right: NDArray,
                           edge_position: int = 0) -> None:
        """Calculate contrast measurements for inner and outer regions of image tiles"""
        inner_rois = self._get_inner_rois(image_tile_left, image_tile_right, edge_position)
        outer_rois = self._get_outer_rois(image_tile_left, image_tile_right, edge_position)

        self.measurements = ContrastMeasurements(
            inner_left=float(np.median(inner_rois[0])),
            inner_right=float(np.median(inner_rois[1])),
            outer_left=float(np.median(outer_rois[0])),
            outer_right=float(np.median(outer_rois[1]))
        )
        self._calculate_totals()

    def _get_inner_rois(self, left_tile: NDArray, right_tile: NDArray, edge_pos: int) -> tuple[NDArray, NDArray]:
        """Extract regions of interest for inner (pre-edge) measurements"""
        max_pos = edge_pos - self.vs_settings.contrast_pic_offset
        min_pos = max_pos - self.vs_settings.contrast_pic_height
        return (
            self._extract_roi(left_tile, min_pos, max_pos),
            self._extract_roi(right_tile, min_pos, max_pos)
        )

    def _get_outer_rois(self, left_tile: NDArray, right_tile: NDArray, edge_pos: int) -> tuple[NDArray, NDArray]:
        """Extract regions of interest for outer (post-edge) measurements"""
        min_pos = edge_pos + self.vs_settings.contrast_pic_offset
        max_pos = min_pos + self.vs_settings.contrast_pic_height
        return (
            self._extract_roi(left_tile, min_pos, max_pos),
            self._extract_roi(right_tile, min_pos, max_pos)
        )

    @staticmethod
    def _extract_roi(image: NDArray, min_pos: int, max_pos: int) -> NDArray:
        """Extract a region of interest from an image tile"""
        return image[min_pos:max_pos, 0:image.shape[1]]

    def _calculate_totals(self) -> None:
        """Calculate total contrast values for inner and outer measurements"""
        self.measurements.inner_total = self.measurements.inner_left + self.measurements.inner_right
        self.measurements.outer_total = self.measurements.outer_left + self.measurements.outer_right

#
# class CalculateContrast:
#     def __init__(self, vs_settings:VisionSensorSettings):
#         self.in_pic_contr_tile_left:float = 0.0
#         self.in_pic_contr_tile_right:float = 0.0
#         self.in_pic_contr_total:float = 0.0
#         self.out_pic_contr_tile_left: float = 0.0
#         self.out_pic_contr_tile_right: float = 0.0
#         self.out_pic_contr_total: float = 0.0
#         self.vs_settings = vs_settings
#
#     def calculate_contrast(self,
#                            image_tile_left:NDArray,
#                            image_tile_right:NDArray,
#                            edge_position:int = 0):
#         in_pic_max_pos = edge_position - self.vs_settings.contrast_pic_offset
#         in_pic_min_pos = in_pic_max_pos - self.vs_settings.contrast_pic_height
#         in_pic_image_roi_left = image_tile_left[in_pic_min_pos:in_pic_max_pos, 0:image_tile_left.shape[1]]
#         in_pic_image_roi_right = image_tile_right[in_pic_min_pos:in_pic_max_pos, 0:image_tile_right.shape[1]]
#         out_pic_min_pos = edge_position + self.vs_settings.contrast_pic_offset
#         out_pic_max_pos = out_pic_min_pos + self.vs_settings.contrast_pic_height
#         out_pic_image_roi_left = image_tile_left[out_pic_min_pos:out_pic_max_pos, 0:image_tile_left.shape[1]]
#         out_pic_image_roi_right = image_tile_right[out_pic_min_pos:out_pic_max_pos, 0:image_tile_right.shape[1]]
#         self.in_pic_contr_tile_left = np.median(in_pic_image_roi_left)
#         self.in_pic_contr_tile_right = np.median(in_pic_image_roi_right)
#         self.out_pic_contr_tile_left = np.median(out_pic_image_roi_left)
#         self.out_pic_contr_tile_right = np.median(out_pic_image_roi_right)
#         self.in_pic_contr_total = self.in_pic_contr_tile_left + self.in_pic_contr_tile_right
#         self.out_pic_contr_total = self.out_pic_contr_tile_left + self.out_pic_contr_tile_right


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
        self.vs_settings = vs_settings
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
        self.contrast_offset_data = CalculateContrast(vs_settings=self.vs_settings)
        self._edge_detected: bool = False
        self._edge_in_position: bool = False

    def process_image(self, image_data:NDArray):
        if image_data is not None:
            self._t_edge_position = -1
            self.image_np = image_data
            self.image_height, self.image_width = self.image_np.shape[:2]

            self.left_tile_slope_data.image_data, self.right_tile_slope_data.image_data = self._get_image_tiles(
                image_data=self.image_np,
                vs_settings=self.vs_settings
            )

            self.left_tile_slope_data.calculate()
            self.right_tile_slope_data.calculate()

            if self.vs_settings.is_film_type_negative:
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


            if self.result_mean.edge_slope > self.vs_settings.slope_threshold:
                if self.vs_settings.is_film_type_negative:
                    self._t_edge_position = (self.left_tile_slope_data.pos_film_neg + self.right_tile_slope_data.pos_film_neg) // 2
                else:
                    self._t_edge_position = (self.left_tile_slope_data.pos_film_pos + self.right_tile_slope_data.pos_film_pos) // 2

            if self._t_edge_position > (self.vs_settings.contrast_pic_height + self.vs_settings.contrast_pic_offset):
                self.contrast_offset_data.calculate_contrast(
                    image_tile_left=self.left_tile_slope_data.image_data,
                    image_tile_right=self.right_tile_slope_data.image_data,
                    edge_position=self._t_edge_position
                )

            if self.vs_settings.is_film_type_negative:
                if self.contrast_offset_data.measurements.inner_total + self.vs_settings.contrast_offset < self.contrast_offset_data.measurements.outer_total:
                    self.result_mean.edge_position = self._t_edge_position
                else:
                    self.result_mean.edge_position = -1
                self.result_tile_right.edge_position = self.right_tile_slope_data.pos_film_neg
                self.result_tile_left.edge_position = self.left_tile_slope_data.pos_film_neg
                self.result_mean.edge_state = self._get_edge_state(self.result_mean.edge_position)
                self.result_tile_left.edge_state = self._get_edge_state(edge_position=self.left_tile_slope_data.pos_film_neg)
                self.result_tile_right.edge_state = self._get_edge_state(edge_position=self.right_tile_slope_data.pos_film_neg)
            else:
                if self.contrast_offset_data.measurements.inner_total + self.vs_settings.contrast_offset > self.contrast_offset_data.measurements.outer_total:
                    self.result_mean.edge_position = self._t_edge_position
                else:
                    self.result_mean.edge_position = -1
                self.result_tile_right.edge_position = self.right_tile_slope_data.pos_film_pos
                self.result_tile_left.edge_position = self.left_tile_slope_data.pos_film_pos
                self.result_mean.edge_state = self._get_edge_state(self.result_mean.edge_position)
                self.result_tile_left.edge_state = self._get_edge_state(edge_position=self.left_tile_slope_data.pos_film_neg)
                self.result_tile_right.edge_state = self._get_edge_state(edge_position=self.right_tile_slope_data.pos_film_neg)

            # print(self.result_tile_right.edge_state, self.result_tile_right.edge_state, self.result_mean.edge_state)

    def _get_edge_state(self, edge_position:int):
        edge_state = 0
        edge_min_pos = int(self.vs_settings.stop_position - (self.vs_settings.edge_detection_range // 2))
        edge_max_pos = int(self.vs_settings.stop_position + (self.vs_settings.edge_detection_range // 2))

        if edge_position > 0:
            edge_state = 1

        # print(self.vs_settings.stop_position, edge_min_pos, edge_max_pos, edge_position)
        if edge_min_pos <= edge_position <= edge_max_pos:
            edge_state = 2

        return edge_state

    @staticmethod
    def _get_image_tiles(
            image_data:NDArray,
            vs_settings:VisionSensorSettings,
    ) -> Tuple[NDArray, NDArray]:
        img_height, img_width = image_data.shape[:2]
        tile_height = max(0, min(vs_settings.tile_height, img_height))
        # Clamp against the actual image width (not the fixed PREVIEW_WIDTH the
        # settings are validated against), otherwise a narrower raw_image_width
        # can push the tile bounds out of range and yield an empty slice.
        center_offset = max(0, min(vs_settings.tile_center_offset, img_width // 2))
        tile_width = max(0, min(vs_settings.tile_width, (img_width // 2) - center_offset))

        np_image_tile_left = image_data[
                             0 : tile_height,
                             (img_width // 2) - tile_width - center_offset : (img_width // 2) - center_offset]

        np_image_tile_right = image_data[
                              0 : tile_height,
                              (img_width // 2) + center_offset : (img_width // 2) + tile_width + center_offset]
        return np_image_tile_left, np_image_tile_right