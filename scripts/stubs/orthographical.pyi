from typing import Any, List

import numpy as np

from cfrankr import Affine


class OrthographicImage:
    mat: np.ndarray
    pixel_size: float
    min_depth: float
    max_depth: float
    camera: str
    pose: Affine

    def __init__(self, image: Any, pixel_size: float, min_depth: float, max_depth: float, camera: str = '', pose: Affine = Affine()):
        ...

    def depth_from_value(self, value: float) -> float:
        ...

    def value_from_depth(self, depth: float) -> float:
        ...

    def project(self, point: Affine) -> List[int]:
        ...

    def position_from_index(self, idx: int, length: int) -> int:
        ...

    def index_from_position(self, position: float, length: int) -> int:
        ...

    def translate(self, vector: List[float]) -> OrthographicImage:
        ...

    def rotate_x(self, angle: float, vector: List[float]) -> OrthographicImage:
        ...

    def rotate_y(self, angle: float, vector: List[float]) -> OrthographicImage:
        ...

    def rotate_z(self, angle: float, vector: List[float]) -> OrthographicImage:
        ...

    def rescale(self, new_pixel_size: float, new_min_depth: float, new_max_depth: float) -> OrthographicImage:
        ...
