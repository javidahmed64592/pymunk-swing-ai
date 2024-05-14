import numpy as np
from pymunk import Vec2d

AIR_DENSITY = 0.45
DRAG_COEFFICIENT = 0.025
NORM_VECTOR = Vec2d(0, -1)


def dir_vec(angle: float) -> Vec2d:
    return NORM_VECTOR.rotated(angle * np.pi / 180)
