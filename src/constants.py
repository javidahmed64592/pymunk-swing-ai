import numpy as np
import pymunk

AIR_DENSITY = 0.45
DRAG_COEFFICIENT = 0.025
NORM_VECTOR = pymunk.Vec2d(0, -1)


def dir_vec(angle: float) -> pymunk.Vec2d:
    return NORM_VECTOR.rotated(angle * np.pi / 180)
