from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymunk
from pymunk.vec2d import Vec2d


@dataclass
class Limb:
    space: pymunk.Space
    start_pos: Vec2d
    start_angle: float
    length: float
    angle_constraints: tuple[float, float]
    joint_body: pymunk.Body = None
    joint_motor: int = None
    limb_segment: pymunk.Segment = None

    @property
    def start_limb_position(self) -> Vec2d:
        return self.joint_body.position

    @property
    def end_limb_position(self) -> Vec2d:
        return self.joint_body.position + (Limb._get_dir_vec(self.start_angle) * self.length)

    @classmethod
    def generate(
        cls,
        space: pymunk.Space,
        start_pos: Vec2d,
        length: float,
        mass: float,
        start_angle: float,
        angle_constraints: tuple[float, float],
    ) -> Limb:
        body = pymunk.Body(mass, length / 12)
        body.position = start_pos

        dir_vec = Limb._get_dir_vec(start_angle)
        segment = pymunk.Segment(body, Vec2d(0, 0), (dir_vec * length), 3)
        space.add(body, segment)
        limb = cls(space, start_pos, start_angle, length, angle_constraints, body, 1, segment)
        return limb

    @staticmethod
    def _get_dir_vec(angle: float) -> Vec2d:
        return Vec2d(np.sin(angle * np.pi / 180), -np.cos(angle * np.pi / 180))
