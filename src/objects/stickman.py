from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymunk
from pymunk.vec2d import Vec2d


@dataclass
class LimbConfig:
    length: float
    mass: float
    start_pos: Vec2d
    start_angle: 0
    angle_constraints: tuple[float, float]


@dataclass
class Limb:
    space: pymunk.Space
    limb_config: LimbConfig
    joint_body: pymunk.Body = None
    joint_motor: int = None
    limb_segment: pymunk.Segment = None

    @property
    def start_limb_position(self) -> Vec2d:
        return self.joint_body.position

    @property
    def end_limb_position(self) -> Vec2d:
        return self.joint_body.position + (Limb._get_dir_vec(self.limb_config.start_angle) * self.limb_config.length)

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

        limb_config = LimbConfig(length, mass, start_pos, start_angle, angle_constraints)
        limb = cls(space, limb_config, body, 1, segment)
        return limb

    @staticmethod
    def _get_dir_vec(angle: float) -> Vec2d:
        return Vec2d(np.sin(angle * np.pi / 180), -np.cos(angle * np.pi / 180))
