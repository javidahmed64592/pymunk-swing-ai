from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymunk
from pymunk.vec2d import Vec2d

from src.constants import NORM_VECTOR


@dataclass
class LimbConfig:
    length: float
    mass: float
    start_angle: 0
    angle_constraints: tuple[float, float]

    @classmethod
    def from_dict(cls, limb_data: dict) -> LimbConfig:
        return cls(limb_data["length"], limb_data["mass"], limb_data["start_angle"], limb_data["angle_constraints"])


@dataclass
class Limb:
    space: pymunk.Space
    limb_config: LimbConfig
    start_pos: Vec2d
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

        limb_config = LimbConfig(length, mass, start_angle, angle_constraints)
        limb = cls(space, limb_config, start_pos, body, 1, segment)
        return limb

    @classmethod
    def from_config(
        cls, config: LimbConfig, space: pymunk.Space, start_pos: Vec2d, offset: Vec2d | None = None
    ) -> Limb:
        if offset:
            start_pos += offset
        return cls.generate(space, start_pos, config.length, config.mass, config.start_angle, config.angle_constraints)

    @staticmethod
    def _get_dir_vec(angle: float) -> Vec2d:
        return NORM_VECTOR.rotated(angle * np.pi / 180)


@dataclass
class Stickman:
    space: pymunk.Space
    start_pos: Vec2d
    foot: Limb
    lower_leg: Limb
    upper_leg: Limb
    torso: Limb
    upper_arm: Limb
    lower_arm: Limb
    neck: Limb

    @classmethod
    def create(cls, space: pymunk.Space, start_pos: Vec2d, config: dict) -> Stickman:
        foot_config = LimbConfig.from_dict(config["foot"])
        lower_leg_config = LimbConfig.from_dict(config["lower_leg"])
        upper_leg_config = LimbConfig.from_dict(config["upper_leg"])
        torso_config = LimbConfig.from_dict(config["torso"])
        upper_arm_config = LimbConfig.from_dict(config["upper_arm"])
        lower_arm_config = LimbConfig.from_dict(config["lower_arm"])
        neck_config = LimbConfig.from_dict(config["neck"])

        foot = Limb.from_config(foot_config, space, start_pos)
        lower_leg = Limb.from_config(
            lower_leg_config, space, foot.start_limb_position, -Vec2d(0, lower_leg_config.length)
        )
        upper_leg = Limb.from_config(
            upper_leg_config, space, lower_leg.start_limb_position, -Vec2d(upper_leg_config.length, 0)
        )
        torso = Limb.from_config(torso_config, space, upper_leg.start_limb_position, -Vec2d(0, torso_config.length))
        upper_arm = Limb.from_config(upper_arm_config, space, torso.start_limb_position)
        lower_arm = Limb.from_config(lower_arm_config, space, upper_arm.end_limb_position)
        neck = Limb.from_config(neck_config, space, torso.start_limb_position)
        return cls(space, start_pos, foot, lower_leg, upper_leg, torso, upper_arm, lower_arm, neck)
