from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymunk

from src.constants import AIR_DENSITY, DRAG_COEFFICIENT, dir_vec
from src.data_types import HeadConfigType, LimbConfigType, StickmanConfigType


@dataclass
class HeadConfig:
    radius: float
    mass: float
    start_angle: 0
    angle_constraints: tuple[float, float]
    shape_filter_group: int

    @classmethod
    def from_config(cls, config: HeadConfigType, shape_filter_group: int) -> HeadConfig:
        return cls(
            config.radius,
            config.mass,
            config.start_angle,
            config.angle_constraints,
            shape_filter_group,
        )


@dataclass
class Head:
    space: pymunk.Space
    head_config: HeadConfig
    start_pos: pymunk.Vec2d
    head_body: pymunk.Body
    head_shape: pymunk.Circle
    joint_motor: int = None

    @property
    def start_head_position(self) -> pymunk.Vec2d:
        return self.head_body.position

    @property
    def end_head_position(self) -> pymunk.Vec2d:
        return self.head_body.position + (dir_vec(self.head_config.start_angle) * self.head_config.radius)

    @property
    def area(self) -> pymunk.Vec2d:
        return self.head_shape.area

    @property
    def position(self) -> pymunk.Vec2d:
        return self.head_body.position

    @property
    def velocity(self) -> pymunk.Vec2d:
        return self.head_body.velocity

    @property
    def drag_force(self) -> pymunk.Vec2d:
        k = (AIR_DENSITY * DRAG_COEFFICIENT * self.area) / 2
        direction, speed = self.velocity.normalized_and_length()
        drag_force_magnitude = k * (speed**2)
        drag_force = -direction * drag_force_magnitude
        return drag_force

    @classmethod
    def generate(
        cls,
        space: pymunk.Space,
        start_pos: pymunk.Vec2d,
        radius: float,
        mass: float,
        start_angle: float,
        angle_constraints: tuple[float, float],
        shape_filter_group: int,
    ) -> Head:
        head_config = HeadConfig(radius, mass, start_angle, angle_constraints, shape_filter_group)

        body = Head._generate_body(head_config, start_pos)
        shape = Head._generate_shape(head_config, body, shape_filter_group)
        space.add(body, shape)

        return cls(space, head_config, start_pos, body, shape, 1)

    @classmethod
    def from_config(
        cls, config: HeadConfig, space: pymunk.Space, start_pos: pymunk.Vec2d, offset: pymunk.Vec2d | None = None
    ) -> Head:
        if offset:
            start_pos += offset
        return cls.generate(
            space,
            start_pos,
            config.radius,
            config.mass,
            config.start_angle,
            config.angle_constraints,
            config.shape_filter_group,
        )

    @staticmethod
    def _generate_body(head_config: HeadConfig, start_pos: pymunk.Vec2d) -> pymunk.Body:
        body = pymunk.Body(head_config.mass, np.pi / 4 * (head_config.radius**4))
        body.position = start_pos
        return body

    @staticmethod
    def _generate_shape(head_config: HeadConfig, body: pymunk.Body, shape_filter_group: int) -> pymunk.Circle:
        head_dir = dir_vec(head_config.start_angle)
        shape = pymunk.Circle(body, head_config.radius, head_dir * head_config.radius)
        shape.filter = pymunk.ShapeFilter(shape_filter_group)
        return shape

    def update(self) -> None:
        self.head_body.apply_force_at_world_point(self.drag_force, self.position)


@dataclass
class LimbConfig:
    length: float
    mass: float
    start_angle: 0
    angle_constraints: tuple[float, float]
    shape_filter_group: int

    @classmethod
    def from_config(cls, config: LimbConfigType, shape_filter_group: int) -> LimbConfig:
        return cls(
            config.length,
            config.mass,
            config.start_angle,
            config.angle_constraints,
            shape_filter_group,
        )


@dataclass
class Limb:
    space: pymunk.Space
    limb_config: LimbConfig
    start_pos: pymunk.Vec2d
    limb_body: pymunk.Body
    limb_segment: pymunk.Segment
    joint_motor: int = None

    @property
    def start_limb_position(self) -> pymunk.Vec2d:
        return self.limb_body.position

    @property
    def end_limb_position(self) -> pymunk.Vec2d:
        return self.limb_body.position + (dir_vec(self.limb_config.start_angle) * self.limb_config.length)

    @property
    def area(self) -> pymunk.Vec2d:
        return self.limb_segment.area

    @property
    def position(self) -> pymunk.Vec2d:
        return self.limb_body.position

    @property
    def velocity(self) -> pymunk.Vec2d:
        return self.limb_body.velocity

    @property
    def drag_force(self) -> pymunk.Vec2d:
        k = (AIR_DENSITY * DRAG_COEFFICIENT * self.area) / 2
        direction, speed = self.velocity.normalized_and_length()
        drag_force_magnitude = k * (speed**2)
        drag_force = -direction * drag_force_magnitude
        return drag_force

    def update(self) -> None:
        self.limb_body.apply_force_at_world_point(self.drag_force, self.position)

    @classmethod
    def generate(
        cls,
        space: pymunk.Space,
        start_pos: pymunk.Vec2d,
        length: float,
        mass: float,
        start_angle: float,
        angle_constraints: tuple[float, float],
        shape_filter_group: int,
    ) -> Limb:
        limb_config = LimbConfig(length, mass, start_angle, angle_constraints, shape_filter_group)

        body = Limb._generate_body(limb_config, start_pos)
        segment = Limb._generate_shape(limb_config, body, shape_filter_group)
        space.add(body, segment)

        return cls(space, limb_config, start_pos, body, segment, 1)

    @classmethod
    def from_config(
        cls, config: LimbConfig, space: pymunk.Space, start_pos: pymunk.Vec2d, offset: pymunk.Vec2d | None = None
    ) -> Limb:
        if offset:
            start_pos += offset
        return cls.generate(
            space,
            start_pos,
            config.length,
            config.mass,
            config.start_angle,
            config.angle_constraints,
            config.shape_filter_group,
        )

    @staticmethod
    def _generate_body(limb_config: LimbConfig, start_pos: pymunk.Vec2d) -> pymunk.Body:
        body = pymunk.Body(limb_config.mass, limb_config.length / 12)
        body.position = start_pos
        return body

    @staticmethod
    def _generate_shape(limb_config: LimbConfig, body: pymunk.Body, shape_filter_group: int) -> pymunk.Segment:
        limb_dir = dir_vec(limb_config.start_angle)
        segment = pymunk.Segment(body, pymunk.Vec2d(0, 0), (limb_dir * limb_config.length), 3)
        segment.filter = pymunk.ShapeFilter(shape_filter_group)
        return segment


@dataclass
class Stickman:
    space: pymunk.Space
    start_pos: pymunk.Vec2d
    head: Head
    neck: Limb
    upper_arm: Limb
    lower_arm: Limb
    torso: Limb
    lower_leg: Limb
    upper_leg: Limb
    foot: Limb

    @property
    def body_parts(self) -> list[Head, Limb]:
        return [
            self.head,
            self.neck,
            self.upper_arm,
            self.lower_arm,
            self.torso,
            self.upper_leg,
            self.lower_leg,
            self.foot,
        ]

    @classmethod
    def create(
        cls, config: StickmanConfigType, space: pymunk.Space, start_pos: pymunk.Vec2d, shape_filter_group: int
    ) -> Stickman:
        head_config = HeadConfig.from_config(config.head, shape_filter_group)
        neck_config = LimbConfig.from_config(config.neck, shape_filter_group)
        upper_arm_config = LimbConfig.from_config(config.upper_arm, shape_filter_group)
        lower_arm_config = LimbConfig.from_config(config.lower_arm, shape_filter_group)
        torso_config = LimbConfig.from_config(config.torso, shape_filter_group)
        upper_leg_config = LimbConfig.from_config(config.upper_leg, shape_filter_group)
        lower_leg_config = LimbConfig.from_config(config.lower_leg, shape_filter_group)
        foot_config = LimbConfig.from_config(config.foot, shape_filter_group)

        foot = Limb.from_config(foot_config, space, start_pos)
        lower_leg = Limb.from_config(
            lower_leg_config, space, foot.start_limb_position, -pymunk.Vec2d(0, lower_leg_config.length)
        )
        upper_leg = Limb.from_config(
            upper_leg_config, space, lower_leg.start_limb_position, -pymunk.Vec2d(upper_leg_config.length, 0)
        )
        torso = Limb.from_config(
            torso_config, space, upper_leg.start_limb_position, -pymunk.Vec2d(0, torso_config.length)
        )
        upper_arm = Limb.from_config(upper_arm_config, space, torso.start_limb_position)
        lower_arm = Limb.from_config(lower_arm_config, space, upper_arm.end_limb_position)
        neck = Limb.from_config(neck_config, space, torso.start_limb_position)
        head = Head.from_config(head_config, space, neck.end_limb_position)
        return cls(space, start_pos, head, neck, upper_arm, lower_arm, torso, upper_leg, lower_leg, foot)

    def update(self) -> None:
        for body_part in self.body_parts:
            body_part.update()
