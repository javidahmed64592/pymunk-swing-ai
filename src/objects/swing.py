from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pymunk

from src.constants import AIR_DENSITY, DRAG_COEFFICIENT, dir_vec
from src.data_types import SwingConfigType


@dataclass
class ChainLinkConfig:
    mass: float
    radius: float
    shape_filter_group: int


@dataclass
class ChainLink:
    space: pymunk.Space
    chain_link_config: ChainLinkConfig
    start_pos: pymunk.Vec2d
    body: pymunk.Body
    shape: pymunk.Circle
    constraints: list[pymunk.Constraint] = field(default_factory=lambda: [])

    @property
    def area(self) -> pymunk.Vec2d:
        return self.shape.area

    @property
    def position(self) -> pymunk.Vec2d:
        return self.body.position

    @property
    def velocity(self) -> pymunk.Vec2d:
        return self.body.velocity

    @property
    def drag_force(self) -> pymunk.Vec2d:
        k = (AIR_DENSITY * DRAG_COEFFICIENT * self.area) / 2
        direction, speed = self.velocity.normalized_and_length()
        drag_force_magnitude = k * (speed**2)
        drag_force = -direction * drag_force_magnitude
        return drag_force

    @classmethod
    def generate(
        cls, space: pymunk.Space, start_pos: pymunk.Vec2d, mass: float, radius: float, shape_filter_group: int
    ) -> ChainLink:
        chain_link_config = ChainLinkConfig(mass, radius, shape_filter_group)

        body = ChainLink._generate_body(chain_link_config, start_pos)
        shape = ChainLink._generate_shape(chain_link_config, body)
        space.add(body, shape)

        return cls(space, chain_link_config, start_pos, body, shape)

    @classmethod
    def static_link(
        cls, space: pymunk.Space, start_pos: pymunk.Vec2d, mass: float, radius: float, shape_filter_group: int
    ) -> ChainLink:
        link = cls.generate(space, start_pos, mass, radius, shape_filter_group)
        link.body.body_type = pymunk.Body.STATIC
        return link

    @classmethod
    def dynamic_link(cls, other_link: ChainLink, length: float, angle: float) -> ChainLink:
        pos = other_link.body.position + (dir_vec(angle) * length)
        link = cls.generate(
            other_link.space,
            pos,
            other_link.chain_link_config.mass,
            other_link.chain_link_config.radius,
            other_link.chain_link_config.shape_filter_group,
        )
        link.add_pinjoint(other_link.body, link.body, pymunk.Vec2d(0, 0))
        return link

    @staticmethod
    def _generate_body(chain_link_config: ChainLinkConfig, start_pos: pymunk.Vec2d) -> pymunk.Body:
        body = pymunk.Body(chain_link_config.mass, np.pi / 4 * (chain_link_config.radius**4))
        body.position = start_pos
        return body

    @staticmethod
    def _generate_shape(chain_link_config: ChainLinkConfig, body: pymunk.Body) -> pymunk.Circle:
        shape = pymunk.Circle(body, chain_link_config.radius)
        shape.filter = pymunk.ShapeFilter(chain_link_config.shape_filter_group)
        return shape

    def add_pinjoint(self, body: pymunk.Body, other_body: pymunk.Body, offset: pymunk.Vec2d) -> pymunk.Constraint:
        constraint = pymunk.PinJoint(body, other_body, offset)
        self.space.add(constraint)
        self.constraints.append(constraint)
        return constraint

    def update(self) -> None:
        self.body.apply_force_at_world_point(self.drag_force, self.position)

    def remove_from_space(self) -> None:
        self.space.remove(self.body, self.shape, *self.constraints)


@dataclass
class Swing:
    space: pymunk.Space
    start_pos: pymunk.Vec2d
    start_angle: float
    links: list[ChainLink]
    link_length: float

    @property
    def num_links(self) -> int:
        return len(self.links)

    @property
    def top_link(self) -> ChainLink:
        return self.get_link_by_index(0)

    @property
    def seat(self) -> ChainLink:
        return self.get_link_by_index(-1)

    @classmethod
    def create(
        cls,
        swing_config: SwingConfigType,
        space: pymunk.Space,
        start_pos: pymunk.Vec2d,
        shape_filter_group: int,
    ) -> Swing:
        _links: list[ChainLink] = []
        link = ChainLink.static_link(
            space, start_pos, swing_config.link_mass, swing_config.link_radius, shape_filter_group
        )
        _links.append(link)
        for _ in range(1, swing_config.num_links):
            _links.append(ChainLink.dynamic_link(_links[-1], swing_config.link_length, swing_config.start_angle))

        return cls(space, start_pos, swing_config.start_angle, _links, swing_config.link_length)

    def get_link_by_index(self, index: int) -> ChainLink:
        return self.links[index]

    def update(self) -> None:
        for link in self.links:
            link.update()

    def remove(self) -> None:
        for link in self.links:
            link.remove_from_space()
