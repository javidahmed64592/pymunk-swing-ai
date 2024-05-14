from __future__ import annotations

from dataclasses import dataclass, field

import pymunk
from pymunk.vec2d import Vec2d

from src.constants import AIR_DENSITY, DRAG_COEFFICIENT, dir_vec


@dataclass
class ChainLinkConfig:
    mass: float
    radius: float
    shape_filter_group: int


@dataclass
class ChainLink:
    space: pymunk.Space
    chain_link_config: ChainLinkConfig
    start_pos: Vec2d
    body: pymunk.Body
    shape: pymunk.Circle
    constraints: list[pymunk.Constraint] = field(default_factory=lambda: [])

    @property
    def area(self) -> Vec2d:
        return self.shape.area

    @property
    def position(self) -> Vec2d:
        return self.body.position

    @property
    def velocity(self) -> Vec2d:
        return self.body.velocity

    @property
    def drag_force(self) -> Vec2d:
        k = (AIR_DENSITY * DRAG_COEFFICIENT * self.area) / 2
        direction, speed = self.velocity.normalized_and_length()
        drag_force_magnitude = k * (speed**2)
        drag_force = -direction * drag_force_magnitude
        return drag_force

    @classmethod
    def generate(
        cls, space: pymunk.Space, start_pos: Vec2d, mass: float, radius: float, shape_filter_group: int
    ) -> ChainLink:
        chain_link_config = ChainLinkConfig(mass, radius, shape_filter_group)

        body = ChainLink._generate_body(chain_link_config, start_pos)
        shape = ChainLink._generate_shape(chain_link_config, body)
        space.add(body, shape)

        link = cls(space, chain_link_config, start_pos, body, shape)
        return link

    @classmethod
    def static_link(
        cls, space: pymunk.Space, start_pos: Vec2d, mass: float, radius: float, shape_filter_group: int
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
        link.add_pinjoint(other_link.body, link.body, Vec2d(0, 0))
        return link

    @staticmethod
    def _generate_body(chain_link_config: ChainLinkConfig, start_pos: Vec2d) -> pymunk.Body:
        body = pymunk.Body(chain_link_config.mass, chain_link_config.radius / 2)
        body.position = start_pos
        return body

    @staticmethod
    def _generate_shape(chain_link_config: ChainLinkConfig, body: pymunk.Body) -> pymunk.Circle:
        shape = pymunk.Circle(body, chain_link_config.radius)
        shape.filter = pymunk.ShapeFilter(chain_link_config.shape_filter_group)
        return shape

    def add_pinjoint(self, body: pymunk.Body, other_body: pymunk.Body, offset: Vec2d) -> pymunk.Constraint:
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
    start_pos: Vec2d
    start_angle: float
    num_links: int
    link_length: float
    links: list[ChainLink] = field(default_factory=lambda: [])

    @property
    def top_link(self) -> ChainLink:
        return self.get_link_by_index(0)

    @property
    def seat(self) -> ChainLink:
        return self.get_link_by_index(-1)

    @classmethod
    def create(
        cls,
        swing_config: dict,
        space: pymunk.Space,
        start_pos: Vec2d,
        shape_filter_group: int,
    ) -> Swing:
        swing = cls(
            space, start_pos, swing_config["start_angle"], swing_config["num_links"], swing_config["link_length"]
        )
        swing._generate_links(swing_config["link_mass"], swing_config["link_radius"], shape_filter_group)
        return swing

    def _generate_links(self, mass: float, radius: float, shape_filter_group: int) -> None:
        link = ChainLink.static_link(self.space, self.start_pos, mass, radius, shape_filter_group)
        self.links.append(link)
        for _ in range(1, self.num_links):
            self.links.append(ChainLink.dynamic_link(self.links[-1], self.link_length, self.start_angle))

    def get_link_by_index(self, index: int) -> ChainLink:
        return self.links[index]

    def update(self) -> None:
        for link in self.links:
            link.update()

    def remove(self) -> None:
        for link in self.links:
            link.remove_from_space()
