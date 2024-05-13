from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pymunk
from pymunk.vec2d import Vec2d

from src.constants import AIR_DENSITY, DRAG_COEFFICIENT, NORM_VECTOR


@dataclass
class ChainLink:
    space: pymunk.Space
    start_pos: Vec2d
    mass: float
    radius: float
    shape_filter_group: int
    constraints: list[pymunk.Constraint] = field(default_factory=lambda: [])
    body: pymunk.Body = None
    shape: pymunk.Circle = None

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
        link = cls(space, start_pos, mass, radius, shape_filter_group)
        link._generate_body()
        link._generate_shape()
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
        pos = other_link.body.position + (NORM_VECTOR.rotated(angle * np.pi / 180) * length)
        link = cls.generate(other_link.space, pos, other_link.mass, other_link.radius, other_link.shape_filter_group)
        link.add_pinjoint(other_link.body, link.body, Vec2d(0, 0))
        return link

    def _generate_body(self) -> pymunk.Body:
        self.body = pymunk.Body(self.mass, self.radius / 2)
        self.body.position = self.start_pos
        self.space.add(self.body)
        return self.body

    def _generate_shape(self) -> pymunk.Circle:
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.filter = pymunk.ShapeFilter(self.shape_filter_group)
        self.space.add(self.shape)
        return self.shape

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
    pos: Vec2d
    num_links: int
    link_length: float
    start_angle: float
    links: list[ChainLink] = field(default_factory=lambda: [])

    @property
    def top_link(self) -> ChainLink:
        return self.get_link_by_index(0)

    @property
    def seat(self) -> ChainLink:
        return self.get_link_by_index(-1)

    @classmethod
    def generate(
        cls,
        space: pymunk.Space,
        pos: Vec2d,
        num_links: int,
        link_length: float,
        start_angle: float,
        mass: float,
        radius: float,
        shape_filter_group: int,
    ) -> Swing:
        swing = cls(space, pos, num_links, link_length, start_angle)
        swing._generate_links(mass, radius, shape_filter_group)
        return swing

    def _generate_links(self, mass: float, radius: float, shape_filter_group: int) -> None:
        link = ChainLink.static_link(self.space, self.pos, mass, radius, shape_filter_group)
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
