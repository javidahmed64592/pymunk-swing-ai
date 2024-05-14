from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppConfigType:
    name: str
    width: int
    height: int
    fps: int
    font: str
    font_size: int

    @classmethod
    def from_dict(cls, data: dict) -> AppConfigType:
        return cls(data["name"], data["width"], data["height"], data["fps"], data["font"], data["font_size"])


@dataclass
class SwingConfigType:
    start_angle: float
    num_links: int
    link_length: float
    link_mass: float
    link_radius: float

    @classmethod
    def from_dict(cls, data: dict) -> SwingConfigType:
        return cls(data["start_angle"], data["num_links"], data["link_length"], data["link_mass"], data["link_radius"])


@dataclass
class HeadConfigType:
    radius: float
    mass: float
    start_angle: float
    angle_constraints: tuple[float, float]

    @classmethod
    def from_dict(cls, data: dict) -> SwingConfigType:
        return cls(data["radius"], data["mass"], data["start_angle"], data["angle_constraints"])


@dataclass
class LimbConfigType:
    length: float
    mass: float
    start_angle: float
    angle_constraints: tuple[float, float]

    @classmethod
    def from_dict(cls, data: dict) -> SwingConfigType:
        return cls(data["length"], data["mass"], data["start_angle"], data["angle_constraints"])


@dataclass
class StickmanConfigType:
    head: HeadConfigType
    neck: LimbConfigType
    upper_arm: LimbConfigType
    lower_arm: LimbConfigType
    torso: LimbConfigType
    upper_leg: LimbConfigType
    lower_leg: LimbConfigType
    foot: LimbConfigType
