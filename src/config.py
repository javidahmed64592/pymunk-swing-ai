import json
import os
from pathlib import Path

from src.data_types import HeadConfigType, LimbConfigType, StickmanConfigType, SwingConfigType

CONFIG_FOLDER = Path(os.path.realpath(__file__)).parent.parent / "config"


def load_json(filepath: Path) -> dict:
    with open(filepath) as file:
        data = json.load(file)
    return data


def get_swing_config() -> SwingConfigType:
    swing_data = load_json(CONFIG_FOLDER / "swing.json")
    swing_config = SwingConfigType.from_dict(swing_data)
    return swing_config


def get_stickman_config() -> StickmanConfigType:
    stickman_data = load_json(CONFIG_FOLDER / "stickman.json")
    head = HeadConfigType.from_dict(stickman_data["head"])
    neck = LimbConfigType.from_dict(stickman_data["neck"])
    upper_arm = LimbConfigType.from_dict(stickman_data["upper_arm"])
    lower_arm = LimbConfigType.from_dict(stickman_data["lower_arm"])
    torso = LimbConfigType.from_dict(stickman_data["torso"])
    upper_leg = LimbConfigType.from_dict(stickman_data["upper_leg"])
    lower_leg = LimbConfigType.from_dict(stickman_data["lower_leg"])
    foot = LimbConfigType.from_dict(stickman_data["foot"])
    return StickmanConfigType(head, neck, upper_arm, lower_arm, torso, upper_leg, lower_leg, foot)
