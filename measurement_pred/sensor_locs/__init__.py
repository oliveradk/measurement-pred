from enum import Enum

from .stories import StoriesSensorLocFinder
from .locs_from_token import SensorLocFinderFromToken


SENSOR_LOC_REGISTRY = {
    "stories": StoriesSensorLocFinder,
    "locs_from_token": SensorLocFinderFromToken
}
