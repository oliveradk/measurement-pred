from enum import Enum

from .sensor_loc_stories import StoriesSensorLocFinder
from .sensor_locs_from_token import SensorLocFinderFromToken


SENSOR_LOC_REGISTRY = {
    "stories": StoriesSensorLocFinder,
    "locs_from_token": SensorLocFinderFromToken
}
