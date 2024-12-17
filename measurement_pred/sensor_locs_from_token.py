import torch
from transformers import PreTrainedTokenizerBase

from .sensor_loc_finder import SensorLocFinder


class SensorLocFinderFromToken(SensorLocFinder):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, sensor_token: str, n_sensors: int):
        self.sensor_token_id = tokenizer.encode(sensor_token)[0]
        self.n_sensors = n_sensors

    def find_sensor_locs(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat_sensor_token_idxs = (input_ids == self.sensor_token_id).nonzero(as_tuple=True)[1]
        sensor_token_idxs = flat_sensor_token_idxs.view(-1, self.n_sensors)
        return sensor_token_idxs
