from abc import ABC, abstractmethod
import torch
from transformers import PreTrainedTokenizerBase


class SensorLocFinder(ABC):


    @abstractmethod
    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        pass

    @abstractmethod
    def find_sensor_locs(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.find_sensor_locs(input_ids)
