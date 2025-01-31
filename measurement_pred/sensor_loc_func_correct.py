import torch
from transformers import PreTrainedTokenizerBase

from .sensor_loc_finder import SensorLocFinder

PRE_ANSWER_STR = " pass or fail or omit:"
MAX_ANSWER_COUNTS = 10


# thx gpt4
def pad_nonzero_indices(tensor: torch.Tensor, max_count: int, offset: int = 0, pad_value: int = -1):
    # Get the indices of non-zero elements
    indices = torch.nonzero(tensor, as_tuple=True)

    # Count the number of non-zero elements per row
    counts = torch.sum(tensor, dim=1)
    assert int(counts.max()) <= max_count

    # Create a mask for valid indices
    mask = torch.arange(max_count, device=tensor.device).expand(tensor.shape[0], max_count) < counts.unsqueeze(1)

    # Initialize the padded tensor with -1 (or any other padding value)
    padded_indices = torch.full((tensor.shape[0], max_count), pad_value, dtype=torch.long, device=tensor.device)

    # Fill the padded tensor with the non-zero indices using the mask
    padded_indices[mask] = indices[1] + offset

    return padded_indices

class FuncCorrectSensorLocFinder(SensorLocFinder):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        self.pre_answer_toks = tokenizer.encode(PRE_ANSWER_STR)
        self.max_answer_count = MAX_ANSWER_COUNTS

    def find_sensor_locs(self, input_ids: torch.Tensor) -> torch.Tensor:
        answer_at_loc = torch.stack(
            [input_ids[:, i : (-len(self.pre_answer_toks)) + i] == tok for i, tok in enumerate(self.pre_answer_toks)],
            dim=0,
        ).all(dim=0)
        raw_sensor_locs = pad_nonzero_indices(
            answer_at_loc, 
            offset=len(self.pre_answer_toks) - 1, 
            max_count=self.max_answer_count, 
            pad_value=0
        )
        overall_loc = torch.max(raw_sensor_locs, dim=-1).values
        sensor_locs = torch.cat([raw_sensor_locs, overall_loc.unsqueeze(-1)], dim=-1)

        return sensor_locs 
