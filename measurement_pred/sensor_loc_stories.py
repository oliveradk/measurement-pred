import torch
from transformers import PreTrainedTokenizerBase

from .sensor_loc_finder import SensorLocFinder


class StoriesSensorLocFinder(SensorLocFinder):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        self.questions_section_toks = tokenizer.encode("## Questions")
        self.question_mark_tok = tokenizer.encode("?")[0]
        self.other_question_mark_tok = tokenizer.encode(")?")[0]
        assert len(self.questions_section_toks) == 2

    def find_sensor_locs(self, input_ids: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        question_mark_locs = self._is_sensor_loc(input_ids)
        total_locs = torch.cumsum(question_mark_locs, dim=-1)
        total_overall = total_locs[:, -1]
        assert (
            total_overall == 3
        ).all(), "can handle different cases, but assuming this is easiest"
        eqs = total_locs[:, :, None] == torch.arange(1, 4)[None, None].to(device)
        locs = torch.where(
            eqs.any(dim=-2),
            torch.argmax(eqs.to(torch.uint8), dim=-2),
            input_ids.shape[-1] - 3,
        ).clamp(max=input_ids.shape[-1] - 3)
        aggregate_sensor_loc = locs[:, -1].unsqueeze(1)
        locs = torch.cat([locs, aggregate_sensor_loc], dim=1)
        return locs

    
    def _is_sensor_loc(self, input_ids: torch.Tensor):
        questions_section_toks = self.questions_section_toks
        question_mark_tok = self.question_mark_tok
        other_question_mark_tok = self.other_question_mark_tok
        eq_question_item = (input_ids[:, :-1] == questions_section_toks[0]) & (
            input_ids[:, 1:] == questions_section_toks[1]
        )
        assert (eq_question_item.sum(dim=-1, dtype=torch.int) == 1).all(), "could relax"

        summed = torch.cumsum(
            torch.cat([eq_question_item, eq_question_item[:, -1:]], dim=-1), dim=-1
        )
        return (summed > 0) & (
            (input_ids == question_mark_tok) | (input_ids == other_question_mark_tok)
        )
