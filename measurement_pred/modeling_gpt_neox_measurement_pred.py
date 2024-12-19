from transformers.models.gpt_neox import GPTNeoXPreTrainedModel, GPTNeoXModel
from transformers import PreTrainedTokenizerBase
from .modeling_measurement_pred import MeasurementPredictorMixin
from .configuration_gpt_neox_measurement_pred import GPTNeoXMeasurementPredictorConfig

class GPTNeoXMeasurementPredictor(GPTNeoXPreTrainedModel, MeasurementPredictorMixin):
    config_class = GPTNeoXMeasurementPredictorConfig

    def __init__(self, config):
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)
        self.post_init()
    
    def set_pad_token(self, tokenizer: PreTrainedTokenizerBase):
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
