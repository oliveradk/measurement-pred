from transformers.models.codegen import CodeGenPreTrainedModel, CodeGenModel
from transformers import PreTrainedTokenizerBase
from .modeling_measurement_pred import MeasurementPredictorMixin
from .configuration_code_gen_measuremet_pred import CodeGenMeasurementPredictorConfig


class CodeGenMeasurementPredictor(CodeGenPreTrainedModel, MeasurementPredictorMixin):
    config_class = CodeGenMeasurementPredictorConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CodeGenModel(config)
        self.post_init()
    
    def set_pad_token(self, tokenizer: PreTrainedTokenizerBase):
        pad_token = ' .'
        pad_token_id = tokenizer.encode(pad_token)[0]
        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = pad_token_id
