from transformers.models.codegen import CodeGenPreTrainedModel, CodeGenModel

from .modeling_mp import MeasurementPredictorMixin
from .configuration_code_gen_mp import CodeGenMeasurementPredictorConfig


class CodeGenMeasurementPredictor(CodeGenPreTrainedModel, MeasurementPredictorMixin):
    config_class = CodeGenMeasurementPredictorConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CodeGenModel(config)
        self.post_init()
