from transformers.models.codegen import CodeGenPreTrainedModel, CodeGenModel

from .modeling_measurement_pred import MeasurementPredictorMixin
from .configuration_code_gen_measuremet_pred import CodeGenMeasurementPredictorConfig


class CodeGenMeasurementPredictor(CodeGenPreTrainedModel, MeasurementPredictorMixin):
    config_class = CodeGenMeasurementPredictorConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CodeGenModel.from_pretrained(config.name_or_path, config=config) #TODO: test
        self.post_init()
