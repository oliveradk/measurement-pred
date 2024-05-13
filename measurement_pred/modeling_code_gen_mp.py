from transformers.models.codegen import CodeGenPreTrainedModel

from .modeling_mp import MeasurementPredictorMixin
from .configuration_code_gen_mp import CodeGenMeasurementPredictorConfig


class CodeGenMeasurementPredictor(CodeGenPreTrainedModel, MeasurementPredictorMixin):
    config_class = CodeGenMeasurementPredictorConfig
