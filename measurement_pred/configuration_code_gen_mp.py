from transformers.models.codegen import CodeGenConfig
from .configuration_mp import MeasurementPredictorConfig

class CodeGenMeasurementPredictorConfig(CodeGenConfig, MeasurementPredictorConfig):
    model_type = "codegen-mp"
    def __init__(self, **kwargs):
        kwargs["sensor_token_id"] = 35991
        super().__init__(**kwargs)
