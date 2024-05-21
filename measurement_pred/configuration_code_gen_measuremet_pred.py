from transformers.models.codegen import CodeGenConfig
from .configuration_measurement_pred import MeasurementPredictorConfig

class CodeGenMeasurementPredictorConfig(MeasurementPredictorConfig, CodeGenConfig):
    model_type = "codegen_mp"
    def __init__(self, **kwargs):
        kwargs["sensor_token_id"] = 42848
        super().__init__(**kwargs)
    
    def get_emb_dim(self):
        return self.n_embd
