from transformers.models.gpt_neox import GPTNeoXConfig
from .configuration_measurement_pred import MeasurementPredictorConfig


class GPTNeoXMeasurementPredictorConfig(MeasurementPredictorConfig, GPTNeoXConfig):
    model_type = "gpt_neox_mp"
    def __init__(self, **kwargs):
        kwargs["sensor_token_id"] = 35991
        super().__init__(**kwargs)
    
    def get_emb_dim(self):
        return self.hidden_size