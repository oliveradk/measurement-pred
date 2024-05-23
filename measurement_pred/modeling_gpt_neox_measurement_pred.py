from transformers.models.gpt_neox import GPTNeoXPreTrainedModel, GPTNeoXModel

from .modeling_measurement_pred import MeasurementPredictorMixin
from .configuration_gpt_neox_measurement_pred import GPTNeoXMeasurementPredictorConfig

class GPTNeoXMeasurementPredictor(GPTNeoXPreTrainedModel, MeasurementPredictorMixin):
    config_class = GPTNeoXMeasurementPredictorConfig

    def __init__(self, config):
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel.from_pretrained(config.name_or_path, config)
        self.post_init()