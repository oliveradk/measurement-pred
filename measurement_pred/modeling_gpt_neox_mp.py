from transformers.models.gpt_neox import GPTNeoXPreTrainedModel, GPTNeoXModel

from .modeling_mp import MeasurementPredictorMixin
from .configuration_gpt_neox_mp import GPTNeoXMeasurementPredictorConfig

class GPTNeoXMeasurementPredictor(GPTNeoXPreTrainedModel, MeasurementPredictorMixin):
    config_class = GPTNeoXMeasurementPredictorConfig

    def __init__(self, config):
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)
        self.post_init()