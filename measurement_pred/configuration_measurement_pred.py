from abc import abstractmethod
from transformers import PretrainedConfig
class MeasurementPredictorConfig(PretrainedConfig):
    
    def __init__(
        self, 
        sensor_token=" omit",
        sensor_loc_type="locs_from_token",
        n_sensors=3,
        sensors_weight = 0.7,
        aggregate_weight=0.3,
        shared_probe=False,
        add_eos_token=False,
        no_mask=False,
        **kwargs
    ):
        self.sensor_token = sensor_token 
        self.sensor_loc_type = sensor_loc_type
        self.n_sensors = n_sensors
        self.sensors_weight = sensors_weight
        self.aggregate_weight = aggregate_weight
        self.shared_probe = shared_probe
        self.add_eos_token = add_eos_token
        self.no_mask = no_mask
        super().__init__(**kwargs)
        self.emb_dim = self.get_emb_dim()
    
    @abstractmethod
    def get_emb_dim(self):
        raise NotImplementedError