from abc import abstractmethod
from transformers import PretrainedConfig
class MeasurementPredictorConfig(PretrainedConfig):
    
    def __init__(
        self, 
        sensor_token=" omit",
        sensor_loc_type="locs_from_token",
        n_sensors=3,
        use_aggregated=True,
        sensors_weight = 0.7,
        aggregate_weight=0.3,
        **kwargs
    ):
        self.sensor_token = sensor_token 
        self.sensor_loc_type = sensor_loc_type
        self.n_sensors = n_sensors
        self.use_aggregated = use_aggregated
        self.sensors_weight = sensors_weight
        self.aggregate_weight = aggregate_weight
        super().__init__(**kwargs)
        self.emb_dim = self.get_emb_dim()
    
    @abstractmethod
    def get_emb_dim(self):
        raise NotImplementedError