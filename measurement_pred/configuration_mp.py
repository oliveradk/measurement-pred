from transformers import PretrainedConfig

class MeasurementPredictorConfig(PretrainedConfig):
    
    def __init__(
        self, 
        sensor_token=" omit",
        sensor_token_id=None, # 35991
        n_sensors=3,
        use_aggregated=True,
        sensors_weight = 0.7,
        aggregate_weight=0.3,
        **kwargs
    ):
        self.sensor_token = sensor_token 
        self.sensor_token_id = sensor_token_id
        self.n_sensors = n_sensors
        self.use_aggregated = use_aggregated
        self.sensors_weight = sensors_weight
        self.aggregate_weight = aggregate_weight
        super().__init__(**kwargs)