from enum import Enum

from transformers import AutoConfig, AutoTokenizer

from measurement_pred.configuration_code_gen_measuremet_pred import CodeGenMeasurementPredictorConfig
from measurement_pred.modeling_code_gen_measurement_pred import CodeGenMeasurementPredictor
from measurement_pred.configuration_gpt_neox_measurement_pred import GPTNeoXMeasurementPredictorConfig
from measurement_pred.modeling_gpt_neox_measurement_pred import GPTNeoXMeasurementPredictor

class ModelTypes(Enum):
    CODEGEN = "codegen"
    GPT_NEOX = "gpt_neox"

KEYS_TO_REMOVE = ("_name_or_path", "use_cache")

def filter_config(config: dict):
    return {k: v for k, v in config.items() if k not in KEYS_TO_REMOVE}

def load_model(model_type: ModelTypes, pretrained_model_name: str):
    if model_type in (ModelTypes.CODEGEN, ModelTypes.CODEGEN.value):
        CodeGenMeasurementPredictorConfig.register_for_auto_class()
        CodeGenMeasurementPredictor.register_for_auto_class("AutoModelForSequenceClassification")
        config = CodeGenMeasurementPredictorConfig(**filter_config(AutoConfig.from_pretrained(pretrained_model_name).to_dict()), use_cache=False)
        model = CodeGenMeasurementPredictor(config)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, padding_side="left", truncation_side="left") 
        return config, model, tokenizer
    elif model_type in (ModelTypes.GPT_NEOX, ModelTypes.GPT_NEOX.value):
        GPTNeoXMeasurementPredictorConfig.register_for_auto_class()
        GPTNeoXMeasurementPredictor.register_for_auto_class("AutoModelForSequenceClassification")
        config = GPTNeoXMeasurementPredictorConfig(**filter_config(AutoConfig.from_pretrained(pretrained_model_name).to_dict()), use_cache=False)
        model = GPTNeoXMeasurementPredictor(config)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, padding_side="left", truncation_side="left")
        return config, model, tokenizer
    else: 
        raise ValueError(f"{model_type} not supported, must be one of {[v.value for v in ModelTypes]}")
        



