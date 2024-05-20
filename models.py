from enum import Enum

from transformers import AutoConfig, AutoTokenizer

from measurement_pred.configuration_code_gen_mp import CodeGenMeasurementPredictorConfig
from measurement_pred.modeling_code_gen_mp import CodeGenMeasurementPredictor
from measurement_pred.configuration_gpt_neox_mp import GPTNeoXMeasurementPredictorConfig
from measurement_pred.modeling_gpt_neox_mp import GPTNeoXMeasurementPredictor

class ModelTypes(Enum):
    CODEGEN = "codegen"
    GPT_NEOX = "gpt_neox"

def load_model(model_type: ModelTypes, pretrained_model_name: str):
    if model_type in (ModelTypes.CODEGEN, ModelTypes.CODEGEN.value):
        config = CodeGenMeasurementPredictorConfig(**AutoConfig.from_pretrained(pretrained_model_name).to_dict())
        model = CodeGenMeasurementPredictor(config)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, padding_side="left", truncation_side="left") 
        return config, model, tokenizer
    elif model_type in (ModelTypes.GPT_NEOX, ModelTypes.GPT_NEOX.value):
        config = GPTNeoXMeasurementPredictorConfig(**AutoConfig.from_pretrained(pretrained_model_name).to_dict())
        model = GPTNeoXMeasurementPredictor(config)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, padding_side="left", truncation_side="left")
        return config, model, tokenizer
    else: 
        raise ValueError(f"{model_type} not supported, must be one of {[v.value for v in ModelTypes]}")
        


