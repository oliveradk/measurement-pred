from enum import Enum
import os

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from measurement_pred.configuration_code_gen_measuremet_pred import CodeGenMeasurementPredictorConfig
from measurement_pred.modeling_code_gen_measurement_pred import CodeGenMeasurementPredictor
from measurement_pred.configuration_gpt_neox_measurement_pred import GPTNeoXMeasurementPredictorConfig
from measurement_pred.modeling_gpt_neox_measurement_pred import GPTNeoXMeasurementPredictor



class ModelTypes(Enum):
    CODEGEN = "codegen"
    GPT_NEOX = "gpt_neox"
    AUTO_MODEL_FOR_SEQUENCE_CLASSIFICATION = "auto_model_for_sequence_classification"

KEYS_TO_REMOVE = ("use_cache")

def filter_config(config: dict):
    return {k: v for k, v in config.items() if k not in KEYS_TO_REMOVE}


def get_model_id(base_name: str, dataset_name: str):
    return "-".join([
        os.path.basename(base_name),
        "measurement_pred",
        os.path.basename(dataset_name)
    ])

def load_model(model_type: ModelTypes, pretrained_model_name: str, config_params: dict={}):
    if model_type in (ModelTypes.CODEGEN, ModelTypes.CODEGEN.value):
        CodeGenMeasurementPredictorConfig.register_for_auto_class()
        CodeGenMeasurementPredictor.register_for_auto_class("AutoModelForSequenceClassification")
        base_model_config = AutoConfig.from_pretrained(pretrained_model_name, **config_params)
        config = CodeGenMeasurementPredictorConfig(**filter_config(base_model_config.to_dict()), **config_params, use_cache=False)
        model = CodeGenMeasurementPredictor.from_pretrained(pretrained_model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, padding_side="left", truncation_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        return config, model, tokenizer
    elif model_type in (ModelTypes.GPT_NEOX, ModelTypes.GPT_NEOX.value):
        GPTNeoXMeasurementPredictorConfig.register_for_auto_class()
        GPTNeoXMeasurementPredictor.register_for_auto_class("AutoModelForSequenceClassification")
        base_model_config = AutoConfig.from_pretrained(pretrained_model_name)
        config = GPTNeoXMeasurementPredictorConfig(**filter_config(base_model_config.to_dict()), **config_params, use_cache=False)
        model = GPTNeoXMeasurementPredictor.from_pretrained(pretrained_model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, padding_side="left", truncation_side="left")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.pad_token_id = tokenizer.pad_token_id
        return config, model, tokenizer
    elif model_type in (ModelTypes.AUTO_MODEL_FOR_SEQUENCE_CLASSIFICATION, ModelTypes.AUTO_MODEL_FOR_SEQUENCE_CLASSIFICATION.value):
        config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True, **config_params)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, config=config, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, padding_side="left", truncation_side="left", trust_remote_code=True)
        print("Warning: pad token not set")
        return config, model, tokenizer
    else: 
        raise ValueError(f"{model_type} not supported, must be one of {[v.value for v in ModelTypes]}")
        



