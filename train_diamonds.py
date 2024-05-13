from functools import partial

import numpy as np
from transformers import AutoConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers.optimization import get_cosine_schedule_with_warmup
from datasets import load_dataset
from torchmetrics.functional.classification import binary_auroc


from measurement_pred.configuration_code_gen_mp import CodeGenMeasurementPredictorConfig
from measurement_pred.modeling_code_gen_mp import CodeGenMeasurementPredictor

# load data
dataset = load_dataset("redwoodresearch/diamonds-seed0")

# load config
code_gen_tiny_config = AutoConfig.from_pretrained("Salesforce/codegen-350M-mono")
code_gen_mp_config = CodeGenMeasurementPredictorConfig(**code_gen_tiny_config.to_dict())

# load model
code_gen_mp = CodeGenMeasurementPredictor(code_gen_mp_config)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token 

# tokenize dataset
max_length = 1024
padding="max_length"
return_tensors = "pt"
def tokenize_dataset(dataset):
    return tokenizer(
        dataset["text"], 
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors
    )
dataset = dataset.map(tokenize_dataset, batched=True)
# construct data collator
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, 
    padding=padding,
    max_length=max_length,
    return_tensors=return_tensors
)

# define metrics
def compute_metrics(eval_preds, n_sensors: int, use_aggregated: bool):
    logits, labels = eval_preds
    preds = np.round(logits)
    metrics = {}
    metrics["accuracy"] = np.mean(preds == labels)
    for i in range(n_sensors):
        metrics[f"accuracy_sensor_{i}"] = np.mean(preds[..., i], labels[..., i])
        metrics[f"auroc_sensor_{i}"] = binary_auroc(preds[..., i], labels[..., i])
    if use_aggregated:
        metrics[f"accuracy_aggregated"] = np.mean(preds[..., -1], labels[..., -1])
        metrics[f"auroc_aggregated"] = binary_auroc(preds[..., -1], labels[..., -1])

    return metrics


# train 
training_args = TrainingArguments(
    output_dir="output/codegen_350M_mono_mp/",
    learning_rate=2e-5,
    weight_decay=2e-2,
    lr_scheduler_type=get_cosine_schedule_with_warmup,
    warmup_steps=64,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    fp16=True
)

trainer = Trainer(
    model=code_gen_mp,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=partial(compute_metrics, n_senors=code_gen_mp_config.n_sensors, use_aggregated=code_gen_mp_config.use_aggregated)
)

if __name__ == "__main__":
    trainer.train()
