from functools import partial
from datetime import datetime
import os
import argparse

import numpy as np
from transformers import AutoConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers.trainer_utils import SchedulerType
from datasets import load_dataset
from torchmetrics.functional.classification import binary_auroc


from measurement_pred.configuration_code_gen_mp import CodeGenMeasurementPredictorConfig
from measurement_pred.modeling_code_gen_mp import CodeGenMeasurementPredictor

# load data
def train(exp_dir):
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

    training_args = TrainingArguments(
        output_dir=exp_dir,
        logging_dir=os.path.join(exp_dir, "logs"),
        learning_rate=2e-5,
        weight_decay=2e-2,
        lr_scheduler_type=SchedulerType.COSINE,
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
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, n_senors=code_gen_mp_config.n_sensors, use_aggregated=code_gen_mp_config.use_aggregated)
    )
    trainer.train()

def train_slurm(exp_dir, slurm_params):
    import submitit
    executor = submitit.AutoExecutor(folder=exp_dir)
    executor.update_parameters(**slurm_params)
    executor.submit(
        train, 
        exp_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--submitit', action="store_true", help="train with submitit")
    parser.add_argument("--exp_dir", type=str, 
                        default=os.path.join("output", "codegen_350M_mono_mp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    
    args = parser.parse_args()

    if args.submitit:
        slurm_params = {
            "slurm_mem_gb": 80, 
            "slurm_gres": "gpu:A100-SXM4-80GB:1",
            "nodes": 1, 
            "timeout_min": 60 * 10,
            "slurm_job_name": "bash",
            "slurm_qos": "high"
        }
        train_slurm(args.exp_dir, slurm_params)
        
    else:
        train(args.exp_dir)
