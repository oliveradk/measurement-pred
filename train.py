from functools import partial
from datetime import datetime
import os

import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from torchmetrics.functional.classification import binary_auroc
from torchmetrics.functional.classification import multilabel_accuracy, binary_accuracy

import hydra
from omegaconf import DictConfig

from models import load_model

# load data
@hydra.main(config_path="conf", config_name="codegen_diamonds_slurm")
def train(cfg: DictConfig):
    # set exp_dir
    exp_dir = "." # hydra/slurm takes care of this os.path.join(cfg.model.model_type, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    # load data
    dataset = load_dataset(cfg.model.dataset_name)
    
    # get subset (for testing)
    if "dataset_len" in cfg:
        for k, subset in dataset.items():
            dataset[k] = subset.select(range(cfg.dataset_len))

    def add_measurement_labels(dataset):
        labels = dataset["measurements"] + [all(dataset["measurements"])]
        labels = [float(label) for label in labels]
        dataset["labels"] = labels
        return dataset
    dataset = dataset.map(add_measurement_labels)

    # load model
    model_config, model, tokenizer = load_model(cfg.model.model_type, cfg.model.pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # confirm sensor token is correct
    model.check_tokenizer(tokenizer)

    # tokenize dataset
    max_length = 1024
    padding="max_length"
    return_tensors = "pt"
    
    def tokenize_dataset(dataset):
        return tokenizer(
            dataset["text"], 
            max_length=max_length,
            padding=padding,
            truncation=True,
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
        logits = torch.tensor(logits)
        labels = torch.tensor(labels, dtype=torch.int)
        metrics = {}
        metrics["accuracy"] = multilabel_accuracy(logits, labels, n_sensors + 1)
        for i in range(n_sensors):
            metrics[f"accuracy_sensor_{i}"] = binary_accuracy(logits[..., i], labels[..., i])
            metrics[f"auroc_sensor_{i}"] = binary_auroc(torch.tensor(logits[..., i]), torch.tensor(labels[..., i]))
        if use_aggregated:
            metrics[f"accuracy_aggregated"] = binary_accuracy(logits[...,-1], labels[...,-1])
            metrics[f"auroc_aggregated"] = binary_accuracy(logits[...,-1], labels[...,-1])

        return metrics
    training_args = TrainingArguments(
        output_dir=exp_dir,
        logging_dir=os.path.join(exp_dir, "logs"),
        learning_rate=cfg.hparams.learning_rate,
        weight_decay=cfg.hparams.weight_decay,
        lr_scheduler_type=cfg.hparams.lr_scheduler_type,
        warmup_steps=cfg.hparams.warmup_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.hparams.effective_batch_size // cfg.per_device_train_batch_size, 
        num_train_epochs=cfg.hparams.num_train_epochs,
        fp16=cfg.fp16,
        logging_steps=1024,
        eval_strategy="epoch"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"], #TODO: fix dataset (need to format y's, add as util)
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, n_sensors=model_config.n_sensors, use_aggregated=model_config.use_aggregated)
    )
    trainer.train()

if __name__ == "__main__":
    train()
