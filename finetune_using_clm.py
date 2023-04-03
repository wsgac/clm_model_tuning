# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 03:08:09 2023

@author: HP
"""

# !/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

import logging
import math
import os
import random
from itertools import chain
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from huggingface_hub import Repository

import datasets
import hydra
import torch
import transformers
from accelerate import Accelerator, DistributedType, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
    GPTNeoXForCausalLM,
    AutoTokenizer
)

import bittensor
import random
import re


# deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, gradient_accumulation_steps=4)

def preprocess_text(text):
    unicode_characters = {r'\\u003e': '>',
                          r'\\u0026': '&',
                          r'\\u003c': '<',
                          r'\\u2029': '\n',
                          }

    for code, value in unicode_characters.items():
        text = re.sub(code, value, text)
    #     text = re.sub(r'\\u[a-f0-9]{4}', "", text)
    text = text.replace('\\n\\n\\n\\n', '|<-NL->|')
    text = text.replace('\\n\\n\\n', '|<-NL->|')
    text = text.replace('\\n\\n', '|<-NL->|')
    text = text.replace(':\\n-', ':-')
    text = text.replace(' \\n', ' ')
    text = text.replace("\\n", " ")
    text = text.replace("|<-NL->|", "\\n")
    #     text = text.replace('\\n', ' ')
    #     text = re.sub(r'-{6,1000}', r'-----', text)
    #     #text = re.sub(r' -----{1,10}', r'-----', text)
    #     text = re.sub(r'={6,20}',r'=====', text)
    #     text = re.sub(r'\s?\\{2,10}', r'\\', text)
    #     text = re.sub(r'\s?\\{1,10}\s+', r'', text)

    # text = text.replace(r'\\u', r'\u')
    # text = re.sub()
    # text = text.replace('\\n', ' ')
    # text = text.replace('\\"', '\"')

    return text


@hydra.main(version_base=None, config_path="conf", config_name="config")
def check_cfg_and_load_defaults(cfg: DictConfig) -> DictConfig:
    subtensor = bittensor.subtensor(network=cfg.bittensor.network)
    if cfg.dataset.block_size is None:
        cfg.dataset.block_size = subtensor.validator_sequence_length
    if cfg.training.train_batch_size is None:
        cfg.training.train_batch_size = subtensor.validator_batch_size
    if cfg.training.eval_batch_size is None:
        cfg.training.eval_batch_size = subtensor.validator_batch_size

    return cfg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def load_raw_datasets(cfg: DictConfig, separator="||--<<END>>--||") -> DatasetDict:
    if not os.path.exists("train_data"):
        os.mkdir("train_data")

    cfg = check_cfg_and_load_defaults(cfg)

    if cfg.dataset.name == "bittensor":

        dataset = bittensor.dataset(
            no_tokenizer=True,
            batch_size=cfg.training.train_batch_size,
            block_size=cfg.dataset.block_size,
            num_batches=cfg.dataset.num_batches,
            save_dataset=True
        )
        dataloader = dataset.dataloader(cfg.dataset.num_batches * cfg.dataset.n_shards)
        bittensor_dataset = {"text": []}
        shard_count = 0
        samples_count = 0
        print("Downloading the dataset....")
        for batch in tqdm(dataloader, desc="Loading data from bittensor IPFS"):
            bittensor_dataset["text"].extend(batch)
            samples_count += len(batch)

            if samples_count == cfg.dataset.n_samples:
                print("Saving downloaded data into text file..")
                parent_path = os.path.join(os.path.abspath("train_data"), "new")
                path = os.path.join(parent_path, f"{cfg.bittensor.network}_train_sharded-{shard_count}.txt")
                with open(path, "w") as f:
                    f.write(separator.join(bittensor_dataset["text"]))

                bittensor_dataset["text"] = []
                shard_count += 1
                samples_count = 0

        # raw_datasets = Dataset.from_dict(bittensor_dataset)

        dataset.close()  # Avoid leaving threadqueue running.
        return
    #         return raw_datasets

    if os.path.exists(cfg.dataset.name):
        data_files = {"text": cfg.dataset.name}
        dataset_args = {}

        extension = os.path.splitext(cfg.dataset.name)[-1].lstrip(".")

        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = cfg.dataset.keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        raw_datasets = raw_datasets["text"]
    else:
        raw_datasets = load_dataset(cfg.dataset.name, cfg.dataset.config_name)

    return raw_datasets


def load_text(file_path, sep):
    with open(file_path, 'r') as f:
        train_dataset = f.read().split(sep)
        for each in train_dataset:
            preprocess_text(each)
        train_dataset = Dataset.from_dict(
            {"text": random.sample(train_dataset, len(train_dataset))})  # len(train_dataset)

    return train_dataset


# def split_data(file, sep= "||--<<END>>--||" ):
#     print(f"Loading data from {file}")
#     with open(file, "r") as f:
#         dataset = f.read().split(sep)

#     train_dataset = dataset[:1500000]
#     eval_dataset = dataset[1500000:]

#     print(f"Number of training samples: {len(train_dataset)}")
#     print(f"Number of samples in evaluation : {len(eval_dataset)}")

#     train_dataset = sep.join(train_dataset)
#     eval_dataset = sep.join(eval_dataset)

#     eval_name = file.replace("train", "eval")
#     train_name = file.replace("train", "Training")
#     with open(train_name, "w") as f:
#         f.write(train_dataset)
#     with open(eval_name, "w") as f:
#         f.write(eval_dataset)

#     return

def split_data(file, sep="||--<<END>>--||"):
    if not os.path.exists("train_data"):
        os.mkdir("train_data")

    with open(file, "r") as f:
        dataset = f.read().split(sep)
        for count, start in enumerate(range(0, 1500000, 300000)):
            data_batch = dataset[start: start + 300000]
            print(len(data_batch))
            with open(os.path.join(os.path.abspath("train_data"), f"train_shard-{count}.txt"), "w") as f:
                f.write(sep.join(data_batch))
    return


def load_model_and_tokenizer(cfg: DictConfig):
    if cfg.load_dir is not None:
        print('Loading model weights from specified file...')
        path = os.path.join(os.path.abspath(os.curdir), cfg.output_dir)
        path = os.path.join(path, cfg.load_dir)
        # config_file = os.path.join(path, cfg.config)

        # config = AutoConfig.from_pretrained(config_file)

        if cfg.tokenizer.name is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.tokenizer.name, use_fast=cfg.tokenizer.use_fast
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.name, use_fast=cfg.tokenizer.use_fast
            )
        # tokenizer.pad_token = cfg.tokenizer.pad_token
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            path,
            # use_auth_token=True
            # from_tf=bool(".ckpt" in cfg.model.name),
            # config=config,
        )
        model.resize_token_embeddings(len(tokenizer))
        # print("Successfully loaded model and tokenizer...")



    else:
        if cfg.model.config_name is not None:
            config = AutoConfig.from_pretrained(cfg.model.config_name)
        else:
            config = AutoConfig.from_pretrained(cfg.model.name)

        if cfg.tokenizer.name is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.tokenizer.name, use_fast=cfg.tokenizer.use_fast
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.name, use_fast=cfg.tokenizer.use_fast
            )
        # tokenizer.pad_token = cfg.tokenizer.pad_token
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            from_tf=bool(".ckpt" in cfg.model.name),
            config=config,
        )
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def preprocess(cfg, accelerator, tokenizer, raw_datasets):
    # First we tokenize all the texts.
    column_names = raw_datasets.column_names
    text_column_name = "text" if "text" in column_names else column_names["train"][0]
    if cfg.dataset.concatenate_raw is True:
        pad = False
    else:
        pad = "max_length"

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= cfg.dataset.block_size:
            total_length = (
                                   total_length // cfg.dataset.block_size
                           ) * cfg.dataset.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i: i + cfg.dataset.block_size]
                for i in range(0, total_length, cfg.dataset.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def tokenize_fn(examples):
        result = tokenizer(
            examples[text_column_name],
            padding=pad,
            truncation=True,
            max_length=cfg.dataset.block_size,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():

        tokenized_datasets = raw_datasets.map(
            tokenize_fn,
            batched=True,
            num_proc=cfg.tokenizer.preprocessing_num_workers,
            load_from_cache_file=not cfg.dataset.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        if cfg.dataset.concatenate_raw is True:
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=cfg.tokenizer.preprocessing_num_workers,
                load_from_cache_file=not cfg.dataset.overwrite_cache,
                desc=f"Grouping texts in chunks of {cfg.dataset.block_size}",
            )

    return tokenized_datasets


# New Code #
def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


# New Code #
def load_training_checkpoint(model, load_dir, tag=None, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    _, checkpoint_state_dict = model.load_checkpoint(load_dir, tag=tag, **kwargs)
    epoch = checkpoint_state_dict["epoch"]
    last_global_step = checkpoint_state_dict["last_global_step"]
    del checkpoint_state_dict
    return (epoch, last_global_step)


def evaluate(cfg, model, eval_dataloader, accelerator, device, tokenizer, to_text=True):
    model.eval()
    #     losses = []
    eval_losses = []
    inputs = []
    label = []
    output = []

    with torch.no_grad():
        for _eval_step, eval_batch in enumerate(eval_dataloader):
            # outputs = model(**eval_batch)

            # print("Number of samples in eval batch: ", len(eval_batch["input_ids"]))
            input_ids = eval_batch["input_ids"].to(device)
            labels = eval_batch["labels"].to(device)
            attention_mask = eval_batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

            loss = outputs.loss
            # if _eval_step < 2:
            # print(f"Loss shape: {loss.shape}")
            # print(f"Loss tensor: {loss}")

            #             print(f"Loss repeat shape: {loss.repeat(cfg.training.eval_batch_size).shape}")
            #             print(f"Loss repeat tensor: {loss.repeat(cfg.training.eval_batch_size)}")

            gathered_loss = accelerator.gather(loss).detach().cpu()
            eval_losses.append(gathered_loss)
            # print("Eval losses list: ", eval_losses)

            # accelerator.wait_for_everyone()
    #             if to_text:
    #                 for each in input_ids:
    #                     inputs.append(tokenizer.decode(each))
    #                 #label.extend(list(tokenizer.decode(labels).cpu()))
    #                 predictions = torch.argmax(outputs.logits, axis=-1)
    #                 for each in predictions:
    #                     output_text = tokenizer.decode(each)
    #                     output.append(output_text)
    losses = torch.cat(eval_losses)
    # print(f"Concatenated losses shape: {losses.shape}" )
    # print(f"Concatenated losses tensor: {losses[:500]}")
    #     losses = losses[: eval_length]
    #     print(f"Eval_length: {eval_length}")
    #     print(f"Losses shape: {losses.shape}")
    #     print(f"Losses tensor: {losses[:500]}")
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    #     if to_text:
    #         with open("eval_inputs.txt", "a") as f:
    #             f.write("||--<<END>>--||".join(inputs))

    # #         with open("eval_labels.txt", "w") as f:
    # #             f.write("||--<<END>>--||".join(label))

    #         with open("eval_outputs.txt", "a") as f:
    #             f.write("||--<<END>>--||".join(output))

    return perplexity, eval_loss


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig, separator="||--<<END>>--||"):
    cfg = check_cfg_and_load_defaults(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    #     accelerator = (
    #         Accelerator(log_with=cfg.tracking.report_to, logging_dir=cfg.output_dir)
    #         if cfg.tracking.enabled
    #         else Accelerator(fp16=True, deepspeed_plugin=deepspeed_plugin)
    #         #else Accelerator()
    #     )
    accelerator = (
        Accelerator(log_with=cfg.tracking.report_to,
                    logging_dir=cfg.output_dir) if cfg.tracking.enabled else Accelerator()
    )

    accelerator.wait_for_everyone()
    device = accelerator.device

    # if cfg.training.seed is not None:
    #         logger.info(f"Setting random seed to {cfg.training.seed}")
    #         set_seed(cfg.training.seed)

    logger.info(accelerator.state, main_process_only=False)
    logger.info(OmegaConf.to_yaml(cfg))

    tokenizer, model = load_model_and_tokenizer(cfg)
    # optimizer = create_optimizer(cfg, model)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # New Code #
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
           or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=cfg.training.learning_rate)

    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=cfg.training.max_train_steps, warmup_num_steps=cfg.training.lr_warmup_steps
        )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # for file in os.listdir
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.training.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_metric = None
    best_metric_checkpoint = None

    # New Code
    # Get gradient accumulation steps from deepspeed config if available
    if accelerator.state.deepspeed_plugin is not None:
        cfg.training.gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ]

    # print(files)
    # n_files = len(files)

    eval_dataset = load_text("eval_set.txt", separator)
    eval_dataset = preprocess(cfg, accelerator, tokenizer, eval_dataset)
    eval_length = len(eval_dataset)

    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=cfg.training.eval_batch_size,
    )

    train_begin = True
    # epoch_progress = tqdm(range(starting_epoch, cfg.training.num_epochs)
    for epoch in tqdm(range(starting_epoch, cfg.training.num_epochs)):
        files = [os.path.join(os.path.abspath("train_data"), each) for each in os.listdir(os.path.abspath("train_data"))
                 if each.endswith(".txt")]
        n_files = len(files)

        file_seen = []
        model.train()
        file_count = 0
        for file in files:
            if file in file_seen:
                continue

            file_seen.append(file)
            logger.info(f"\n\nLoading and processing text file: {file} for fine tuning.\n\n")
            # Load and preprocess data
            train_dataset = load_text(file, separator)
            # raw_datasets = load_raw_datasets(cfg)
            train_dataset = preprocess(cfg, accelerator, tokenizer, train_dataset)

            if train_begin:
                # Log a few random samples from the training set:
                for index in random.sample(range(len(train_dataset)), 3):
                    ex = train_dataset[index]
                    logger.info(f"Sample {index} of the training set: {ex}: \n")
                    logger.info(tokenizer.decode(ex["input_ids"]))

                # Log a few random samples from the training set:
                for index in random.sample(range(len(eval_dataset)), 3):
                    ex = eval_dataset[index]
                    logger.info(f"Sample {index} of the evaluation set: {ex}: \n")
                    logger.info(tokenizer.decode(ex["input_ids"]))

            # DataLoaders creation:
            train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=default_data_collator,
                batch_size=cfg.training.train_batch_size,
            )

            train_length = len(train_dataset)
            #             if eval_dataset is n:
            #                 eval_length = len(eval_dataset)

            del train_dataset
            if train_begin:
                del eval_dataset

            if train_begin:
                # Prepare everything using our accelerator
                (
                    model,
                    optimizer,
                    train_dataloader,
                    eval_dataloader,
                    lr_scheduler,
                ) = accelerator.prepare(
                    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
                )
            else:
                # Prepare everything using our accelerator
                (
                    train_dataloader,
                    eval_dataloader,
                ) = accelerator.prepare(
                    train_dataloader, eval_dataloader
                )

            if train_begin:
                num_update_steps_per_epoch = math.ceil(
                    len(train_dataloader) * n_files / cfg.training.gradient_accumulation_steps)
                if cfg.training.max_train_steps is None:
                    cfg.training.max_train_steps = cfg.training.num_epochs * num_update_steps_per_epoch
                else:
                    cfg.training.num_epochs = math.ceil(cfg.training.max_train_steps / num_update_steps_per_epoch)
                # We need to recalculate our total training steps as the size of the training dataloader may have changed.
                num_update_steps_per_epoch = math.ceil(
                    len(train_dataloader) * n_files / cfg.training.gradient_accumulation_steps)
                cfg.training.max_train_steps = cfg.training.num_epochs * num_update_steps_per_epoch

            # Potentially load in the weights and states from a previous save
            if train_begin and cfg.training.checkpoint.resume_from_checkpoint:
                # New Code #
                # Loads the DeepSpeed checkpoint from the specified path
                _, last_global_step = load_training_checkpoint(
                    model,
                    cfg.training.checkpoint.resume_from_checkpoint,
                    **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
                )
                accelerator.print(f"Resumed from checkpoint: {cfg.training.checkpoint.resume_from_checkpoint}")
                resume_step = last_global_step
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

            #          # Figure out how many steps we should save the Accelerator states
            #             if hasattr(cfg.training.checkpoint.checkpointing_steps, "isdigit"):
            #                 checkpointing_steps = cfg.training.checkpoint.checkpointing_steps
            #                 if cfg.training.checkpoint.checkpointing_steps.isdigit():
            #                     checkpointing_steps = int(cfg.training.checkpoint.checkpointing_steps)
            #             else:
            #                 checkpointing_steps = None

            # total_batch_size = cfg.training.train_batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps
            if train_begin:
                logger.info("***** Running training *****")
                logger.info(f"  Num examples = {train_length * n_files}")
                logger.info(f"  Num Epochs = {cfg.training.num_epochs}")
                logger.info(f"  Instantaneous batch size per device = {cfg.training.train_batch_size}")
                #     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
                logger.info(f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}")
                logger.info(f"  Total optimization steps = {cfg.training.max_train_steps}")
                train_begin = False

            if cfg.tracking:
                total_loss = 0
            train_losses = []
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if cfg.training.checkpoint.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                # outputs = model(**batch)
                loss = outputs.loss
                train_losses.append(
                    accelerator.gather(loss.repeat(cfg.training.train_batch_size))
                )
                # print("Train_losses shape: ", le
                # print("Train_losses: ", train_losess)
                train_losses_tensor = torch.cat(train_losses)
                train_loss = torch.mean(train_losses_tensor)
                # We keep track of the loss at each epoch
                if cfg.tracking:
                    total_loss += loss.detach().float()
                loss = loss / cfg.training.gradient_accumulation_steps
                accelerator.backward(loss)

                if step % cfg.training.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                #             if isinstance(checkpointing_steps, int):
                # New Code #
                if step % cfg.training.eval_every == 0:
                    perplexity, eval_loss = evaluate(cfg, model, eval_dataloader, accelerator, device, tokenizer)
                    logger.info(
                        f"Epoch {epoch}, Step {step} : perplexity: {perplexity} train_loss: {train_loss} eval_loss:{eval_loss}")

                    model.train()

                    if step % 125 == 0:
                        epoch_dir = f"epoch_{epoch}_most_recent"
                        if cfg.output_dir is not None:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)

                            output_dir = os.path.join(cfg.output_dir, epoch_dir)

                            # New Code #

                            unwrapped_model.save_pretrained(
                                output_dir,
                                is_main_process=accelerator.is_main_process,
                                save_function=accelerator.save,
                                state_dict=accelerator.get_state_dict(model),
                            )
                            if accelerator.is_main_process:
                                tokenizer.save_pretrained(output_dir)

                    # New Code #
                    # Tracks the best checkpoint and best metric
                    if (best_metric is None or best_metric > eval_loss):
                        best_metric = eval_loss
                        best_metric_checkpoint = os.path.join(cfg.output_dir, str(epoch))
                        accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")
                        accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")
                        logger.info(f"Saving model with best metric: Eval loss {best_metric}...")

                        epoch_dir = "model_with_best_eval"
                        if cfg.output_dir is not None:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)

                            output_dir = os.path.join(cfg.output_dir, epoch_dir)

                            # New Code #
                            # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
                            # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
                            # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
                            # For Zero Stages 1 and 2, models are saved as usual in the output directory.
                            # The model name saved is `pytorch_model.bin`
                            unwrapped_model.save_pretrained(
                                output_dir,
                                is_main_process=accelerator.is_main_process,
                                save_function=accelerator.save,
                                state_dict=accelerator.get_state_dict(model),
                            )
                            if accelerator.is_main_process:
                                tokenizer.save_pretrained(output_dir)

                if completed_steps >= cfg.training.max_train_steps:
                    break

                if step > len(train_dataloader):
                    break

            perplexity, eval_loss = evaluate(cfg, model, eval_dataloader, accelerator, device, tokenizer)
            model.train()
            logger.info(f"epoch {epoch}: perplexity: {perplexity} train_loss: {train_loss} eval_loss: {eval_loss}")

            checkpoint_model(cfg.output_dir, epoch, model, epoch, completed_steps)
            files = [os.path.join(os.path.abspath("train_data"), each) for each in
                     os.listdir(os.path.abspath("train_data")) if each.endswith(".txt")]
            # New Code #
    # Loads the best checkpoint after the training is finished
    if cfg.load_best_model:
        _, last_global_step = load_training_checkpoint(
            model,
            "/".join(best_metric_checkpoint.split("/")[:-1]),
            tag=best_metric_checkpoint.split("/")[-1],
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )

        # New Code #
        # Evaluates using the best checkpoint
        perplexity, eval_loss = evaluate(cfg, model, eval_dataloader, accelerator, tokenizer)
        logger.info(f"Best model metrics: perplexity: {perplexity}  eval_loss: {eval_loss}")
    #     if eval_loss != best_metric:
    #         raise AssertionError(
    #             f"Best metric {best_metric} does not match the metric {eval_loss} of the loaded best model."
    #         )

    print('Saving the model using the best weights checkpoint in the current output directory')
    if cfg.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        # New Code #
        # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
        # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
        # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
        # For Zero Stages 1 and 2, models are saved as usual in the output directory.
        # The model name saved is `pytorch_model.bin`
        unwrapped_model.save_pretrained(
            cfg.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(cfg.output_dir)


#     print('Started Pushing the Model and Tokenizer to Hugging Face Hub')

#     print('Pushing Model weights and other related files to Hugging Face Hub')
#     model.push_to_hub(cfg.output_dir)
#     print('Pushing the Tokenizer and related files to Hugging Face Hub')
#     tokenizer.push_to_hub(cfg.output_dir)

if __name__ == "__main__":
    # print(os.path.abspath('train_data'))
    load_raw_datasets()
    # main()

