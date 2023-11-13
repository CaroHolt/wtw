#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import json
import random
import torch
from dataclasses import dataclass, field
from itertools import chain
from collections.abc import Mapping
from typing import Optional, Union, Any, List
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from scipy.special import softmax

# Import time module
import time


import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler

import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
    default_data_collator,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    ShardSampler,
    nested_truncate,
    nested_concat,
    nested_numpify,
    nested_xla_mesh_reduce,
    distributed_concat,
    nested_detach,
    LabelSmoother
)
from transformers.adapters import AdapterArguments, AdapterTrainer, setup_adapter_training, AdapterConfig
from transformers.trainer_utils import get_last_checkpoint, has_length
from transformers.utils import check_min_version, find_labels, can_return_loss
from transformers.utils.versions import require_version
import torch.distributions as distributions


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # Additional Arguments
    adapter_dir: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Adapters that should be used for evaluation"
        },
    )
    adapter_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of adapter"
        },
    )
    combination_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help": "How should the adapters be combined? Possible options are: 'average' or 'ensemble'"
        },
    )
    adapter_weighting: Optional[str] = field(
        default="uniform",
        metadata={
            "help": "How should the adapters be weighted? Possible options are: 'uniform' or 'sent_sim', 'tfidf', 'prior', 'entropy'."
                    "If 'tfidf' or 'sent_sim' is chosen, you additionally need to provide the adapterval file paths."
        },
    )
    adapter_val_files: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of validation files of the adapters that should be merged. Needs to be provided to perform sent_sim and tfidf"
        },
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "name of the file to write evaluation results to"
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "Add up to k adapters to the weighting"
        },
    )
    topk_uniform: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Choose if topk selected adapters should be weighted equally"
        },
    )
    cumulative_gain: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Add up to k adapters to the weighting"
        },
    )
    evaluate_efficiency: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set to true if you want to print out timings for the different calculation steps. These will be stored in a file called log_timings.txt"
        },
    )

    # Usual arguments
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    weight_vector = []

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()



    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file

        raw_datasets = load_dataset(
            "text",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with training_args.main_process_first(desc="grouping texts together"):
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))


    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    if training_args.do_eval:
        eval_batch_size = training_args.per_device_eval_batch_size * max(1, training_args.n_gpu)

        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

        def num_examples(dataloader: DataLoader) -> int:
            """
            Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
            dataloader.dataset does not exist or has no length, estimates as best it can
            """
            try:
                dataset = dataloader.dataset
                if isinstance(dataset, IterableDatasetShard):
                    return len(dataloader.dataset.dataset)
                return len(dataloader.dataset)
            except (NameError, AttributeError, TypeError):  
                return len(dataloader) * training_args.per_device_train_batch_size

        def get_uncertainty_by_logits(logits: torch.LongTensor) -> torch.Tensor:

            by_word_uncertainty = distributions.Categorical(logits=logits).entropy() 
            
            if torch.isnan(by_word_uncertainty).sum() > 0:
                print("Nan entropy!")
                raise ValueError

            return by_word_uncertainty

        def _remove_unused_columns(dataset: "datasets.Dataset", description: Optional[str] = None):
            import inspect
            from packaging import version
            if not training_args.remove_unused_columns:
                return dataset
            signature = inspect.signature(model.forward)
            _signature_columns = list(signature.parameters.keys())
            label_names = find_labels(model.__class__)
            _signature_columns += list(set(["label", "label_ids"] + label_names))
            signature_columns = _signature_columns

            ignored_columns = list(set(dataset.column_names) - set(signature_columns))
            if len(ignored_columns) > 0:
                dset_description = "" if description is None else f"in the {description} set"
                logger.info(
                    f"The following columns {dset_description} don't have a corresponding argument in "
                    f"`{model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                    f" If {', '.join(ignored_columns)} are not expected by `{model.__class__.__name__}.forward`, "
                    " you can safely ignore this message."
                )

            columns = [k for k in signature_columns if k in dataset.column_names]

            if version.parse(datasets.__version__) < version.parse("1.4.0"):
                dataset.set_format(
                    type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
                )
                return dataset
            else:
                return dataset.remove_columns(ignored_columns)


        def get_dataloader(eval_dataset):
            eval_batch_size = training_args.per_device_eval_batch_size * max(1, training_args.n_gpu)


            eval_dataset = _remove_unused_columns(eval_dataset, description="evaluation")


            # Get the right dataloader
            if isinstance(eval_dataset, torch.utils.data.IterableDataset):
                if training_args.world_size > 1:
                    eval_dataset = IterableDatasetShard(
                        eval_dataset,
                        batch_size=training_args.per_device_eval_batch_size,
                        drop_last=training_args.dataloader_drop_last,
                        num_processes=training_args.world_size,
                        process_index=training_args.process_index,
                    )
                eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=eval_batch_size,
                    collate_fn=data_collator,
                    num_workers=training_args.dataloader_num_workers,
                    pin_memory=training_args.dataloader_pin_memory,
                )
            else:
                eval_sampler = SequentialSampler(eval_dataset) if training_args.world_size <= 1 else ShardSampler(
                    eval_dataset)

                eval_dataloader = DataLoader(
                    eval_dataset,
                    sampler=eval_sampler,
                    batch_size=eval_batch_size,
                    collate_fn=data_collator,
                    drop_last=training_args.dataloader_drop_last,
                    num_workers=training_args.dataloader_num_workers,
                    pin_memory=training_args.dataloader_pin_memory,
                )

            return eval_dataloader

        def get_current_ppl(eval_dataset, weight_vector):

            if data_args.combination_strategy == 'average' or data_args.combination_strategy == 'average+ensemble':
                torch.cuda.empty_cache()

                state_dicts_current = []
                for i in range(len(data_args.adapter_dir)):

                    eval_model = AutoModelForMaskedLM.from_pretrained(
                        model_args.model_name_or_path,
                        from_tf=bool(".ckpt" in model_args.model_name_or_path),
                        config=config,
                        cache_dir=model_args.cache_dir,
                        revision=model_args.model_revision,
                        use_auth_token=True if model_args.use_auth_token else None,
                    )

                    eval_model.load_adapter(
                        data_args.adapter_dir[i],
                        load_as="adapter",
                    )
                    eval_model.set_active_adapters("adapter")
                    state_dicts_current.append(eval_model.state_dict())


                    # Deactivate all adapters
                    eval_model.set_active_adapters(None)
                    # Delete the added adapter
                    eval_model.delete_adapter("adapter")




                logger.info("*** Adapters are weighted according to the weight vector ***")
                for key in state_dicts_current[0]:
                    sum_state_dicts_of_key = 0
                    if "adapter" in key or 'cls' in key:
                        for i in range(len(state_dicts_current)):
                            sum_state_dicts_of_key += (state_dicts_current[i][key] * weight_vector[i])
                        state_dicts_current[0][key] = sum_state_dicts_of_key




                torch.cuda.empty_cache()

                eval_model = AutoModelForMaskedLM.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
                eval_model.load_adapter(
                    data_args.adapter_dir[0],
                    load_as="adapter",
                    config=AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=12,
                                         non_linearity="relu", ln_before=True, ln_after=False)
                )
                eval_model.set_active_adapters("adapter")
                eval_model.load_state_dict(state_dicts_current[0])

                trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
                eval_trainer = trainer_class(
                    model=eval_model,
                    args=training_args,
                    train_dataset=train_dataset if training_args.do_train else None,
                    eval_dataset=eval_dataset if training_args.do_eval else None,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
                    preprocess_logits_for_metrics=preprocess_logits_for_metrics
                    if training_args.do_eval and not is_torch_tpu_available()
                    else None,
                )

                metrics = eval_trainer.evaluate()

                try:
                    ppl = math.exp(metrics["eval_loss"])
                except OverflowError:
                    ppl = float("inf")

                # Deactivate all adapters
                eval_model.set_active_adapters(None)
                # Delete the added adapter
                eval_model.delete_adapter("adapter")


            if data_args.combination_strategy == 'ensemble' or data_args.combination_strategy == 'average+ensemble':

                if data_args.combination_strategy == 'average+ensemble':
                    torch.cuda.empty_cache()
                    eval_model = AutoModelForMaskedLM.from_pretrained(
                        model_args.model_name_or_path,
                        from_tf=bool(".ckpt" in model_args.model_name_or_path),
                        config=config,
                        cache_dir=model_args.cache_dir,
                        revision=model_args.model_revision,
                        use_auth_token=True if model_args.use_auth_token else None,
                    )




                eval_dataloader = get_dataloader(eval_dataset)

                # Initialize containers
                # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
                losses_host = None

                # losses/preds/labels on CPU (final containers)
                all_losses = None
                for step, inputs in enumerate(eval_dataloader):
                    inputs = _prepare_input(inputs)

                    # Get labels if task does not have labels
                    default_label_names = find_labels(model.__class__)
                    label_names = default_label_names if training_args.label_names is None else training_args.label_names

                    has_labels = False if len(label_names) == 0 else all(inputs.get(k) is not None for k in label_names)

                    return_loss = inputs.get("return_loss", None)
                    if return_loss is None:
                        return_loss = can_return_loss(model.__class__)
                    loss_without_labels = True if len(label_names) == 0 and return_loss else False

                    if has_labels or loss_without_labels:
                        labels = nested_detach(tuple(inputs.get(name) for name in label_names))
                        if len(labels) == 1:
                            labels = labels[0]
                    else:
                        labels = None

                    preds = None

                    for i in range(len(data_args.adapter_dir)):
                        if weight_vector[i] == 0:
                            continue
                        adapter_name = "adapter" + str(i)
                        eval_model = AutoModelForMaskedLM.from_pretrained(
                            model_args.model_name_or_path,
                            from_tf=bool(".ckpt" in model_args.model_name_or_path),
                            config=config,
                            cache_dir=model_args.cache_dir,
                            revision=model_args.model_revision,
                            use_auth_token=True if model_args.use_auth_token else None,
                        )

                        eval_model.load_adapter(
                            data_args.adapter_dir[i],
                            load_as=adapter_name
                        )
                        eval_model.set_active_adapters(adapter_name)

                        eval_model = eval_model.to(device=training_args.device)

                        with torch.no_grad():
                            outputs = eval_model(**inputs)
                            logits = outputs.logits

                            if data_args.adapter_weighting == 'uniform' and len(weight_vector) == 0:
                                logger.info("*** All adapters are equally weighted ***")
                                preds = logits if preds is None else torch.add(preds, logits)
                            else:
                                logger.info("*** Adapters are weighted according to the weight vector ***")
                                weight = i
                                logits = logits * weight_vector[weight]
                                preds = logits if preds is None else torch.add(preds, logits)

                        # Deactivate all adapters
                        model.set_active_adapters(None)
                        # Delete the added adapter
                        model.delete_adapter(adapter_name)


                    loss_fct = torch.nn.CrossEntropyLoss()
                    masked_lm_loss = loss_fct(preds.view(-1, config.vocab_size), labels.view(-1))
                    loss = masked_lm_loss.mean().detach()

                    # Update containers on host
                    if loss is not None:
                        losses = _nested_gather(loss.repeat(eval_batch_size))
                        losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)

                    # Gather all remaining tensors and put them back on the CPU
                    if losses_host is not None:
                        losses = nested_numpify(losses_host)
                        all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)

                avg_loss = np.mean(all_losses)

                # calculating perplexity
                ppl = math.exp(avg_loss)



            return ppl




        def _prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
            """
            Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
            """
            if isinstance(data, Mapping):
                return type(data)({k: _prepare_input(v) for k, v in data.items()})
            elif isinstance(data, (tuple, list)):
                return type(data)(_prepare_input(v) for v in data)
            elif isinstance(data, torch.Tensor):
                kwargs = {"device": training_args.device}

                return data.to(**kwargs)
            return data


        def _nested_gather(tensors, name=None):
            """
            Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
            concatenating them to `gathered`
            """
            if tensors is None:
                return
            if is_torch_tpu_available():
                if name is None:
                    name = "nested_gather"
                tensors = nested_xla_mesh_reduce(tensors, name)
            elif training_args.local_rank != -1:
                tensors = distributed_concat(tensors)
            return tensors

        def get_uncertainty_for_model(model, eval_dataset, eval_dataloader):
            model = model.to(device=training_args.device)

            model.eval()


            # Initialize containers
            all_uncertainties = None

            observed_num_examples = 0

            for step, inputs in enumerate(eval_dataloader):

                inputs = _prepare_input(inputs)

                with torch.no_grad():
                    outputs = model(**inputs)

                    by_word_uncertainty = get_uncertainty_by_logits(outputs.logits)

                # Update containers on host
                if by_word_uncertainty is not None:
                    all_uncertainties = by_word_uncertainty if all_uncertainties is None else nested_concat(
                        all_uncertainties, by_word_uncertainty,
                        padding_index=-100)

            # Number of samples
            if has_length(eval_dataset):
                num_samples = len(eval_dataset)
            # The instance check is weird and does not actually check for the type, but whether the dataset has the right
            # methods. Therefore we need to make sure it also has the attribute.
            elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples",
                                                                            0) > 0:
                num_samples = eval_dataset.num_examples
            else:
                if has_length(eval_dataloader):
                    num_samples = num_examples(eval_dataloader)
                else:  # both len(dataloader.dataset) and len(dataloader) fail
                    num_samples = observed_num_examples
            if num_samples == 0 and observed_num_examples > 0:
                num_samples = observed_num_examples

            # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
            # samplers has been rounded to a multiple of batch_size, so we truncate.
            if all_uncertainties is not None:
                all_uncertainties = nested_truncate(all_uncertainties, num_samples)

            mean_uncertainty = torch.mean(all_uncertainties).item()

            return mean_uncertainty


        def get_uncertainty_list(model, chosen_adapter_ids = None):
            adapter_uncertainties = []
            eval_dataloader = get_dataloader(eval_dataset)
            if chosen_adapter_ids:
                if data_args.combination_strategy == 'ensemble':
                    # Average output vectors and calculate entropy
                    for i in range(len(data_args.adapter_dir)):
                        if i in chosen_adapter_ids:
                            adapter_uncertainties.append(100)
                            continue
                        else:
                            ids_to_eval = chosen_adapter_ids.copy()
                            ids_to_eval.append(i)

                            all_uncertainties = None
                            for step, inputs in enumerate(eval_dataloader):
                                inputs = _prepare_input(inputs)

                                ignore_keys = None

                                if ignore_keys is None:
                                    if hasattr(model, "config"):
                                        ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
                                    else:
                                        ignore_keys = []

                                # Get labels if task does not have labels
                                default_label_names = find_labels(model.__class__)
                                label_names = default_label_names if training_args.label_names is None else training_args.label_names

                                has_labels = False if len(label_names) == 0 else all(
                                    inputs.get(k) is not None for k in label_names)

                                return_loss = inputs.get("return_loss", None)
                                if return_loss is None:
                                    return_loss = can_return_loss(model.__class__)
                                loss_without_labels = True if len(label_names) == 0 and return_loss else False

                                if has_labels or loss_without_labels:
                                    labels = nested_detach(tuple(inputs.get(name) for name in label_names))
                                    if len(labels) == 1:
                                        labels = labels[0]
                                else:
                                    labels = None

                                preds = None

                                for id in ids_to_eval:

                                    model = AutoModelForMaskedLM.from_pretrained(
                                        model_args.model_name_or_path,
                                        from_tf=bool(".ckpt" in model_args.model_name_or_path),
                                        config=config,
                                        cache_dir=model_args.cache_dir,
                                        revision=model_args.model_revision,
                                        use_auth_token=True if model_args.use_auth_token else None,
                                    )

                                    model.load_adapter(
                                        data_args.adapter_dir[id],
                                        load_as="adapter"
                                    )
                                    model.set_active_adapters("adapter")

                                    model = model.to(device=training_args.device)

                                    with torch.no_grad():
                                        outputs = model(**inputs)
                                        logits = outputs.logits

                                        logger.info("*** All adapters are equally weighted ***")
                                        preds = logits if preds is None else torch.add(preds, logits)

                                preds = preds / len(ids_to_eval)


                                by_word_uncertainty = get_uncertainty_by_logits(preds)

                                all_uncertainties = by_word_uncertainty if all_uncertainties is None else nested_concat(
                                    all_uncertainties, by_word_uncertainty,
                                    padding_index=-100)

                            # Number of samples
                            if has_length(eval_dataset):
                                num_samples = len(eval_dataset)
                            # The instance check is weird and does not actually check for the type, but whether the dataset has the right
                            # methods. Therefore we need to make sure it also has the attribute.
                            elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset,
                                                                                            "num_examples",
                                                                                            0) > 0:
                                num_samples = eval_dataset.num_examples
                            else:
                                if has_length(eval_dataloader):
                                    num_samples = num_examples(eval_dataloader)

                            # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
                            # samplers has been rounded to a multiple of batch_size, so we truncate.
                            if all_uncertainties is not None:
                                all_uncertainties = nested_truncate(all_uncertainties, num_samples)

                            mean_uncertainty = torch.mean(all_uncertainties).item()

                            adapter_uncertainties.append(mean_uncertainty)


                elif  data_args.combination_strategy == 'average' or data_args.combination_strategy == 'average+ensemble':
                    state_dicts_topk = []
                    for id in chosen_adapter_ids:
                        model = AutoModelForMaskedLM.from_pretrained(
                            model_args.model_name_or_path,
                            from_tf=bool(".ckpt" in model_args.model_name_or_path),
                            config=config,
                            cache_dir=model_args.cache_dir,
                            revision=model_args.model_revision,
                            use_auth_token=True if model_args.use_auth_token else None,
                        )

                        model.load_adapter(
                            data_args.adapter_dir[id],
                            load_as="adapter"
                        )
                        model.set_active_adapters("adapter")

                        state_dicts_topk.append(model.state_dict())

                        # Deactivate all adapters
                        model.set_active_adapters(None)
                        # Delete the added adapter
                        model.delete_adapter("adapter")

                    for i in range(len(data_args.adapter_dir)):
                        total_state_dicts = state_dicts_topk
                        if chosen_adapter_ids:
                            if i in chosen_adapter_ids:
                                adapter_uncertainties.append(100)
                                continue
                        model = AutoModelForMaskedLM.from_pretrained(
                            model_args.model_name_or_path,
                            from_tf=bool(".ckpt" in model_args.model_name_or_path),
                            config=config,
                            cache_dir=model_args.cache_dir,
                            revision=model_args.model_revision,
                            use_auth_token=True if model_args.use_auth_token else None,
                        )

                        model.load_adapter(
                            data_args.adapter_dir[i],
                            load_as="adapter"
                        )
                        model.set_active_adapters("adapter")

                        total_state_dicts.append(model.state_dict())

                        # Deactivate all adapters
                        model.set_active_adapters(None)
                        # Delete the added adapter
                        model.delete_adapter("adapter")


                        for key in total_state_dicts[0]:
                            sum_state_dicts_of_key = 0
                            if "adapter" in key or 'cls' in key:
                                for i in range(len(total_state_dicts)):
                                    sum_state_dicts_of_key += total_state_dicts[i][key].to('cpu')
                                total_state_dicts[0][key] = sum_state_dicts_of_key / len(total_state_dicts)

                        torch.cuda.empty_cache()

                        model = AutoModelForMaskedLM.from_pretrained(
                            model_args.model_name_or_path,
                            from_tf=bool(".ckpt" in model_args.model_name_or_path),
                            config=config,
                            cache_dir=model_args.cache_dir,
                            revision=model_args.model_revision,
                            use_auth_token=True if model_args.use_auth_token else None,
                        )
                        model.load_adapter(
                            data_args.adapter_dir[0],
                            load_as="adapter",
                            config=AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=12,
                                                 non_linearity="relu", ln_before=True, ln_after=False)
                        )
                        model.set_active_adapters("adapter")
                        model.load_state_dict(total_state_dicts[0])

                        mean_uncertainty = get_uncertainty_for_model(model, eval_dataset, eval_dataloader)
                        adapter_uncertainties.append(mean_uncertainty)

                        # Deactivate all adapters
                        model.set_active_adapters(None)
                        # Delete the added adapter
                        model.delete_adapter("adapter")


            if not chosen_adapter_ids:
                for i in range(len(data_args.adapter_dir)):
                    adapter_name = "adapter" + str(i)
                    model.load_adapter(
                        data_args.adapter_dir[i],
                        load_as=adapter_name
                    )
                    model.set_active_adapters(adapter_name)

                    mean_uncertainty = get_uncertainty_for_model(model, eval_dataset, eval_dataloader)
                    adapter_uncertainties.append(mean_uncertainty)


                    # Deactivate all adapters
                    model.set_active_adapters(None)
                    # Delete the added adapter
                    model.delete_adapter(adapter_name)

            return adapter_uncertainties

    def get_priors(model, chosen_adapter_ids=None):

        # The following lines: 1214 - 1731 are partially adapted from the following original sources: 
        # https://github.com/kernelmachine/cbtm/blob/d4c169f0dd3d8d4c5fc223eb8558ef4b9ab0705d/metaseq/sequence_scorer_btm.py
        # and
        # https://github.com/kernelmachine/cbtm/blob/d4c169f0dd3d8d4c5fc223eb8558ef4b9ab0705d/metaseq/sequence_scorer.py            # 


        # Copyright 2023 Suchin Gururangan et al. All rights reserved.

        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at

        #     http://www.apache.org/licenses/LICENSE-2.0

        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.

        logger.info("*** We estimate the adapter weights using prior estimation ***")

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        def batch_for_softmax(dec_out, target):
            first = dec_out.logits
            bsz, tsz, dim = first.shape
            if bsz * tsz < sys.maxsize:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + sys.maxsize
                    yield (flat[:, s:e],), flat_tgt[:, s:e], False
                    s = e

        eval_dataset = tokenized_datasets["validation"]

        eval_dataset = _remove_unused_columns(eval_dataset, description="evaluation")
        prior_sampler = SequentialSampler(eval_dataset)

        data_collator = default_data_collator

        prior_dataloader = DataLoader(
            eval_dataset,
            sampler=prior_sampler,
            batch_size=1,
            collate_fn=data_collator,
            drop_last=training_args.dataloader_drop_last,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )

        expert_probs_all = []
        priors_all = []
        pbar = tqdm(enumerate(prior_dataloader))
        for step, sample in pbar:

            sample = _prepare_input(sample)

            # compute scores for each model in the ensemble
            model_probs = []


            for i in range(len(data_args.adapter_dir)):
                adapter_name = "adapter" + str(i)
                model.load_adapter(
                    data_args.adapter_dir[i],
                    load_as=adapter_name
                )
                model.set_active_adapters(adapter_name)

                model = model.to(device=training_args.device)

                model.eval()

                with torch.no_grad():
                    out = model(**sample)

                orig_target = sample['input_ids']

                batched = batch_for_softmax(out, orig_target)

                probs, idx = None, 0
                for bd, tgt, is_single in batched:
                    sample["input_ids"] = tgt
                    curr_prob = torch.nn.functional.softmax(bd.logits, dim=-1).data
                    if is_single:
                        probs = gather_target_probs(curr_prob, orig_target)
                    else:
                        if probs is None:
                            probs = curr_prob.new(orig_target.numel())
                        step = curr_prob.size(0) * curr_prob.size(1)
                        end = step + idx
                        tgt_probs = gather_target_probs(
                            curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt
                        )
                        probs[idx:end] = tgt_probs.view(-1)
                        idx = end
                    sample["input_ids"] = orig_target

                probs = probs.view(sample["input_ids"].shape)

                model_probs.append(probs.unsqueeze(0))

            model_probs = torch.cat(model_probs, dim=0)

            len_models = len(data_args.adapter_dir)

            priors = [1 / len_models] * len_models

            weights = model_probs[:, :, :-1].clone()

            # calculate normalization
            denom = weights.clone()
            for ix, prior in enumerate(priors):
                denom[ix, :].mul_(prior)

            denom = denom.sum(0)

            # calculate posterior
            for ix, prior in enumerate(priors):
                weights[ix, :].mul_(prior).div_(denom)

            beginning_weights = torch.tensor(priors).float().repeat(model_probs.shape[1], 1).t().unsqueeze(2).to(
                weights)

            weights = torch.cat([beginning_weights, weights], -1)

            expert_probs = weights[:, :, -1]
            skipper_counter = 0
            if torch.isnan(expert_probs).any():
                skipper_counter += 1
                continue

            expert_probs = expert_probs.mean(1).unsqueeze(0).cpu().numpy()

            expert_probs_all.append(expert_probs)
            
            prior = pd.DataFrame(np.concatenate(expert_probs_all, 0)).ewm(alpha=0.3, adjust=False).mean().tail(
                n=1).to_numpy().squeeze(0)
            priors_all.append(prior)

            if step >= 99:
                break

        expert_probs_list = [x.tolist()[0] for x in expert_probs_all]

        last_prior = priors_all[-1]

        weight_vector = last_prior

        if data_args.top_k:
            # Enumerate the list elements along with their indices
            enum_weights = list(enumerate(last_prior))

            # Sort the list of tuples by the second element (the weights) in descending order
            sorted_weights = sorted(enum_weights, key=lambda x: x[1], reverse=True)[:data_args.top_k]

            # Create a new list with the same length as `new_weights` and set all elements to 0
            new_prior = [0] * len(last_prior)

            # Set the topk elements in the new list to their corresponding values from the sorted list
            for index, weight in sorted_weights:
                new_prior[index] = weight

            # Normalize the new_weights list
            if data_args.topk_uniform:
                new_prior = [1 if a_ > 0 else 0 for a_ in new_prior]
                new_prior = np.array(new_prior) / sum(new_prior)
                weight_vector = new_prior
            else:
                new_prior = np.array(new_prior) / sum(new_prior)
                weight_vector = new_prior

            adapters_to_use = []
            for i in range(len(weight_vector)):
                if weight_vector[i] != 0:
                    adapters_to_use.append(data_args.adapter_dir[i])



        return weight_vector

    # Calculate the weights of adapters using different strategies
    if data_args.adapter_dir and len(data_args.adapter_dir) > 1 and data_args.adapter_weighting != 'uniform':
        if data_args.adapter_weighting == 'sent_sim':
            # record start time
            start_sent_sim = time.time()
            from sentence_transformers import SentenceTransformer
            from numpy.linalg import norm
            sent_model = SentenceTransformer('all-mpnet-base-v2')

            logger.info(
                "*** We estimate the adapter weights using sentence similarity of the average cosine sim  on 100 sequences of the eval set ***")

            # Load test sequences
            with open(data_args.validation_file, 'r') as f:
                test_data = json.load(f)

            sequences_test_100 = random.choices(test_data, k=100)
            embeddings_test = sent_model.encode(sequences_test_100)

            cos_sims = []
            # Load train sentences
            for file in data_args.adapter_val_files:
                with open(file, 'r') as f:
                    train_data = json.load(f)

                sequences_train_100 = random.choices(train_data, k=100)
                embeddings_train = sent_model.encode(sequences_train_100)

                all_cosine = 0
                counter = 0
                for a in embeddings_train:
                    for b in embeddings_test:
                        cosine = np.dot(a, b) / (norm(a) * norm(b))
                        all_cosine = all_cosine + cosine
                        counter += 1
                avg_cos = all_cosine / counter
                if avg_cos < 0:
                    avg_cos = 0
                cos_sims.append(avg_cos)


            if data_args.top_k:
                adapters_to_use = []
                sorted_cos = sorted(cos_sims, reverse=True)
                # thres = 0.05
                # if sorted_cos[data_args.top_k - 1] > 0.05:
                #    thres = sorted_cos[data_args.top_k - 1]
                thres = sorted_cos[data_args.top_k - 1]

                # Count the occurrences of the threshold (5th largest number)
                count_threshold = sorted_cos.count(thres)

                if count_threshold > 1:
                    temp = []
                    count = 0
                    for a_ in cos_sims:
                        if a_ > thres:
                            if data_args.topk_uniform:
                                temp.append(1)
                            else:
                                temp.append(a_)
                        elif a_ == thres and count == 0:
                            if data_args.topk_uniform:
                                temp.append(1)
                            else:
                                temp.append(a_)
                            count += 1
                        else:
                            temp.append(0)
                    cos_sims = temp
                else:
                    if data_args.topk_uniform:
                        cos_sims = [0 if a_ < thres else 1 for a_ in cos_sims]
                    else:
                        cos_sims = [0 if a_ < thres else a_ for a_ in cos_sims]

                for i in range(len(cos_sims)):
                    if cos_sims[i] != 0:
                        adapters_to_use.append(data_args.adapter_dir[i])
                for c in cos_sims:
                    weight_vector.append(c / sum(cos_sims))
            else:
                for c in cos_sims:
                    weight_vector.append(c / sum(cos_sims))

            end_sent_sim = time.time()

            sent_sim_duration = end_sent_sim - start_sent_sim

            if data_args.evaluate_efficiency:
                with open("log_timings.txt", 'a+') as timing_file:
                    timing_file.write("Duration sent_calc: ")
                    timing_file.write(str(sent_sim_duration))
                    timing_file.write("\n")


        if data_args.adapter_weighting == 'tfidf':
            start_tfidf = time.time()
            if data_args.adapter_val_files is None:
                raise ValueError("To compute the tf-idf we need the name of the validation files of the adapters")
            import nltk
            from sklearn.feature_extraction.text import TfidfVectorizer
            import string
            nltk.download('punkt')
            stemmer = nltk.stem.porter.PorterStemmer()
            remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

            def stem_tokens(tokens):
                return [stemmer.stem(item) for item in tokens]

            '''remove punctuation, lowercase, stem'''

            def normalize(text):
                return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

            def cosine_sim(text1, text2):
                tfidf = vectorizer.fit_transform([text1, text2])
                return ((tfidf * tfidf.T).A)[0, 1]

            logger.info(
                "*** We estimate the adapter weights using tf-idf and cosine sim on 100 sequences of the eval set ***")

            # Load test sequences
            with open(data_args.validation_file, 'r') as f:
                test_data = json.load(f)

            sequences_test_100 = random.choices(test_data, k=100)
            sequences_test_100_string = ''.join(sequences_test_100)

            cos_sims = []
            # Load train sentences
            for file in data_args.adapter_val_files:
                with open(file, 'r') as f:
                    train_data = json.load(f)

                sequences_train_100 = random.choices(train_data, k=100)
                sequences_train_100_string = ''.join(sequences_train_100)

                vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

                cos_sim = cosine_sim(sequences_train_100_string, sequences_test_100_string)
                cos_sims.append(cos_sim)

            if data_args.top_k:
                adapters_to_use = []
                sorted_cos = sorted(cos_sims, reverse=True)
                thres = sorted_cos[data_args.top_k - 1]

                count_threshold = sorted_cos.count(thres)

                if count_threshold > 1:
                    temp = []
                    count = 0
                    for a_ in cos_sims:
                        if a_ > thres:
                            if data_args.topk_uniform:
                                temp.append(1)
                            else:
                                temp.append(a_)
                        elif a_ == thres and count == 0:
                            if data_args.topk_uniform:
                                temp.append(1)
                            else:
                                temp.append(a_)
                            count += 1
                        else:
                            temp.append(0)
                    cos_sims = temp
                else:
                    if data_args.topk_uniform:
                        cos_sims = [0 if a_ < thres else 1 for a_ in cos_sims]
                    else:
                        cos_sims = [0 if a_ < thres else a_ for a_ in cos_sims]

                for i in range(len(cos_sims)):
                    if cos_sims[i] != 0:
                        adapters_to_use.append(data_args.adapter_dir[i])
                for c in cos_sims:
                    weight_vector.append(c / sum(cos_sims))
            else:
                for c in cos_sims:
                    weight_vector.append(c / sum(cos_sims))

            end_tfidf = time.time()

            tfidf_duration = end_tfidf - start_tfidf

            if data_args.evaluate_efficiency:
                with open("log_timings.txt", 'a+') as timing_file:
                    timing_file.write("Duration tfidf_calc: ")
                    timing_file.write(str(tfidf_duration))
                    timing_file.write("\n")



        if data_args.adapter_weighting == 'entropy':
            start_entropy = time.time()
            logger.info("*** We estimate the adapter weights using entropy minimization ***")

            eval_dataloader = get_dataloader(eval_dataset)


            if data_args.cumulative_gain:
                chosen_adapter_ids = []
                for k in range(1, data_args.top_k + 1):
                    adapter_uncertainties = get_uncertainty_list(model, chosen_adapter_ids)
                    thres = min(adapter_uncertainties)
                    old_uncerts = [0 if a_ > thres else a_ for a_ in adapter_uncertainties]
                    for i in range(len(old_uncerts)):
                        if old_uncerts[i] != 0:
                            chosen_adapter_ids.append(i)
                            break


                    for pos in range(len(adapter_uncertainties)):
                        if pos in chosen_adapter_ids:
                            adapter_uncertainties[pos] = 1
                        else:
                            adapter_uncertainties[pos] = 0

                    current_weight_vector = []
                    for u in adapter_uncertainties:
                        current_weight_vector.append(u / sum(adapter_uncertainties))


                    ppl = get_current_ppl(eval_dataset, current_weight_vector)


                    for pos in range(len(adapter_uncertainties)):
                        if pos in chosen_adapter_ids:
                            adapter_uncertainties[pos] = 1
                        else:
                            adapter_uncertainties[pos] = 0

                adapters_to_use = []
                for i in range(len(adapter_uncertainties)):
                    if adapter_uncertainties[i] != 0:
                        k = i
                        adapters_to_use.append(data_args.adapter_dir[k])
                for u in adapter_uncertainties:
                    weight_vector.append(u / sum(adapter_uncertainties))




            else:
                print("Evaluate adapter uncertainties with model")
                adapter_uncertainties = get_uncertainty_list(model, None)

            

            torch.cuda.empty_cache()

            if data_args.top_k:
                if not data_args.cumulative_gain:
                    sorted_cos = sorted(adapter_uncertainties)
                    thres = sorted_cos[data_args.top_k - 1]
                    if data_args.topk_uniform:
                        adapter_uncertainties = [0 if a_ > thres else 1 for a_ in adapter_uncertainties]
                    else:
                        adapter_uncertainties = [0 if a_ > thres else a_ for a_ in adapter_uncertainties]
                    if data_args.top_k > 1:
                        adapter_uncertainties = [0 if a_ == 0 else sum(adapter_uncertainties) - a_ for a_ in
                                                 adapter_uncertainties]

                adapters_to_use = []
                for i in range(len(adapter_uncertainties)):
                    if adapter_uncertainties[i] != 0:
                        k = i
                        adapters_to_use.append(data_args.adapter_dir[k])
                for u in adapter_uncertainties:
                    weight_vector.append(u / sum(adapter_uncertainties))
            else:
                adapter_uncertainties = [0 if a_ == 0 else sum(adapter_uncertainties) - a_ for a_ in
                                         adapter_uncertainties]
                for u in adapter_uncertainties:
                    weight_vector.append(u / sum(adapter_uncertainties))

            end_entropy = time.time()

            entropy_duration = end_entropy - start_entropy

            if data_args.evaluate_efficiency:
                with open("log_timings.txt", 'a+') as timing_file:
                    timing_file.write("Duration entropy_calc: ")
                    timing_file.write(str(entropy_duration))
                    timing_file.write("\n")



        if data_args.adapter_weighting == 'prior':
            start_prior = time.time()

            if data_args.cumulative_gain:
                chosen_adapter_ids = []
                for k in range(1, data_args.top_k + 1):
                    weight_vector = get_priors(model, chosen_adapter_ids)
                    thres = max(weight_vector)
                    old_weights = [0 if a_ < thres else a_ for a_ in weight_vector]
                    for i in range(len(old_weights)):
                        if old_weights[i] != 0:
                            chosen_adapter_ids.append(i)
                    if len(chosen_adapter_ids) != k:
                        random_element = random.choice(chosen_adapter_ids)
                        chosen_adapter_ids.remove(random_element)

                model.set_active_adapters(None)
                # Delete the added adapter
                model.delete_adapter('chosen_top_1')

                for pos in range(len(weight_vector)):
                    if pos in chosen_adapter_ids:
                        weight_vector[pos] = 1
                    else:
                        weight_vector[pos] = 0
            else:
                weight_vector = get_priors(model)

            end_prior = time.time()

            prior_duration = end_prior - start_prior

            if data_args.evaluate_efficiency:
                with open("log_timings.txt", 'a+') as timing_file:
                    timing_file.write("Duration prior_calc: ")
                    timing_file.write(str(prior_duration))
                    timing_file.write("\n")



    # Adapter Combination Setup
    # Here We combine multiple adapters using different strategies
    # Either we only provide one adapter or we combine them weighting them uniformly or weight them according to different strategies
    # Also we either average the parameters of the adapters or we average the output vectors here
    # Setup adapters that have already been trained
    if data_args.adapter_dir and len(data_args.adapter_dir) == 1:
        logger.info("*** Only one adapter provided, no weighting or averaging necessary ***")

        model.load_adapter(
            data_args.adapter_dir[0],
            load_as="adapter",
            config=AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=12,
                                 non_linearity="relu", ln_before=True, ln_after=False)
        )
        model.set_active_adapters("adapter")

    # Weight space averaging of multiple adapters
    elif data_args.adapter_dir is not None and len(data_args.adapter_dir) > 1 and (data_args.combination_strategy == 'average' or data_args.combination_strategy == 'average+ensemble'):
        start_avg = time.time()
        logger.info("*** Applying weight space averaging of the adapters ***")
        state_dicts = []

        for i in range(len(data_args.adapter_dir)):

            model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )


            model.load_adapter(
                data_args.adapter_dir[i],
                load_as="adapter"
            )
            model.set_active_adapters("adapter")
            state_dicts.append(model.state_dict())

            # Deactivate all adapters
            model.set_active_adapters(None)
            # Delete the added adapter
            model.delete_adapter("adapter")

        # TODO: Test if this improves performance
        # get best adapter
        #best_adapter = weight_vector.index(max(weight_vector))


        if data_args.adapter_weighting == 'uniform' and len(weight_vector) == 0:
            logger.info("*** All adapters are equally weighted ***")
            for key in state_dicts[0]:
                sum_state_dicts_of_key = 0
                if "adapter" in key or 'cls' in key:
                #if "adapter" in key:
                    for i in range(len(state_dicts)):
                        sum_state_dicts_of_key += state_dicts[i][key]
                    state_dicts[0][key] = sum_state_dicts_of_key / len(state_dicts)

        else:
            logger.info("*** Adapters are weighted according to the weight vector ***")
            for key in state_dicts[0]:
                sum_state_dicts_of_key = 0
                if "adapter" in key or 'cls' in key:
                #if "adapter" in key:
                    for i in range(len(state_dicts)):
                        sum_state_dicts_of_key += (state_dicts[i][key] * weight_vector[i])
                        #state_dicts[best_adapter][key] = sum_state_dicts_of_key
                    state_dicts[0][key] = sum_state_dicts_of_key


        torch.cuda.empty_cache()


        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model.load_adapter(
            data_args.adapter_dir[0],
            load_as="adapter",
            config=AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=12,
                                 non_linearity="relu", ln_before=True, ln_after=False)
        )

        model.set_active_adapters("adapter")
        model.load_state_dict(state_dicts[0])



    elif training_args.do_train:
        # Add adapter that has not been trained yet
        adapter_args.adapter_config = AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=12,
                                                non_linearity="relu", ln_before=True, ln_after=False)
        setup_adapter_training(model, adapter_args, data_args.dataset_name or "mlm")

    # Initialize our Trainer
    trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if len(weight_vector) > 0:
            logger.info("*** Evaluate with weights***")
            logger.info(weight_vector)

        if data_args.validation_file:
            logger.info("*** Evaluate on file: " + str(data_args.validation_file) + " ***")

        if data_args.adapter_dir is not None and len(data_args.adapter_dir) > 1 and (data_args.combination_strategy == 'average' or data_args.combination_strategy == 'average+ensemble'):

            metrics = trainer.evaluate()

            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
                eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity


            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            end_avg = time.time()

            duration_avg = end_avg - start_avg

            if data_args.evaluate_efficiency:
                with open("log_timings.txt", 'a+') as timing_file:
                    timing_file.write("Duration averaging: ")
                    timing_file.write(str(duration_avg))
                    timing_file.write("\n")                    


            if not training_args.do_train:
                with open(data_args.eval_file, 'a+') as outfile:
                    weight_str = '-'.join(str(e) for e in weight_vector)
                    adapters_str = '-'.join(str(a) for a in adapters_to_use)
                    outfile.write(
                        f"{model_args.model_name_or_path},{'average'},{data_args.adapter_weighting},{weight_str},{adapters_str},{data_args.validation_file},{perplexity},{str(training_args.seed)},{str(data_args.top_k)}\n")



        if data_args.adapter_dir is not None and len(data_args.adapter_dir) > 1 and (data_args.combination_strategy == 'ensemble' or data_args.combination_strategy == 'average+ensemble'):

            if data_args.combination_strategy == 'average+ensemble':
                torch.cuda.empty_cache()
                model = AutoModelForMaskedLM.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )

            start_ens = time.time()

            eval_dataloader = get_dataloader(eval_dataset)

            # Initialize containers
            # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
            losses_host = None

            # losses/preds/labels on CPU (final containers)
            all_losses = None
            for step, inputs in enumerate(eval_dataloader):
                inputs = _prepare_input(inputs)

                ignore_keys = None

                if ignore_keys is None:
                    if hasattr(model, "config"):
                        ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
                    else:
                        ignore_keys = []

                # Get labels if task does not have labels
                default_label_names = find_labels(model.__class__)
                label_names = default_label_names if training_args.label_names is None else training_args.label_names

                has_labels = False if len(label_names) == 0 else all(inputs.get(k) is not None for k in label_names)

                return_loss = inputs.get("return_loss", None)
                if return_loss is None:
                    return_loss = can_return_loss(model.__class__)
                loss_without_labels = True if len(label_names) == 0 and return_loss else False

                if has_labels or loss_without_labels:
                    labels = nested_detach(tuple(inputs.get(name) for name in label_names))
                    if len(labels) == 1:
                        labels = labels[0]
                else:
                    labels = None

                preds = None



                for i in range(len(data_args.adapter_dir)):
                    if weight_vector[i] == 0:
                        continue
                    model = AutoModelForMaskedLM.from_pretrained(
                        model_args.model_name_or_path,
                        from_tf=bool(".ckpt" in model_args.model_name_or_path),
                        config=config,
                        cache_dir=model_args.cache_dir,
                        revision=model_args.model_revision,
                        use_auth_token=True if model_args.use_auth_token else None,
                    )

                    adapter_name = "adapter" + str(i)
                    model.load_adapter(
                        data_args.adapter_dir[i],
                        load_as=adapter_name
                    )
                    model.set_active_adapters(adapter_name)

                    model = model.to(device=training_args.device)

                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits


                        if data_args.adapter_weighting == 'uniform' and len(weight_vector) == 0:
                            logger.info("*** All adapters are equally weighted ***")
                            preds = logits if preds is None else torch.add(preds,logits)
                        else:
                            logger.info("*** Adapters are weighted according to the weight vector ***")
                            weight = i
                            logits = logits * weight_vector[weight]
                            preds = logits if preds is None else torch.add(preds,logits)



                    # Deactivate all adapters
                    model.set_active_adapters(None)
                    # Delete the added adapter
                    model.delete_adapter(adapter_name)

                if data_args.adapter_weighting == 'uniform' and len(weight_vector) == 0:
                    preds = preds / len(data_args.adapter_dir)



                loss_fct = torch.nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(preds.view(-1, config.vocab_size), labels.view(-1))
                loss = masked_lm_loss.mean().detach()


                # Update containers on host
                if loss is not None:
                    losses = _nested_gather(loss.repeat(eval_batch_size))
                    losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)


                # Gather all remaining tensors and put them back on the CPU
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)

            avg_loss = np.mean(all_losses)

            # calculating perplexity
            perplexity = math.exp(avg_loss)
            print('Loss:', avg_loss, 'PP:', perplexity)

            metrics = {}

            metrics["eval_samples"] = len(eval_dataset)

            metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            end_ens = time.time()

            duration_ens = end_ens - start_ens

            if data_args.evaluate_efficiency:
                with open("log_timings.txt", 'a+') as timing_file:  
                    timing_file.write("Duration ensembling: ")
                    timing_file.write(str(duration_ens))
                    timing_file.write("\n")      

            if not training_args.do_train:
                with open(data_args.eval_file, 'a+') as outfile:
                    weight_str = '-'.join(str(e) for e in weight_vector)
                    adapters_str = '-'.join(str(a) for a in adapters_to_use)
                    outfile.write(
                        f"{model_args.model_name_or_path},{'ensemble'},{data_args.adapter_weighting},{weight_str},{adapters_str},{data_args.validation_file},{perplexity},{str(training_args.seed)},{str(data_args.top_k)}\n")

        if not data_args.combination_strategy:

            metrics = trainer.evaluate()

            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            if not training_args.do_train:
                with open(data_args.eval_file, 'a+') as outfile:
                    weight_str = '-'.join(str(e) for e in weight_vector)
                    outfile.write(f"{model_args.model_name_or_path},{data_args.combination_strategy},{data_args.adapter_weighting},{weight_str},{data_args.validation_file},{perplexity},{str(training_args.seed)},{str(data_args.top_k)}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()