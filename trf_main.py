import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertTokenizerFast,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from pl_datamodule_trf import ExamplesBuilder, TokenClassificationDataModule
from pl_vocabulary_trf import custom_tokenizer_from_pretrained

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class CoNLL2003TokenClassificationFeatures:
    """
    Build feature dataset so that the model can load
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        tokens_per_batch: int = 32,
        window_stride: Optional[int] = None,
        label_all_tokens=True,
    ):
        """tokenize_and_align_labels with long text (i.e. truncation is disabled)"""

        self.tokenizer = tokenizer
        self.label_all_tokens = label_all_tokens

        if window_stride is None or window_stride <= 0:
            self.window_stride = tokens_per_batch
        elif window_stride > 0 and window_stride < tokens_per_batch:
            self.window_stride = window_stride
        self.tokens_per_batch = tokens_per_batch

        # self.features: List[InputFeatures] = []
        # self.examples: List[TokenClassificationExample] = []

        datasets = load_dataset("conll2003")
        # input_ids, attention_mask
        tokenized_datasets = datasets.map(self.tokenize_and_align_labels, batched=True)
        # ['attention_mask', 'chunk_tags', 'id', 'input_ids', 'labels', 'ner_tags', 'pos_tags', 'token_type_ids', 'tokens']
        self.train_datasets = [
            {
                k: d[k]
                for k in [
                    "attention_mask",
                    "input_ids",
                    "labels",
                ]
            }
            for d in tokenized_datasets["train"]
        ]
        self.val_datasets = [
            {
                k: d[k]
                for k in [
                    "attention_mask",
                    "input_ids",
                    "labels",
                ]
            }
            for d in tokenized_datasets["validation"]
        ]
        self.test_datasets = [
            {
                k: d[k]
                for k in [
                    "attention_mask",
                    "input_ids",
                    "labels",
                ]
            }
            for d in tokenized_datasets["test"]
        ]

        self.label_list = datasets["train"].features["ner_tags"].feature.names

    def tokenize_and_align_labels(self, examples):
        """
        >> datasets = load_dataset("conll2003")
        >> tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

        example = datasets["train"][0]  # input_ids, attention_mask
        tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
        # len(example["ner_tags"]) != len(tokenized_input["input_ids"])
        word_ids = tokenized_input.word_ids()
        aligned_labels = [PAD_TOKEN_LABEL_ID if i is None else example["ner_tags"][i] for i in word_ids]
        assert len(aligned_labels) == len(tokenized_input["input_ids"])
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        label_ids_list = []
        for i, tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(tags[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100,
                # depending on the label_all_tokens flag.
                else:
                    label_ids.append(tags[word_idx] if self.label_all_tokens else -100)
                previous_word_idx = word_idx

            label_ids_list.append(label_ids)

        tokenized_inputs["labels"] = label_ids_list
        return tokenized_inputs


def make_common_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--data_dir",
        default="/app/workspace/data",
        type=str,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        help="Path to pretrained model config (for transformers)",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        type=str,
        help="Path to pretrained tokenzier JSON config (for transformers)",
    )
    parser.add_argument(
        "--labels",
        default="workspace/data/label_types.txt",
        type=str,
        help="Path to a file containing all labels. (for transformers)",
    )
    parser.add_argument("--nbest", default=1, type=int, help="CNN(to be implemented)")
    parser.add_argument("--number_normalized", default=True, type=bool, help="CNN")
    return parser

def build_args(model_checkpoint=None):
    parser = make_common_args()
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    # parser = TokenClassificationModule.add_model_specific_args(parent_parser=parser)
    parser = TokenClassificationDataModule.add_model_specific_args(parent_parser=parser)
    args = parser.parse_args()
    # if notebook:        
    #     args = parser.parse_args(args=[])
    pl.seed_everything(args.seed)

    if model_checkpoint:
        args.model_name_or_path = model_checkpoint

    args.gpu = torch.cuda.is_available()
    # args.num_samples = 20000

    args.delimiter = "\t"
    args.is_bio = False
    args.scheme = "bio"
    return args

if __name__ == "__main__":

    conll03 = False
    ja_gsd = False
    custom_pretrained = False
    data_dir = "/app/workspace/"

    # make dataset and tokenizer
    if conll03:
        # en-model & en-datasets
        model_checkpoint = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        features = CoNLL2003TokenClassificationFeatures(
            tokenizer, label_all_tokens=True
        )

        train_dataset = features.train_datasets
        val_dataset = features.val_datasets
        test_dataset = features.test_datasets
        label_list = features.label_list
    else:
        args = build_args()
        args.model_name_or_path = "cl-tohoku/bert-base-japanese-v2"
        model_checkpoint = args.model_name_or_path

        if custom_pretrained:
            # args.tokenizer_path = "tokenizer.json"
            tokenizer = custom_tokenizer_from_pretrained(
                args.tokenizer_path
            )
        else:
            args.tokenizer_path = model_checkpoint
            tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)

        if ja_gsd:
            ExamplesBuilder.download_dataset(data_dir)
        if not (Path(data_dir) / f"train.txt").exists():
            exit(0)

        dm = TokenClassificationDataModule(args)
        dm.prepare_data()

        train_dataset = dm.train_dataset.to_dict()
        val_dataset = dm.val_dataset.to_dict()
        test_dataset = dm.test_dataset.to_dict()
        label_list = dm.label_list

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, num_labels=len(label_list)
    )
    args = TrainingArguments(
        "trf-ner",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()

    # prediction
    output = trainer.predict(test_dataset)
    print(output.metrics)
    
    input_ids = [d["input_ids"] for d in test_dataset]
    predictions = np.argmax(output.predictions, axis=2)
    labels = [d["labels"] for d in test_dataset]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    input_tokens = [
        tokenizer.convert_ids_to_tokens([i for (i, l) in zip(ids, label) if l != -100])
        for ids, label in zip(input_ids, labels)
    ]
    with open(os.path.join(data_dir, 'test.tsv'), 'w') as fp:
        for tokens, golds, preds in zip(input_tokens, true_labels, true_predictions):
            for tok, g, p in zip(tokens, golds, preds):
                fp.write(f'{tok}\t{g}\t{p}\n')
            fp.write('\n')

    trainer.save_model(data_dir)