import json
import logging
import os
import time
from argparse import ArgumentParser, Namespace
from itertools import starmap
from pathlib import Path
from typing import List, Optional, Union

import pytorch_lightning as pl
import requests
import torch
from datasets import load_dataset
from tokenizers import Encoding
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BatchEncoding,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
)

from data import *
from pl_vocabulary_trf import (
    LabelTokenAligner,
    custom_tokenizer_from_pretrained,
)

logger = logging.getLogger(__name__)


class NoLocalFileError(Exception):
    pass


class MyDataset(Dataset):
    def __init__(
        self,
        features: List[InputFeatures],
    ):
        self.features = features
        self._n_features = len(self.features)

    def __len__(self):
        return self._n_features

    def __getitem__(self, idx) -> InputFeatures:
        return self.features[idx]


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
        # input_ids, attention_mask, label_ids
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

        self.train_features = [
            InputFeatures(
                input_ids=dataset["input_ids"],
                attention_mask=dataset["attention_mask"],
                label_ids=dataset["labels"],
            )
            for dataset in tokenized_datasets["train"]
        ]
        self.val_features = [
            InputFeatures(
                input_ids=dataset["input_ids"],
                attention_mask=dataset["attention_mask"],
                label_ids=dataset["labels"],
            )
            for dataset in tokenized_datasets["validation"]
        ]
        self.test_features = [
            InputFeatures(
                input_ids=dataset["input_ids"],
                attention_mask=dataset["attention_mask"],
                label_ids=dataset["labels"],
            )
            for dataset in tokenized_datasets["test"]
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

        label_ids_list: IntListList = []
        for i, tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids: IntList = []
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

    @staticmethod
    def bio2biolu(labels: StrList) -> StrList:
        """TODO: label2label_id for datasets"""
        new_labels = []
        n_lines = len(labels)
        for i, current_label in enumerate(labels):
            next_iob = None
            if i < n_lines - 1:
                next_label = labels[i + 1]
                next_iob = next_label[0]

            tag_type = current_label[2:]
            iob = current_label[0]

            iob_transition = (iob, next_iob)
            current_iob = iob
            if iob_transition == ("B", "I"):
                current_iob = "B"
            elif iob_transition == ("I", "I"):
                current_iob = "I"
            elif iob_transition in {("B", "O"), ("B", "B"), ("B", None)}:
                current_iob = "U"
            elif iob_transition in {("I", "B"), ("I", "O"), ("I", None)}:
                current_iob = "L"
            elif iob == "O":
                current_iob = "O"
            else:
                logger.warning(f"Invalid BIO transition: {iob_transition}")
                if iob not in set("BIOLU"):
                    current_iob = "O"
            biolu = f"{current_iob}-{tag_type}" if current_iob != "O" else "O"

            new_labels.append(biolu)
        return new_labels


class ExamplesBuilder:
    def __init__(
        self,
        data_dir: str,
        split: Split,
        delimiter: str = "\t",
        is_bio: bool = False,
    ):
        datadir_p = Path(data_dir)
        if (datadir_p / f"{split.value}.txt").exists():

            start = time.time()
            self.token_examples = self.read_examples_from_file(
                data_dir, split, delimiter=delimiter, is_bio=is_bio
            )
            end = time.time()
            read_time = end - start
            print(f"READ TIME({split.value}): {read_time}")
            start = time.time()
            self.examples = self.convert_spandata(self.token_examples)
            end = time.time()
            conv_time = end - start
            print(f"CONVERT TIME({split.value}): {conv_time}")
        elif (datadir_p / f"{split.value}.jsonl").exists():
            self.examples = self.read_span_examples(data_dir, split)
        else:
            raise NoLocalFileError
            # download_dataset(data_dir)
            # datasets = load_dataset("conll2003")
            # self.examples = datasets[split.value if split.value != "dev" else "validation"]

        # print(f"0-th sentence length: {len(self.examples[0].content)}")
        # print(self.examples[0].content[:10])
        # print(self.examples[0].annotations)

    @staticmethod
    def download_dataset(data_dir: Union[str, Path]):
        def _download_data(url, file_path):
            response = requests.get(url)
            if response.ok:
                with open(file_path, "w") as fp:
                    fp.write(response.content.decode("utf8"))
                return file_path

        for mode in Split:
            mode = mode.value
            print(f"Fetching {mode} dataset...")
            url = f"https://github.com/megagonlabs/UD_Japanese-GSD/releases/download/v2.6-NE/{mode}.bio"
            file_path = os.path.join(data_dir, f"{mode}.txt")
            _download_data(url, file_path)

    @staticmethod
    def is_boundary_line(line: str) -> bool:
        return line.startswith("-DOCSTART-") or line == "" or line == "\n"

    @staticmethod
    def bio2biolu(
        lines: StrList, label_idx: int = -1, delimiter: str = "\t"
    ) -> StrList:
        new_lines = []
        n_lines = len(lines)
        for i, line in enumerate(lines):
            if ExamplesBuilder.is_boundary_line(line):
                new_lines.append(line)
            else:
                next_iob = None
                if i < n_lines - 1:
                    next_line = lines[i + 1].strip()
                    if not ExamplesBuilder.is_boundary_line(next_line):
                        next_iob = next_line.split(delimiter)[label_idx][0]

                line = line.strip()
                current_line_content = line.split(delimiter)
                current_label = current_line_content[label_idx]
                word = current_line_content[0]
                tag_type = current_label[2:]
                iob = current_label[0]

                iob_transition = (iob, next_iob)
                current_iob = iob
                if iob_transition == ("B", "I"):
                    current_iob = "B"
                elif iob_transition == ("I", "I"):
                    current_iob = "I"
                elif iob_transition in {("B", "O"), ("B", "B"), ("B", None)}:
                    current_iob = "U"
                elif iob_transition in {("I", "B"), ("I", "O"), ("I", None)}:
                    current_iob = "L"
                elif iob == "O":
                    current_iob = "O"
                else:
                    logger.warning(f"Invalid BIO transition: {iob_transition}")
                    if iob not in set("BIOLU"):
                        current_iob = "O"
                biolu = f"{current_iob}-{tag_type}" if current_iob != "O" else "O"
                new_line = f"{word}{delimiter}{biolu}"
                new_lines.append(new_line)
        return new_lines

    @staticmethod
    def read_lines(data_dir, mode, ext=".txt"):
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}{ext}")
        lines = []
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        return lines

    @staticmethod
    def read_examples_from_file(
        data_dir: str,
        mode: Union[Split, str],
        label_idx: int = -1,
        delimiter: str = "\t",
        is_bio: bool = True,
    ) -> List[TokenClassificationExample]:
        """
        Read token-wise data like CoNLL2003 from file
        """
        # lines = ExamplesBuilder.read_lines(data_dir, mode, ".txt")
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        lines = []
        with open(file_path, encoding="utf-8") as f:
            lines = f.read().split("\n")

        if is_bio:
            lines = ExamplesBuilder.bio2biolu(lines)
        guid_index = 1
        examples = []
        words = []
        labels = []
        for line in lines:
            if ExamplesBuilder.is_boundary_line(line):
                if words:
                    examples.append(
                        TokenClassificationExample(
                            guid=f"{mode}-{guid_index}", words=words, labels=labels
                        )
                    )
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.strip().split(delimiter)
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[label_idx])
                else:
                    # for mode = "test"
                    labels.append("O")
        if words:
            examples.append(
                TokenClassificationExample(
                    guid=f"{mode}-{guid_index}", words=words, labels=labels
                )
            )
        return examples

    @staticmethod
    def _get_original_spans(words, text):
        word_spans = []
        start = 0
        for w in words:
            word_spans.append((start, start + len(w)))
            start += len(w)
        assert words == [text[s:e] for s, e in word_spans]
        return word_spans

    @staticmethod
    def convert_spandata(
        examples: List[TokenClassificationExample],
    ) -> List[StringSpanExample]:
        """
        Convert token-wise data like CoNLL2003 into string-wise span data
        """

        new_examples: List[StringSpanExample] = []
        for example in examples:
            words = example.words
            text = "".join(words)
            labels = example.labels
            annotations: List[SpanAnnotation] = []

            word_spans = ExamplesBuilder._get_original_spans(words, text)
            label_span = []
            labeltype = ""
            for span, label in zip(word_spans, labels):
                if label == "O" and label_span and labeltype:
                    start, end = label_span[0][0], label_span[-1][-1]
                    annotations.append(
                        SpanAnnotation(start=start, end=end, label=labeltype)
                    )
                    label_span = []
                elif label != "O":
                    labeltype = label[2:]
                    label_span.append(span)
            if label_span and labeltype:
                start, end = label_span[0][0], label_span[-1][-1]
                annotations.append(
                    SpanAnnotation(start=start, end=end, label=labeltype)
                )

            new_examples.append(
                StringSpanExample(
                    guid=example.guid, content=text, annotations=annotations
                )
            )
        return new_examples

    @staticmethod
    def read_span_examples(
        data_dir: str,
        mode: Union[Split, str],
    ) -> List[StringSpanExample]:
        """
        Read span data like camphr input into string-wise span data.
        https://camphr.readthedocs.io/en/latest/notes/finetune_transformers.html#id13
        """

        # lines = ExamplesBuilder.read_lines(data_dir, mode, ".jsonl")
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.jsonl")
        lines = []
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]

        span_examples: List[StringSpanExample] = []
        for i, jl in enumerate(lines):
            jd = json.loads(jl)
            if "meta" in jd and "doc_id" in jd["meta"]:
                guid = jd["meta"]["doc_id"]
            else:
                guid = str(i)
            text = jd[0]
            annotations = [
                SpanAnnotation(start=tpl[0], end=tpl[1], label=tpl[2])
                for tpl in jd[1]["entities"]
                if len(tpl) == 3
            ]
            span_examples.append(
                StringSpanExample(guid=guid, content=text, annotations=annotations)
            )
        return span_examples


class TokenClassificationDataset(Dataset):
    """
    Build feature dataset so that the model can load
    """

    def __init__(
        self,
        examples: List[StringSpanExample],
        tokenizer: PreTrainedTokenizerFast,
        label_token_aligner: LabelTokenAligner,
        tokens_per_batch: int = 32,
        window_stride: int = -1,
    ):
        """tokenize_and_align_labels with long text (i.e. truncation is disabled)"""

        self.window_stride = tokens_per_batch
        if window_stride > 0 and window_stride < tokens_per_batch:
            self.window_stride = window_stride
        self.tokens_per_batch = tokens_per_batch

        self.tokenizer = tokenizer
        self.label_token_aligner = label_token_aligner
        self.label_all_tokens = True

        self.features: List[InputFeatures] = []
        self.examples: List[TokenClassificationExample] = []
        self.tokenize_and_align_labels_v1(examples)
        self._n_features = len(self.features)
        self.features_dict = self.to_dict()

    def tokenize_and_align_labels_v1(self, examples):
        # tokenize text into subwords
        texts: StrList = [ex.content for ex in examples]
        tokenized_batch: BatchEncoding = self.tokenizer(texts, add_special_tokens=False)
        encodings: List[Encoding] = tokenized_batch.encodings

        # align character span labels with subwords
        annotations: List[List[SpanAnnotation]] = [ex.annotations for ex in examples]
        aligned_label_ids: IntListList = list(
            starmap(
                self.label_token_aligner.align_labels_with_tokens,
                zip(encodings, annotations),
            )
        )

        # perform manual padding and register features
        # TODO: padding using Encoding
        guids: StrList = [ex.guid for ex in examples]
        for guid, encoding, label_ids in zip(guids, encodings, aligned_label_ids):
            seq_length = len(label_ids)
            window_spans = [
                (start, min(start + self.tokens_per_batch, seq_length))
                for start in range(0, seq_length, self.window_stride)
            ]
            for start, end in window_spans:
                n_padding_to_add = max(0, self.tokens_per_batch - end + start)
                input_ids = (
                    encoding.ids[start:end]
                    + [self.tokenizer.pad_token_id] * n_padding_to_add
                )
                label_ids = (
                    label_ids[start:end] + [PAD_TOKEN_LABEL_ID] * n_padding_to_add
                )
                attention_mask = (
                    encoding.attention_mask[start:end] + [0] * n_padding_to_add
                )
                self.features.append(
                    InputFeatures(
                        input_ids=input_ids,
                        label_ids=label_ids,
                        attention_mask=attention_mask,
                    )
                )

                subwords = encoding.tokens[start:end]
                labels = [
                    self.label_token_aligner.ids_to_label[i]
                    for i in label_ids[start:end]
                ]
                self.examples.append(
                    TokenClassificationExample(guid=guid, words=subwords, labels=labels)
                )

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs: BatchEncoding = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        label_ids_list: IntListList = []
        for i, tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids: IntList = []
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

    def __len__(self):
        return self._n_features

    def __getitem__(self, idx) -> InputFeatures:
        return self.features[idx]

    def to_dict(self):
        return [
            {
                "input_ids": self.features[ix].input_ids,
                "attention_mask": self.features[ix].attention_mask,
                "labels": self.features[ix].label_ids,
            }
            for ix in range(self._n_features)
        ]


class InputFeaturesBatch:
    def __init__(self, features: List[InputFeatures]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.label_ids: Optional[torch.Tensor]

        self._n_features = len(features)
        input_ids_list: IntListList = []
        masks_list: IntListList = []
        label_ids_list: IntListList = []
        for f in features:
            input_ids_list.append(f.input_ids)
            masks_list.append(f.attention_mask)
            if f.label_ids is not None:
                label_ids_list.append(f.label_ids)
        self.input_ids = torch.LongTensor(input_ids_list)
        self.attention_mask = torch.LongTensor(masks_list)
        if label_ids_list:
            self.label_ids = torch.LongTensor(label_ids_list)

    def __len__(self):
        return self._n_features

    def __getitem__(self, item):
        return getattr(self, item)


class TokenClassificationDataModule(pl.LightningDataModule):
    """
    Prepare dataset and build DataLoader
    """

    def __init__(self, hparams: Namespace):
        self.tokenizer: PreTrainedTokenizerFast
        self.train_dataset: TokenClassificationDataset
        self.val_dataset: TokenClassificationDataset
        self.test_dataset: TokenClassificationDataset
        # self.label_token_aligner: LabelTokenAligner
        # self.train_examples: List[StringSpanExample]
        # self.val_examples: List[StringSpanExample]
        # self.test_examples: List[StringSpanExample]

        super().__init__()

        self.cache_dir = hparams.cache_dir if hparams.cache_dir else None
        if self.cache_dir is not None and not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.data_dir = hparams.data_dir
        # if not os.path.exists(self.data_dir):
        #     os.mkdir(self.data_dir)

        self.tokenizer_name_or_path = (
            hparams.tokenizer_path
            if hparams.tokenizer_path
            else hparams.model_name_or_path
        )
        self.labels_path = hparams.labels

        self.train_batch_size = hparams.train_batch_size
        self.eval_batch_size = hparams.eval_batch_size
        self.num_workers = hparams.num_workers
        self.num_samples = hparams.num_samples

        self.tokens_per_batch = hparams.tokens_per_batch
        self.window_stride = hparams.window_stride

        self.delimiter = hparams.delimiter
        self.is_bio = hparams.is_bio
        self.bilou = hparams.scheme == "bilou"

    def prepare_data(self):
        """
        Downloads the data and prepare the tokenizer
        """
        self.tokenizer = custom_tokenizer_from_pretrained(
            self.tokenizer_name_or_path, self.cache_dir
        )
        try:
            self.train_examples = ExamplesBuilder(
                self.data_dir,
                Split.train,
                delimiter=self.delimiter,
                is_bio=self.is_bio,
            ).examples
            self.val_examples = ExamplesBuilder(
                self.data_dir,
                Split.dev,
                delimiter=self.delimiter,
                is_bio=self.is_bio,
            ).examples
            self.test_examples = ExamplesBuilder(
                self.data_dir,
                Split.test,
                delimiter=self.delimiter,
                is_bio=self.is_bio,
            ).examples

            if self.num_samples > 0:
                self.train_examples = self.train_examples[: self.num_samples]
                self.val_examples = self.val_examples[: self.num_samples]
                self.test_examples = self.test_examples[: self.num_samples]

            # create label vocabulary from dataset
            all_examples = self.train_examples + self.val_examples + self.test_examples
            all_labels = {
                f"{bio}-{anno.label}" if bio != "O" else "O"
                for ex in all_examples
                for anno in ex.annotations
                for bio in ("BILOU" if self.bilou else "BIO")
            }
            self.label_list = sorted(all_labels)
            print(self.label_list)
            if not os.path.exists(self.labels_path):
                label_types = sorted({l[2:] for l in sorted(all_labels) if l != "O"})
                with open(self.labels_path, "w") as fp:
                    for l in label_types:
                        fp.write(l)
                        fp.write("\n")
            self.label_token_aligner = LabelTokenAligner(self.labels_path, self.bilou)

            start = time.time()
            self.train_dataset = self.create_dataset(
                self.train_examples, self.tokenizer, self.label_token_aligner
            )
            end = time.time()
            read_time = end - start
            print(f"DATASET TIME(train): {read_time}")

            start = time.time()
            self.val_dataset = self.create_dataset(
                self.val_examples, self.tokenizer, self.label_token_aligner
            )
            end = time.time()
            read_time = end - start
            print(f"DATASET TIME(val): {read_time}")

            start = time.time()
            self.test_dataset = self.create_dataset(
                self.test_examples, self.tokenizer, self.label_token_aligner
            )
            end = time.time()
            read_time = end - start
            print(f"DATASET TIME(test): {read_time}")

            self.dataset_size = len(self.train_dataset)

            self.use_datasets = False
            self.data_collator = InputFeaturesBatch

        except NoLocalFileError:

            features = CoNLL2003TokenClassificationFeatures(self.tokenizer)
            # input_ids, attention_mask, label_ids
            self.train_dataset = (
                features.train_datasets
            )  # MyDataset(features.train_features)
            self.val_dataset = features.val_datasets  # MyDataset(features.val_features)
            self.test_dataset = (
                features.test_datasets
            )  # MyDataset(features.test_features)
            self.dataset_size = len(self.train_dataset)
            self.bilou = False
            self.use_datasets = True
            self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
            # create label vocabulary from dataset
            if not os.path.exists(self.labels_path):
                all_labels = features.label_list
                label_types = sorted({l[2:] for l in sorted(all_labels) if l != "O"})
                with open(self.labels_path, "w") as fp:
                    fp.write("\n".join(label_types))

            # print(self.val_dataset[0].keys())
            # print(self.val_dataset[:10])

    def setup(self, stage=None):
        """
        split the data into train, test, validation data
        :param stage: Stage - training or testing
        """
        # our dataset is splitted in prior

    def create_dataset(
        self, data: List[StringSpanExample], tokenizer, label_token_aligner
    ) -> TokenClassificationDataset:
        return TokenClassificationDataset(
            data,
            tokenizer,
            label_token_aligner,
            tokens_per_batch=self.tokens_per_batch,
            window_stride=self.window_stride,
        )

    @staticmethod
    def create_dataloader(
        ds: TokenClassificationDataset,
        data_collator,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = False,
    ) -> DataLoader:

        return DataLoader(
            ds,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.create_dataloader(
            self.train_dataset,
            self.data_collator,
            self.train_batch_size,
            self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return self.create_dataloader(
            self.val_dataset,
            self.data_collator,
            self.eval_batch_size,
            self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.create_dataloader(
            self.test_dataset,
            self.data_collator,
            self.eval_batch_size,
            self.num_workers,
            shuffle=False,
        )

    def total_steps(self) -> int:
        """
        The number of total training steps that will be run. Used for lr scheduler purposes.
        """
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = (
            self.hparams.train_batch_size
            * self.hparams.accumulate_grad_batches
            * num_devices
        )
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=32,
            help="input batch size for training (default: 32)",
        )
        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=32,
            help="input batch size for validation/test (default: 32)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            metavar="N",
            help="number of workers (default: 3)",
        )
        parser.add_argument(
            "--tokens_per_batch",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--window_stride",
            default=-1,
            type=int,
            help="The stride of moving window over input sequence."
            "This must be shorter than tokens_per_batch.",
        )
        parser.add_argument(
            "--num_samples",
            type=int,
            default=100,
            metavar="N",
            help="Number of samples to be used for training and evaluation steps (default: 15000) Maximum:100000",
        )
        parser.add_argument(
            "--delimiter",
            default=" ",
            type=str,
            help="delimiter between token and label in one line.",
        )
        parser.add_argument("--is_bio", action="store_true")
        parser.add_argument(
            "--scheme",
            default="bilou",
            type=str,
            help="tag scheme.",
        )
        return parser
