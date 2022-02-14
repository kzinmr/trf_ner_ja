from multiprocessing.sharedctypes import Value
import os
import random
import re
import unicodedata
from itertools import tee
from typing import Dict, List, Union

import numpy as np
import tokenizations
import torch
from datasets import load_metric
from mojimoji import zen_to_han
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForTokenClassification,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from predictor_ja import TrfNERSlow
from span_dataset_reader import DatasetPath, QuasiDataset

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


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def align_tokens_with_words_bad(words: List[str], tokens: List[str]) -> List[int]:
    """FastTokenizer の BatchEncoding.word_ids() をトークナイズ結果から計算."""

    def _pairwise(iterable):
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def _normalize(t: str) -> str:
        t = zen_to_han(t, kana=False, ascii=True, digit=True)
        t = unicodedata.normalize("NFKD", t)
        t = re.sub(" +", "", t)
        return t

    word_ids = [None]  # '[CLS]'
    _cursor = 0
    subword = ""
    for tok, next_tok in _pairwise(tokens[1:]):
        tok = _normalize(tok)
        next_tok = _normalize(next_tok)
        _word = words[_cursor]
        _word = _normalize(_word)
        if tok == _word and not next_tok.startswith("##"):
            word_ids.append(_cursor)
            _cursor += 1
            subword = ""
        elif _word.startswith(tok + re.sub("##", "", next_tok)):
            subword = tok
            word_ids.append(_cursor)
        else:
            if subword and subword + re.sub("##", "", tok) in _word:
                subword = subword + re.sub("##", "", tok)
                word_ids.append(_cursor)
            else:
                _cursor += 1
                if _cursor < len(words):
                    word_ids.append(_cursor)
                    __word = words[_cursor]
                    __word = _normalize(__word)
                    if tok == __word or tok == "[UNK]":
                        subword = ""
                    elif __word.startswith(tok):
                        subword = tok
                    else:
                        # print(tok, words[_cursor])
                        # print(words)
                        # print(tokens)
                        raise ValueError(
                            f"word-token alignment failed.. {_cursor} {tok} {words[_cursor]}"
                        )
                else:
                    # print(words)
                    # print(tokens)
                    raise ValueError(
                        f"word-token alignment failed.. {_cursor} {len(words)} {tok} {words[-1]}"
                    )
    word_ids.append(None)  # '[CLS]'
    # assertionしたいが、正規化など含めて一致をとるのが手間なため省く
    # for tok, wid in zip(tokens, word_ids):
    #     assert wid is None or tok == '[UNK]' or tok.replace("##", "") in words[wid]
    return word_ids


def align_tokens_with_words(words: List[str], tokens: List[str]) -> List[int]:
    w2t, t2w = tokenizations.get_alignments(words, tokens[1:-1])
    word_ids = [None]
    prev_wid = 0
    for wids in t2w:
        if len(wids) > 0:
            word_ids.append(wids[0])
            prev_wid = wids[0]
        else:
            word_ids.append(prev_wid + 1)  # [UNK] のケース
            prev_wid  += 1
    word_ids.append(None)
    return word_ids

def tokenize_and_align_labels(
    examples: Dict[str, List[List]],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    label_all_tokens=True,
) -> BatchEncoding:
    """Tokenizer結果とPreTokenizer結果のラベルのアラインメントをとる."""
    # conll2003形式ではPreTokenize済みなため is_split_into_words=True
    token_batches = examples["tokens"]
    label_batches = examples["ner_tags"]
    tokenized_inputs = tokenizer(
        token_batches, truncation=True, is_split_into_words=True
    )

    # word-token alignment は FastTokenizer経由しか使えない
    if tokenizer.is_fast:
        word_ids_batches = [
            tokenized_inputs.word_ids(batch_index=i)
            for i, _ in enumerate(label_batches)
        ]
        label_wids = zip(label_batches, word_ids_batches)
    else:
        # cl-tohoku のケース(WordPieceのprefixが ## であること)を想定した対応
        subtokens_batches = [
            tokenizer.convert_ids_to_tokens(ids)
            for ids in tokenized_inputs["input_ids"]
        ]
        # special_tokens = set(tokenizer.special_tokens_map.values())
        label_wids = []
        for tokens, subtokens, tags in zip(
            token_batches, subtokens_batches, label_batches
        ):
            try:
                word_ids = align_tokens_with_words(tokens, subtokens)
                label_wids.append((tags, word_ids))
            except ValueError:
                # 元の文の分かち書き(word)と、文窓の分かち書き(token)とが一致しないことがある
                continue

    label_ids_list = []
    for tags, word_ids in label_wids:
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None.
            # We set the label to -100 so they are automatically ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(tags[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100,
            # NOTE: チャンク先頭の単語がsubtoken化されると、 B-PERSON, B-PERSON と続いたりする
            else:
                label_ids.append(tags[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        label_ids_list.append(label_ids)

    tokenized_inputs["labels"] = label_ids_list
    return tokenized_inputs


class QuasiCoNLL2003TokenClassificationFeatures:
    train_datasets: List[Dict[str, int]]
    val_datasets: List[Dict[str, int]]
    test_datasets: List[Dict[str, int]]
    label_list: List[str]
    id2label: Dict[int, str]
    label2id: Dict[str, int]

    def __init__(
        self,
        dataset: QuasiDataset,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        label_all_tokens=True,
        batch_size=32,
    ):
        """
        Build features for transformers.
        # Dataset spec.
        # - train/val/test splitted
        # - pretokenized and token-label aligned
        """

        # ラベルのdecoding情報
        self.label_list = dataset.label_list
        self.label2id = dataset.label2id
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Prepare actual input for transformers
        self.tokenizer = tokenizer
        self.label_all_tokens = label_all_tokens

        _train = dataset.train_batch(batch_size)
        _valid = dataset.validation_batch(batch_size)
        _test = dataset.test_batch(batch_size)
        assert len(_train) > 0 and len(_valid) > 0 and len(_test) > 0
        _train_tokenized = map(self._tokenize_and_align_labels, _train)
        _valid_tokenized = map(self._tokenize_and_align_labels, _valid)
        _test_tokenized = map(self._tokenize_and_align_labels, _test)

        # Set features: input_ids, attention_mask, and labels
        self.train_datasets = [
            {
                k: d[k]
                for k in [
                    "attention_mask",
                    "input_ids",
                    "labels",
                ]
            }
            for d in _train_tokenized
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
            for d in _valid_tokenized
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
            for d in _test_tokenized
        ]

    def _tokenize_and_align_labels(self, examples: Dict[str, List[List]]):
        return tokenize_and_align_labels(
            examples,
            self.tokenizer,
            self.label_all_tokens,
        )


def predict(trainer, test_dataset):
    """Trainer経由のNER予測.
    NOTE: window処理は事前に済ませてある前提.
    """
    output = trainer.predict(test_dataset)

    input_ids = [d["input_ids"] for d in test_dataset]
    labels = [d["labels"] for d in test_dataset]
    predictions = np.argmax(output.predictions, axis=2)

    # decode
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    # 予測だけだと label_id==-100 が確約できないため、先頭に [CLS] 末尾に [SEP] が来る事実を用いる
    true_predictions = [[label_list[p] for p in preds[1:-1]] for preds in predictions]
    input_tokens = [tokenizer.convert_ids_to_tokens(ids[1:-1]) for ids in input_ids]
    return input_tokens, true_predictions, true_labels


if __name__ == "__main__":

    data_dir = "/app/workspace/"
    is_splitted = True

    model_checkpoint = "cl-tohoku/bert-base-japanese-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if is_splitted:
        train = os.path.join(data_dir, "train.jsonl")
        valid = os.path.join(data_dir, "valid.jsonl")
        test = os.path.join(data_dir, "test.jsonl")
        pathinfo = DatasetPath(train=train, validation=valid, test=test)
    else:
        filename = os.path.join(data_dir, "dataset.jsonl")
        pathinfo = DatasetPath(whole=filename, validation_ratio=0.2, test_ratio=0.2)

    dataset = QuasiDataset.load_from_span_dataset(pathinfo)

    # NOTE: CoNLL2003ではデータセット側で長さの調整（window処理）は事前に済ませてあると想定
    features = QuasiCoNLL2003TokenClassificationFeatures(
        dataset, tokenizer, label_all_tokens=True
    )
    train_dataset = features.train_datasets
    val_dataset = features.val_datasets
    test_dataset = features.test_datasets
    label_list = features.label_list

    # Build Trainer:
    # - DataLoaderのラッパー (batcher, sampler, collator)
    # - Training Loop管理
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
        # dataloader_drop_last=True,
    )
    # DataCollatorForTokenClassification:
    # - 各バッチサンプルに対して tokenizer.pad() が呼ばれ、torch.Tensorが返される
    # - バッチ内のトークン単位ラベルも処理される (See. DataCollatorWithPadding)
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
    # Avoid the default "LABEL_0"-style labeling
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {v: k for k, v in label2id.items()}
    trainer.model.config.label2id = label2id
    trainer.model.config.id2label = id2label

    trainer.train()

    trainer.evaluate()

    trainer.save_model(data_dir)
    TrfNERSlow.pickle_bert_model(data_dir, os.path.join(data_dir, "predictor_ja.pkl"))

    # prediction
    input_tokens, true_predictions, true_labels = predict(trainer, test_dataset)
    with open(os.path.join(data_dir, "test.tsv"), "w") as fp:
        for tokens, golds, preds in zip(input_tokens, true_labels, true_predictions):
            for tok, g, p in zip(tokens, golds, preds):
                fp.write(f"{tok}\t{g}\t{p}\n")
            fp.write("\n")
