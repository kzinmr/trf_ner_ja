import os

import evaluate
import numpy as np
import pickle
import random
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from data import Token
from span2conll import (
    Span2WordLabelConverter,
    WordTokenizerWithAlignment,
    bio_sorter,
)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
workdir = "/app/workspace/"

with open(os.path.join(workdir, "label_set.pkl"), "rb") as fp:
    label_set = pickle.load(fp)

label_list = sorted(label_set, key=bio_sorter)
id2label = dict(enumerate(label_list))
label2id = {l: i for i, l in id2label.items()}
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
model.eval()
model.to("cuda")


metric = evaluate.load("seqeval")


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


def _make_token_labels_from_word_token_map(
    label_wids: list[tuple[list[int], list[int]]],
    label2id: dict[str, int],
    id2label: dict[int, str],
    label_all_tokens: bool = True,
) -> list[list[int]]:
    """単語-トークン間マップを使い単語単位ラベルをトークン単位ラベルに変換する。
    TokenizerのFast/Slowに依存しない処理。
    """
    _label_ids_list = []
    for tags, word_ids in label_wids:
        previous_word_idx = None
        previous_label = None
        label_ids = []
        # word_ids: 各トークンがどの単語にあたるかのマップ
        for word_idx in word_ids:
            if word_idx is None:
                # 特殊記号に対しては-100にセット(loss計算時に無視される)
                tag = -100
            elif word_idx != previous_word_idx:
                # 単語先頭のトークンに対しては、先頭トークンのラベルをセット
                tag = tags[word_idx]
            else:
                # その他はSet the current label or -100,
                if label_all_tokens:
                    tag = tags[word_idx]
                    # チャンク先頭の単語がsubtokenで B-xxx開始ならI-xxxを続ける
                    if previous_label and len(previous_label.split("-")) > 1:
                        label = "I-" + previous_label.split("-")[1]
                        tag = label2id[label]
                else:
                    tag = -100
            label_ids.append(tag)
            previous_word_idx = word_idx
            previous_label = id2label[tag] if tag != -100 else None
        # 窓境界のI-xxx開始をOに倒す
        if label_ids and id2label[label_ids[1]].startswith("I"):
            target_id = label_ids[1]
            replace_id = label2id["O"]
            for i in range(len(label_ids)):
                if label_ids[i] == target_id:
                    label_ids[i] = replace_id
                elif label_ids[i] == -100:
                    continue
                else:
                    break

        _label_ids_list.append(label_ids)
    return _label_ids_list


def make_dataset(
    span2wordlabel_converter: Span2WordLabelConverter,
    path: str,
    tokenizer,
    id2label: dict[int, str],
    label2id: dict[str, int],
    max_length=128,
    label_all_tokens=True,
) -> list[dict]:
    # 単語単位のラベルアラインメント (span形式 -> conll03-like形式)
    # ※ sentencepieceベース手法の場合不要
    _word_labels_windows = span2wordlabel_converter.convert(path)

    # train = [
    #     {
    #         "id": i,
    #         "tokens": [t_l.token for t_l in token_labels],
    #         "labels": [t_l.label for t_l in token_labels],
    #         "ner_tags": [label2id[t_l.label] for t_l in token_labels],
    #     }
    #     for i, token_labels in enumerate(token_labels_windows_train)
    # ]

    # train_batch=32
    # make_batch(train, train_batch)

    # 各単語内の、トークン単位のラベルアラインメント
    assert tokenizer.is_fast
    word_batches = [
        [t_l.token for t_l in word_labels] for word_labels in _word_labels_windows
    ]
    # conll2003形式ではPreTokenize済みなため is_split_into_words=True
    # collatorでコケるのでpadding="max_length"で固定長化している
    enc = tokenizer(
        word_batches,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        is_split_into_words=True,
        # stride=window_stride,  # fast-only
        # return_overflowing_tokens=True,  # fast-only
        return_offsets_mapping=True,  # fast-only
    )
    # 単語-ラベルのアラインメント
    label_batches = [
        [label2id[t_l.label] for t_l in word_labels]
        for word_labels in _word_labels_windows
    ]
    word_ids_batches = [
        enc.word_ids(batch_index=i) for i, _ in enumerate(label_batches)
    ]
    _label_wids = list(zip(label_batches, word_ids_batches))

    _label_ids_list = _make_token_labels_from_word_token_map(
        _label_wids, label2id, id2label, label_all_tokens
    )

    assert all(len(d.ids) == len(lbs) for d, lbs in zip(enc.encodings, _label_ids_list))

    enc["labels"] = _label_ids_list

    return [
        {
            "attention_mask": d.attention_mask,
            "input_ids": d.ids,
            "labels": d.labels,
        }
        for d in enc.encodings
        # for ams, ids, ls in zip(d["attention_mask"], d["input_ids"], d["labels"])  # unbatch
    ]


class FastWordTokenizerWithAlignment(WordTokenizerWithAlignment):
    """FastTokenizerに含まれるWordTokenizerを利用する"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        assert self.tokenizer.is_fast

    @staticmethod
    def ordered_uniq(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if x not in seen and not seen_add(x)]

    def tokenize(self, text: str) -> list[str]:
        batch_enc = self.tokenizer(text)
        tokens = []
        for w in batch_enc.word_ids():
            if w is not None:
                span = batch_enc.word_to_chars(w)
                tokens.append(text[span.start : span.end])
        return tokens

    def tokenize_with_alignment(self, text: str) -> list[Token]:
        batch_enc = self.tokenizer(text)
        tokens = []
        for w in self.ordered_uniq(batch_enc.word_ids()):  # subword文はまとめる
            if w is not None:
                span = batch_enc.word_to_chars(w)
                tokens.append(Token(text[span.start : span.end], span.start, span.end))
        return tokens


max_length: int = 128
window_stride: int = 32
word_tokenizer = FastWordTokenizerWithAlignment(tokenizer)
converter = Span2WordLabelConverter(word_tokenizer, max_length, window_stride)

train_pth = os.path.join(workdir, "train.jsonl")
valid_pth = os.path.join(workdir, "valid.jsonl")
test_pth = os.path.join(workdir, "test.jsonl")
train_dataset = make_dataset(converter, train_pth, id2label, label2id, max_length)
valid_dataset = make_dataset(converter, valid_pth, id2label, label2id, max_length)
test_dataset = make_dataset(converter, test_pth, id2label, label2id, max_length)


# Build Trainer:
# - DataLoaderのラッパー (batcher, sampler, collator)
# - Training Loop管理

args = TrainingArguments(
    os.path.join(workdir, "trf-ner"),
    num_train_epochs=15,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    # dataloader_drop_last=True,
)
# DataCollatorForTokenClassification:
# - 各バッチサンプルに対して tokenizer.pad() が呼ばれ、torch.Tensorが返される
# - バッチ内のトークン単位ラベルも処理される (See. DataCollatorWithPadding)
data_collator = DataCollatorForTokenClassification(
    tokenizer,
    padding="max_length",
    max_length=max_length,
    # pad_to_multiple_of=8,
    label_pad_token_id=-100,
    return_tensors="pt",
)
trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()
trainer.save_model(workdir)


def pickle_bert_model(model_dir: str, model_out_path: str):
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_dict = {"tokenizer": tokenizer, "model": model}
    with open(model_out_path, "wb") as fp:
        pickle.dump(model_dict, fp)


pickle_bert_model(workdir, os.path.join(workdir, "predictor.pkl"))

# prediction
output = trainer.predict(test_dataset)

input_ids = [d["input_ids"] for d in test_dataset]
labels = [d["labels"] for d in test_dataset]
predictions = np.argmax(output.predictions, axis=2)

# decode
true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
# 予測だけだと label_id==-100 が確約できないため、先頭に [CLS] 末尾に [SEP] が来る事実を用いる
true_predictions = [[label_list[p] for p in preds[1:-1]] for preds in predictions]
input_tokens = [tokenizer.convert_ids_to_tokens(ids[1:-1]) for ids in input_ids]

with open(os.path.join(workdir, "test.tsv"), "w") as fp:
    for tokens, golds, preds in zip(input_tokens, true_labels, true_predictions):
        for tok, g, p in zip(tokens, golds, preds):
            fp.write(f"{tok}\t{g}\t{p}\n")
        fp.write("\n")
