import os
import pickle
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain, tee
from typing import Dict, List

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
    TokenClassifierOutput,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class BIO(Enum):
    B = "B"
    I = "I"
    O = "O"


class TagType(Enum):
    Other = "Other"
    Target = "TARGET"


@dataclass
class BIOTag:
    _label: str
    bio: BIO = field(init=False)
    tagtype: TagType = field(init=False)
    label: str = field(init=False)

    def __post_init__(self):
        label = self._label
        if len(label.split("-")) == 2:
            bio_repr, tagtype_repr = label.split("-")
            try:
                self.bio = BIO(bio_repr)
                self.tagtype = TagType(tagtype_repr)
            except ValueError:
                self.bio = BIO.O
                self.tagtype = TagType.Other
        else:
            self.bio = BIO.O
            self.tagtype = TagType.Other
        self.label = (
            f"{self.bio.value}-{self.tagtype.value}" if self.bio != BIO.O else "O"
        )

    @staticmethod
    def from_values(bio: BIO, tag_type: TagType):
        if bio != BIO.O:
            return BIOTag(f"{bio.value}-{tag_type.value}")
        else:
            return BIOTag("O")


@dataclass
class Token:
    text: str
    start: int
    end: int


@dataclass
class TokenLabelPair:
    token: Token
    label: BIOTag

    @staticmethod
    def build(text: str, start_pos: int, label: BIOTag):
        end_pos = start_pos + len(text)
        return TokenLabelPair(Token(text, start_pos, end_pos), label)

    @property
    def start_pos(self) -> int:
        return self.token.start

    @property
    def end_pos(self) -> int:
        return self.token.end


@dataclass
class Chunk:
    text: str
    start_pos: int
    label: TagType


class FastEncoder:
    tokenizer: PreTrainedTokenizerFast
    max_length: int
    window_stride: int

    def __init__(self, tokenizer, max_length, window_stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_stride = window_stride

    def encode(self, sentence_text: str):
        """Tokenizationを用いてテキストをトークナイズ・Tensor化する。
        NOTE: デコーディングのために offset_mapping も保持する。
        NOTE: 長文の対策として、ストライド付きウィンドウ分割も行う。
        """
        # sentence -> [sentence_window(max_length, window_stride)]
        enc: BatchEncoding = self.tokenizer(
            sentence_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.window_stride,
            return_overflowing_tokens=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            return_length=True,
        )
        dataset = [
            {"input_ids": encoding.ids, "attention_mask": encoding.attention_mask}
            for encoding in enc.encodings
        ]
        return dataset, enc.offset_mapping


class Decoder:
    id2label: dict[int, BIOTag]
    window_stride: int

    def __init__(self, id2label: dict[int, BIOTag], window_stride: int):
        self.id2label = id2label
        self.window_stride = window_stride

    def unwindow(
        self, tokens_in_windows: list[list[TokenLabelPair]]
    ) -> list[TokenLabelPair]:
        """ window毎の予測結果を、連続した一文内の予測結果に変換する. """
        window_stride = self.window_stride

        def _merge_label_pair(left: BIOTag, right: BIOTag) -> BIOTag:
            """I>B>Oの順序関係の下でペアのmaxをとる"""
            if left == right:
                return left
            elif left.bio == BIO.I or right.bio == BIO.I:
                return left if left.bio == BIO.I else right
            elif left.bio == BIO.B or right.bio == BIO.B:
                return left if left.bio == BIO.B else right
            else:
                return BIOTag.from_values(BIO.O, TagType.Other)

        tokens_in_sentence: List[TokenLabelPair] = []
        if len(tokens_in_windows) == 1:
            tokens_in_sentence = tokens_in_windows[0]
        elif len(tokens_in_windows) > 1:
            # windowが複数ある場合、strideぶん被っているwindow境界のラベルをマージする
            startpos2token: Dict[int, TokenLabelPair] = {}
            for prev_w, next_w in pairwise(tokens_in_windows):
                #     print(prev_w[-window_stride:])
                #     print(next_w[:window_stride])
                for token in prev_w:
                    if token.start_pos not in startpos2token:
                        startpos2token[token.start_pos] = token

                for prev_token, next_token in zip(
                    prev_w[-window_stride:], next_w[:window_stride]
                ):
                    start_pos = prev_token.start_pos
                    merged_label = _merge_label_pair(prev_token.label, next_token.label)
                    startpos2token[start_pos] = TokenLabelPair.build(
                        prev_token.text, start_pos, merged_label
                    )
            for token in tokens_in_windows[-1]:
                if token.start_pos not in startpos2token:
                    startpos2token[token.start_pos] = token

            tokens_in_sentence = [
                v for _, v in sorted(startpos2token.items(), key=lambda x: x[0])
            ]

        return tokens_in_sentence

    def decode(
        self,
        sentence_text: str,
        offset_mapping: list[list[tuple[int, int]]],
        outputs: list[TokenClassifierOutput],
    ) -> list[TokenLabelPair]:
        # decode logits into label ids
        batch_label_ids = [
            torch.argmax(out.logits, axis=2).detach().numpy().tolist()
            for out in outputs
        ]
        # unbatch & restore label text
        labels_list = [
            [self.id2label[li] for li in label_ids]
            for label_ids in chain.from_iterable(batch_label_ids)
        ]
        # align with token text, skipping special characters
        # NOTE: Token := token文字列およびその位置とBIOタグのコンテナ
        tokens_in_windows = [
            [
                TokenLabelPair.build(sentence_text[start:end], start, label)
                for (start, end), label in zip(
                    spans[1:-1], labels[1:-1]
                )  # skip [CLS] and [SEP] (予測が-100である保証はない)
            ]
            for spans, labels in zip(offset_mapping, labels_list)
        ]
        tokens_in_sentence = self.unwindow(tokens_in_windows)
        return tokens_in_sentence


class Predictor:
    tokenizer: AutoTokenizer
    model: AutoModelForTokenClassification
    batch_size: int
    max_length: int

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForTokenClassification,
        batch_size: int,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length

    def predict(self, dataset: list[dict]) -> list[TokenClassifierOutput]:
        """ Dataset(Tensor) から Dataloader を構成し、数値予測を行う. """

        # 予測なのでシャッフルなしのサンプラーを使用
        sampler = SequentialSampler(dataset)
        # DataCollatorWithPadding:
        # - 各バッチサンプルに対して tokenizer.pad() が呼ばれ、torch.Tensorが返される
        # - バッチ内のラベルは削除される (See. DataCollatorForTokenClassification)
        data_collator = DataCollatorWithPadding(
            self.tokenizer, padding=True, max_length=self.max_length
        )
        dl = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=data_collator,
            pin_memory=True,
        )

        outputs = [self.model(**b) for b in dl]

        return outputs


class TrfNERFast:
    # Transformersの吐くラベル -> BIOTag
    label2bio_tag = {
        "LABEL_0": BIOTag.from_values(BIO.O, TagType.Other),
        "LABEL_1": BIOTag.from_values(BIO.B, TagType.Target),
        "LABEL_2": BIOTag.from_values(BIO.I, TagType.Target),
    }

    def __init__(
        self, model_path: str, max_length: int, window_stride: int, batch_size: int
    ):
        """FastTokenizer対応のNERモデル"""
        with open(model_path, "rb") as fp:
            model_dict = pickle.load(fp)
        self.tokenizer = model_dict["tokenizer"]
        assert self.tokenizer.is_fast
        self.model = model_dict["model"]
        # for label decoding
        id2label: dict[int, str] = self.model.config.id2label
        id2bio_tag = {i: self.label2bio_tag[l] for i, l in id2label.items()}

        # pipeline
        self.encoder = FastEncoder(self.tokenizer, max_length, window_stride)
        self.predictor = Predictor(self.tokenizer, self.model, batch_size, max_length)
        self.decoder = Decoder(id2bio_tag, window_stride)

    @staticmethod
    def pickle_bert_model(model_dir: str, model_out_path: str):
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        assert tokenizer.is_fast
        model_dict = {"tokenizer": tokenizer, "model": model}
        with open(model_out_path, "wb") as fp:
            pickle.dump(model_dict, fp)

    def predict(
        self,
        sentence_text: str,
    ) -> List[TokenLabelPair]:
        """長文のtokenize付きNER:
        - 長文でもstride付きwindowに区切り(tokenizer)、バッチで予測する(dataloader)
        - windowの境界については予測結果をマージする
        """
        dataset, offset_mapping = self.encoder.encode(sentence_text)
        outputs = self.predictor.predict(dataset)
        tokens_in_sentence = self.decoder.decode(sentence_text, offset_mapping, outputs)
        return tokens_in_sentence
