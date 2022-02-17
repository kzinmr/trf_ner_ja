import gc
import os
import pickle
from dataclasses import dataclass
from itertools import chain, tee
from typing import Dict, List, Optional

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


@dataclass
class Token:
    text: str
    start: int
    end: Optional[int] = None


@dataclass
class TokenLabelPair:
    token: Token
    label: str
    word_text: Optional[str] = None
    word_start_pos: Optional[int] = None

    @staticmethod
    def build(
        text: str,
        start_pos: int,
        label: str,
        word_text: Optional[str] = None,
        word_start_pos: Optional[int] = None,
    ):
        # NOTE: tokenによってはend位置計算ができないことがある
        end_pos = None
        if text != "[UNK]" and not text.startswith("##"):
            end_pos = start_pos + len(text)
        return TokenLabelPair(
            Token(text, start_pos, end_pos), label, word_text, word_start_pos
        )

    @property
    def text(self) -> str:
        return self.token.text

    @property
    def start_pos(self) -> int:
        return self.token.start

    @property
    def end_pos(self) -> Optional[int]:
        # NOTE: tokenによってはend位置が信頼できないおそれがある
        if (
            self.token.text != "[UNK]"
            and not self.token.text.startswith("##")
            and self.token.end
        ):
            return self.token.end
        else:
            return None

    @property
    def word_end_pos(self) -> Optional[int]:
        if self.word_start_pos and self.word_text:
            return self.word_start_pos + len(self.word_text)
        else:
            return None


class FastEncoder:
    tokenizer: PreTrainedTokenizerFast
    max_length: int
    window_stride: int

    def __init__(self, tokenizer, max_length, window_stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_stride = window_stride

    def encode(self, sentence_text: str):
        """FastTokenizerによりテキストをトークナイズ・Tensor化する。"""
        assert self.window_stride < self.max_length - 2
        # NOTE: デコーディングのために、元文字列を解析しつつoffset_mappingを保持.
        # NOTE: 長文のメモリ溢れ対策として、ストライド付きウィンドウ分割を行う.
        enc: BatchEncoding = self.tokenizer(
            sentence_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.window_stride,  # fast-only
            return_overflowing_tokens=True,  # fast-only
            return_offsets_mapping=True,  # fast-only
            add_special_tokens=True,
            return_token_type_ids=False,
            return_length=True,
        )
        # NOTE: [CLS] and [SEP] は含む
        dataset = [
            {"input_ids": encoding.ids, "attention_mask": encoding.attention_mask}
            for encoding in enc.encodings
        ]
        # NOTE: [CLS] and [SEP] は含まない
        # 単語の情報は予測時には冗長だが分析用途のため登録しておく
        tokens_in_windows = []
        for _enc, spans in zip(enc.encodings, enc.offset_mapping):
            token_labels = []
            # skipping [CLS] and [SEP]
            for wid, (start, end) in zip(_enc.word_ids[1:-1], spans[1:-1]):
                word_start_pos, word_end_pos = _enc.word_to_chars(wid)
                word_text = sentence_text[word_start_pos:word_end_pos]
                token_labels.append(
                    TokenLabelPair.build(
                        sentence_text[start:end],
                        start,
                        "O",
                        word_start_pos=word_start_pos,
                        word_text=word_text,
                    )
                )
            tokens_in_windows.append(token_labels)

        return dataset, tokens_in_windows


class FastDecoder:
    id2label: Dict[int, str]
    window_stride: int

    def __init__(self, id2label: Dict[int, str], window_stride: int):
        self.id2label = id2label
        self.window_stride = window_stride

    def update_labels(
        self,
        tokens_in_windows: List[List[TokenLabelPair]],
        labels_list: List[List[str]],
    ) -> List[List[TokenLabelPair]]:
        """ トークン-ラベルペアリストに対して、ラベル属性に予測結果ラベルを書き込む処理 """
        return [
            [
                TokenLabelPair.build(token_label.text, token_label.start_pos, label)
                for token_label, label in zip(token_labels, labels)
            ]
            for token_labels, labels in zip(tokens_in_windows, labels_list)
        ]

    def unwindow(
        self, tokens_in_windows: List[List[TokenLabelPair]]
    ) -> List[TokenLabelPair]:
        """ window毎の予測結果を、連続した一文内の予測結果に変換する. """
        window_stride = self.window_stride

        def _merge_label_pair(left: str, right: str) -> str:
            """I>B>Oの順序関係の下でペアのmaxをとる"""
            left_bio = left.split("-")[0]
            right_bio = right.split("-")[0]
            if left == right:
                return left
            elif left_bio == "I" or right_bio == "I":
                return left if left_bio == "I" else right
            elif left_bio == "B" or right_bio == "B":
                return left if left_bio == "B" else right
            else:
                return "O"

        tokens_in_sentence: List[TokenLabelPair] = []
        if len(tokens_in_windows) == 1:
            tokens_in_sentence = tokens_in_windows[0]
        elif len(tokens_in_windows) > 1:
            # windowが複数ある場合、strideぶん被っているwindow境界のラベルをマージする
            startpos2token: Dict[int, TokenLabelPair] = {}
            for prev_w, next_w in pairwise(tokens_in_windows):
                #     print(prev_w[-window_stride:])
                #     print(next_w[:window_stride])
                for token_label in prev_w:
                    if token_label.token.start not in startpos2token:
                        startpos2token[token_label.token.start] = token_label

                for prev_token_label, next_token_label in zip(
                    prev_w[-window_stride:], next_w[:window_stride]
                ):
                    start_pos = prev_token_label.token.start
                    merged_label = _merge_label_pair(
                        prev_token_label.label, next_token_label.label
                    )
                    startpos2token[start_pos] = TokenLabelPair.build(
                        prev_token_label.token.text, start_pos, merged_label
                    )
            for token_label in tokens_in_windows[-1]:
                if token_label.token.start not in startpos2token:
                    startpos2token[token_label.token.start] = token_label

            tokens_in_sentence = [
                v for _, v in sorted(startpos2token.items(), key=lambda x: x[0])
            ]

        return tokens_in_sentence

    def decode(
        self,
        tokens_in_windows: List[List[TokenLabelPair]],
        outputs: List,
    ) -> List[TokenLabelPair]:
        # decode logits into label ids
        batch_label_ids = []
        for out in outputs:
            label_ids = torch.argmax(out.logits, axis=2).detach().numpy().tolist()
            batch_label_ids.append(label_ids)
            # NOTE: memory error対策
            del out
            gc.collect()
        # unbatch & restore label text
        # skipping special characters [CLS] and [SEP]
        labels_list = [
            [self.id2label[li] for li in label_ids][1:-1]
            for label_ids in chain.from_iterable(batch_label_ids)
        ]
        # update by predicted labels in each tokens
        tokens_in_windows = self.update_labels(tokens_in_windows, labels_list)

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

    def predict(self, dataset: List[dict]) -> List:
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
    def __init__(
        self,
        model_path: str,
        max_length: int = 128,
        window_stride: int = 8,
        batch_size: int = 32,
    ):
        """FastTokenizer対応のNERモデル"""
        with open(model_path, "rb") as fp:
            model_dict = pickle.load(fp)
        self.tokenizer = model_dict["tokenizer"]
        assert self.tokenizer.is_fast
        self.model = model_dict["model"]
        # for label decoding
        id2label: Dict[int, str] = self.model.config.id2label

        # pipeline
        self.encoder = FastEncoder(self.tokenizer, max_length, window_stride)
        self.predictor = Predictor(self.tokenizer, self.model, batch_size, max_length)
        self.decoder = FastDecoder(id2label, window_stride)

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
        dataset, tokens_in_windows = self.encoder.encode(sentence_text)
        outputs = self.predictor.predict(dataset)
        tokens_in_sentence = self.decoder.decode(tokens_in_windows, outputs)
        return tokens_in_sentence


if __name__ == "__main__":

    data_dir = "/app/workspace/"
    pkl_path = os.path.join(data_dir, "predictor_en.pkl")
    predictor = TrfNERFast(pkl_path)
    tokens = [
        "The",
        "European",
        "Commission",
        "said",
        "on",
        "Thursday",
        "it",
        "disagreed",
        "with",
        "German",
        "advice",
        "to",
        "consumers",
        "to",
        "shun",
        "British",
        "lamb",
        "until",
        "scientists",
        "determine",
        "whether",
        "mad",
        "cow",
        "disease",
        "can",
        "be",
        "transmitted",
        "to",
        "sheep",
        ".",
    ]
    sent = " ".join(tokens)
    tokens_in_sentence = predictor.predict(sent)
    for t_l in tokens_in_sentence:
        print(t_l)
    for t_l in tokens_in_sentence:
        print(t_l.token.text, t_l.label)
