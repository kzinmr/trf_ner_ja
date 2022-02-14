import os
import pickle
from dataclasses import dataclass
from itertools import chain, tee
from typing import Dict, List, Tuple

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
    end: int


@dataclass
class TokenLabelPair:
    token: Token
    label: str

    @staticmethod
    def build(text: str, start_pos: int, label: str):
        end_pos = start_pos + len(text)
        return TokenLabelPair(Token(text, start_pos, end_pos), label)

    @property
    def start_pos(self) -> int:
        return self.token.start

    @property
    def end_pos(self) -> int:
        return self.token.end


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
    id2label: Dict[int, str]
    window_stride: int

    def __init__(self, id2label: Dict[int, str], window_stride: int):
        self.id2label = id2label
        self.window_stride = window_stride

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
        offset_mapping: List[List[Tuple[int, int]]],
        outputs: List,
    ) -> List[TokenLabelPair]:
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
        self.decoder = Decoder(id2label, window_stride)

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
    print(tokens_in_sentence)
    for t_l in tokens_in_sentence:
        print(t_l.token.text, t_l.label)
