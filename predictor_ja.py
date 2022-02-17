from collections import defaultdict
import os
import pickle
from dataclasses import dataclass
from itertools import chain, tee
from typing import Dict, List, Optional

import fugashi
import unidic_lite
import tokenizations
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)


# from predictor_en import Decoder, Predictor, Token, TokenLabelPair


os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    def is_unreliable_token_pos(self) -> bool:
        # サブワードtokenや[UNK]tokenは位置スパンを登録できない
        return self.start_pos >= 0

    @property
    def end_pos(self) -> Optional[int]:
        # サブワードtokenや[UNK]tokenはend位置が計算できない
        if not self.is_unreliable_token_pos:
            return self.token.end
        else:
            return None

    @property
    def word_end_pos(self) -> Optional[int]:
        if self.word_start_pos and self.word_text:
            return self.word_start_pos + len(self.word_text)
        else:
            return None


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class MeCabPreTokenizer:
    def __init__(self):
        # mecab_option = "-Owakati"
        # self.wakati = MeCab.Tagger(mecab_option)
        dic_dir = unidic_lite.DICDIR
        mecabrc = os.path.join(dic_dir, "mecabrc")
        mecab_option = "-d {} -r {} ".format(dic_dir, mecabrc)
        self.mecab = fugashi.GenericTagger(mecab_option)

    def tokenize(self, text: str) -> List[str]:
        # return self.mecab.parse(text).strip().split(" ")
        return self.mecab(text)

    def tokenize_with_alignment(self, text: str) -> List[Token]:
        token_surfaces = [word.surface for word in self.mecab(text)]
        tokens = []
        _cursor = 0
        for token in token_surfaces:
            start = text.index(token, _cursor)
            end = start + len(token)
            tok = Token(token, start, end)
            tokens.append(tok)
            _cursor = tok.end

        return tokens

    def window_by_tokens(
        self,
        sentence: str,
        max_length: int = 128,
        window_stride: int = 8,
    ) -> List[List[Token]]:
        """一文のデータが長い場合にストライド付き固定長分割を施す処理."""
        tokens = self.tokenize_with_alignment(sentence)

        seq_length = len(tokens)
        if seq_length <= max_length:
            return [tokens]
        else:
            sentence_windows = []
            # max_length を窓幅, window_strideをストライド幅とする固定長に分割
            for start in range(0, seq_length, seq_length - window_stride):
                end = min(start + max_length, seq_length)
                if end - start == max_length:
                    window = tokens[start:end]
                    sentence_windows.append(window)
                elif end - start < max_length:  # 末端に達したら抜ける
                    window = tokens[start:end]
                    sentence_windows.append(window)
                    break
            return sentence_windows


class SlowEncoder:
    tokenizer: PreTrainedTokenizer
    max_length: int
    window_stride: int

    def __init__(self, tokenizer, max_length, window_stride):
        self.pretokenizer = MeCabPreTokenizer()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_stride = window_stride

    @staticmethod
    def align_tokens_with_words(words: List[str], tokens: List[str]) -> List[int]:
        # tokens側は[CLS],[SEP]含まない想定
        w2t, t2w = tokenizations.get_alignments(words, tokens)
        word_ids = []
        prev_wid = 0
        max_wid = max([wids[0] if len(wids) > 0 else 0 for wids in t2w])
        for token, wids in zip(tokens, t2w):
            if len(wids) > 0:
                word_ids.append(wids[0])
                prev_wid = wids[0]
            elif token == "[UNK]":
                # [UNK] のケースはなるべく近いidで内挿する
                cur = min(max_wid, prev_wid + 1)
                word_ids.append(cur)
                prev_wid += 1
            else:  # '[PAD]'
                word_ids.append(None)
        return word_ids

    def encode(self, sentence_text: str):
        """SlowTokenizerによりテキストをトークナイズ・Tensor化する。"""
        assert self.window_stride < self.max_length - 2
        # NOTE: デコーディングのために、トークン→元文字列へのスパンを保持したい、が
        # 実際得られるのは、単語→window文字列(PreTokenizer)とトークン→単語(SlowTokenizer)の対応のみ
        # 結果、処理が複雑化してしまう
        # NOTE: 長文のメモリ溢れ対策として、ストライド付きウィンドウ分割を行う.
        words_in_windows = self.pretokenizer.window_by_tokens(
            sentence_text, self.max_length - 2, self.window_stride
        )
        word_str_in_windows = [[tok.text for tok in ws] for ws in words_in_windows]
        # PreTokenizerでwindow分割および単語→window文字列対応をとるため、窓ごとのtokenizer適用となる.
        encodings = [
            self.tokenizer(
                words_in_window,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                is_split_into_words=True,  # 分かち書き済み
                # stride=self.window_stride,  # fast-only
                # return_overflowing_tokens=True,  # fast-only
                # return_offsets_mapping=True,  # fast-only
                add_special_tokens=True,
                return_token_type_ids=False,
                return_length=True,
            )
            for words_in_window in word_str_in_windows
        ]
        # NOTE: [CLS] and [SEP] は含む (予測時に必要(ほんとか?))
        dataset = [
            {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
            for enc in encodings
        ]
        # SlowTokenizerの場合、トークンの文内位置スパンが計算できないため複雑な処理が挟まる.
        # - token.start_pos/end_posは信頼できない形で使う必要あるため、信頼できる単語単位を登録する
        #   - トークン位置スパンが計算できない理由としては、[UNK]トークンの存在やサブワードの存在による.
        # - また、窓ごとの位置スパンでなく、元の文内の位置スパンを計算する必要がある点も注意.
        # - FastTokenizerでは、トークナイザ内部の機構で窓分割&元文内位置スパンを保持した計算がされる.
        tokens_batches: List[List[str]] = [
            self.tokenizer.convert_ids_to_tokens(enc["input_ids"]) for enc in encodings
        ]
        # NOTE: [CLS] and [SEP] は含まない；[UNK] は含みうる
        tokens_in_windows = []
        for ix_w, (words, tokens) in enumerate(zip(words_in_windows, tokens_batches)):
            _tokens = tokens[1:-1]  # [CLS],[SEP]を省く
            # 信頼できる単語単位を登録するために、トークン-単語のアラインメントをとる
            words_str = [w.text for w in words]
            word_ids = self.align_tokens_with_words(words_str, _tokens)
            # NOTE: window内->元文内の位置スパン、トークンでなく単語単位の位置スパンを登録
            window_offset = ix_w * (self.max_length - 2 - self.window_stride)
            words_start_pos_sentence = [w.start + window_offset for w in words]

            token_word_labels = []
            for tok, wid in zip(_tokens, word_ids):
                word_text = words_str[wid]
                start_pos_sentence = words_start_pos_sentence[wid]
                token_word_labels.append(
                    TokenLabelPair.build(
                        text=tok,
                        start_pos=-1,  # Unreliable
                        label="O",
                        word_text=word_text,
                        word_start_pos=start_pos_sentence,
                    )
                )
            tokens_in_windows.append(token_word_labels)

        return dataset, tokens_in_windows


class SlowDecoder:
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

    def update_labels(
        self,
        tokens_in_windows: List[List[TokenLabelPair]],
        labels_list: List[List[str]],
    ) -> List[List[TokenLabelPair]]:
        """ トークン-ラベルペアリストに対して、ラベル属性に予測結果ラベルを書き込む処理 """

        def _bio_sorter(x: str):
            if x.startswith("O"):
                return "1"
            else:
                return "-".join(x.split("-")[::-1])

        def _merge_bio_labels(labels_in_word: List[str]) -> str:
            labels_sorted = sorted(labels_in_word, key=_bio_sorter)
            if all(l == "O" for l in labels_sorted):
                return "O"
            else:
                return [l for l in labels_sorted if l != "O"][0]

        # SlowTokenizerの場合、トークンの文内位置スパンが計算できないため複雑な処理が挟まる.
        # - unwindow処理でtoken.start_pos/end_posを信頼できる形で使う必要あるため、信頼できる単語単位に戻す
        # - 位置スパンが計算できない理由としては、[UNK]トークンの存在やサブワードの存在による.
        # - FastTokenizerでは、トークナイザ内部の機構で文内位置スパンを保持した計算がされるため、この箇所は不要

        word_labels_in_windows = []
        for token_labels, labels in zip(tokens_in_windows, labels_list):
            # 単語サブワード内のBIOラベルを、単語単位B-XXX>I-XXX>Oの順で代表させる処理
            _word_pos2word_labels = defaultdict(list)
            for token_label, label in zip(token_labels, labels):
                _word_pos2word_labels[token_label.word_start_pos].append(
                    (token_label.word_text, label)
                )
            _word_pos2word_label = {
                wpos: (
                    w_text_labels[0][0],
                    _merge_bio_labels([l for _, l in w_text_labels]),
                )
                for wpos, (w_text_labels) in _word_pos2word_labels.items()
            }
            # 単語単位の予測結果に書き換える
            word_labels = [
                TokenLabelPair.build(
                    word_text,
                    word_start_pos,
                    wlabel,
                    word_start_pos=word_start_pos,
                    word_text=word_text,
                )
                for word_start_pos, (word_text, wlabel) in sorted(
                    _word_pos2word_label.items(), key=lambda x: x[0]
                )
            ]
            word_labels_in_windows.append(word_labels)
        return word_labels_in_windows

    def decode(
        self,
        tokens_in_windows: List[List[TokenLabelPair]],
        outputs: List,
    ) -> List[TokenLabelPair]:
        # decode logits into label ids
        batch_label_ids = [
            torch.argmax(out.logits, axis=2).detach().numpy().tolist()
            for out in outputs
        ]
        # unbatch & restore label text
        # skipping special characters [CLS] and [SEP]
        labels_list = [
            [self.id2label[li] for li in label_ids][1:-1]
            for label_ids in chain.from_iterable(batch_label_ids)
        ]
        # update by predicted labels in each words (not tokens)
        tokens_in_windows = self.update_labels(tokens_in_windows, labels_list)
        # window -> sentence
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


class TrfNERSlow:
    def __init__(
        self,
        model_path: str,
        max_length: int = 128,
        window_stride: int = 8,
        batch_size: int = 32,
    ):
        """Non-Fast Tokenizer向けのNERモデル"""
        with open(model_path, "rb") as fp:
            model_dict = pickle.load(fp)
        self.tokenizer = model_dict["tokenizer"]
        assert not self.tokenizer.is_fast
        self.model = model_dict["model"]
        # for label decoding
        id2label: Dict[int, str] = self.model.config.id2label

        # pipeline
        self.encoder = SlowEncoder(self.tokenizer, max_length, window_stride)
        self.predictor = Predictor(self.tokenizer, self.model, batch_size, max_length)
        self.decoder = SlowDecoder(id2label, window_stride)

    @staticmethod
    def pickle_bert_model(model_dir: str, model_out_path: str):
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # NOTE: cl-tohoku BERTはFast Tokenizerの読み込みができない(できても挙動が変になる)
        assert not tokenizer.is_fast
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
    pkl_path = os.path.join(data_dir, "predictor_ja.pkl")
    predictor = TrfNERSlow(pkl_path)

    sent = """特定供給者パワーステーション株式会社（以下「甲」という。）と一般電気事業者又は
特定電気事業者東京電力株式会社（以下「乙」という。）は、電気事業者による再生可能エ
ネルギー電気の調達に関する特別措置法（平成23年法律第108号、その後の改正を含み
以下「再エネ特措法」という。）に定める再生可能エネルギー電気の甲による供給及び乙によ
る調達並びに甲の発電設備と乙の電力系統との接続等に関して、次のとおり契約（以下「本
契約」という。）を締結する。"""
    tokens_in_sentence = predictor.predict(sent)
    for t_l in tokens_in_sentence:
        print(t_l)
    for t_l in tokens_in_sentence:
        print(t_l.token.text, t_l.label)
