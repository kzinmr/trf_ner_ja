import json
import os
from dataclasses import dataclass
from typing import Optional, Union

import fugashi
import unidic_lite
from sklearn.model_selection import train_test_split


@dataclass
class Token:
    text: str
    start: int
    end: int


@dataclass
class ChunkSpan:
    start: int
    end: int
    label: str


@dataclass
class TokenLabelPair:
    token: str
    label: str


class MeCabTokenizer:
    def __init__(self):
        # mecab_option = "-Owakati"
        # self.wakati = MeCab.Tagger(mecab_option)
        dic_dir = unidic_lite.DICDIR
        mecabrc = os.path.join(dic_dir, "mecabrc")
        mecab_option = "-d {} -r {} ".format(dic_dir, mecabrc)
        self.mecab = fugashi.GenericTagger(mecab_option)

    def tokenize(self, text: str) -> list[str]:
        # return self.mecab.parse(text).strip().split(" ")
        return self.mecab(text)

    def tokenize_with_alignment(self, text: str) -> list[Token]:
        token_surfaces = [word.surface for word in self.mecab(text)]
        tokens = []
        _cursor = 0
        for token in token_surfaces:
            start = text.index(token, _cursor)
            end = start + len(token)
            tokens.append(Token(token, start, end))
            _cursor = end

        return tokens


class Span2TokenConverter:

    def __init__(self):
        # tokenizer
        self.tokenizer = MeCabTokenizer()

    @staticmethod
    def _get_chunk_span(
        query_span: tuple[int, int], superspans: list[tuple[int, int]]
    ) -> Optional[tuple[int, int]]:
        """トークンを包摂するチャンクについて、トークンの文字列スパンを包摂するチャンクのスパンを返す.
        NOTE: 一つのチャンクはトークン境界を跨がないと想定.
        """
        for superspan in superspans:
            if query_span[0] >= superspan[0] and query_span[1] <= superspan[1]:
                return superspan
        return None

    @classmethod
    def _get_token2label_map(
        cls, spans_of_tokens: list[tuple[int, int]], spans_of_chunks: list[ChunkSpan]
    ) -> dict[tuple[int, int], str]:
        """トークンの文字列スパンから、トークンを包摂するチャンクのラベルへのマップを構成."""
        span_tuples = [(span.start, span.end) for span in spans_of_chunks]
        _span2label = {(span.start, span.end): span.label for span in spans_of_chunks}
        tokenspan2tagtype: dict[tuple[int, int], str] = {}
        for original_token_span in spans_of_tokens:
            chunk_span = cls._get_chunk_span(original_token_span, span_tuples)
            if chunk_span is not None:
                tokenspan2tagtype[original_token_span] = _span2label[chunk_span]
        return tokenspan2tagtype

    @staticmethod
    def _get_labels_per_tokens(spans_of_tokens: list[tuple[int, int]], tokenspan2tagtype: dict[tuple[int, int], str]) -> list[str]:
        """トークン列に対応するラベル列をトークンスパンからラベルへのマップを基に構成"""
        label = "O"
        token_labels: list[str] = []
        for token_span in spans_of_tokens:
            if token_span in tokenspan2tagtype:
                tagtype = tokenspan2tagtype[token_span]
                if label == "O":
                    label = f"B-{tagtype}"
                else:
                    label = f"I-{tagtype}"
            else:
                label = "O"

            token_labels.append(label)
        return token_labels

    @classmethod
    def get_token_labels(
        cls, tokens: list[Token], spans_of_chunks: list[ChunkSpan]
    ) -> list[TokenLabelPair]:
        """ 文字列スパンとトークンスパンから、トークン-ラベルペアを得る.
        """
        spans_of_tokens = [(token.start, token.end) for token in tokens]
        tokenspan2label = cls._get_token2label_map(spans_of_tokens, spans_of_chunks)
        labels_per_tokens = cls._get_labels_per_tokens(spans_of_tokens, tokenspan2label)
        token_labels = [
            TokenLabelPair(token.text, label) for token, label in zip(tokens, labels_per_tokens)
        ]
        return token_labels

    def get_token_label_pairs(
        self, text: str, spans_of_chunks: list[ChunkSpan]
    ) -> list[TokenLabelPair]:
        """
        文字列チャンクスパン+元テキスト -> トークン-BIOラベルのペアを返すパイプライン
        - 文字列チャンクスパン:	[(0, 2, "PERSON")], 元テキスト: "太郎の家"
        - トークン-BIOラベル: [("太郎", "B-PERSON"), ("の", "O"), ("家", "O")]
        """
        # 1. tokenize
        tokens = self.tokenizer.tokenize_with_alignment(text)
        # 2. get token-label pairs from spans of chunks and spans of tokens
        token_labels = self.get_token_labels(tokens, spans_of_chunks)
        return token_labels


def window_token_labels(
    sentence,
    max_length: int = 128,
    window_stride: int = 5,
):
    """ 一文のデータが長い場合にストライド付き固定長分割を施す処理.
    """
    seq_length = len(sentence)
    if seq_length <= max_length:
        return [sentence]
    else:
        sentence_windows = []
        # max_length を窓幅, window_strideをストライド幅とする固定長に分割
        for start in range(0, seq_length, window_stride):
            end = min(start + max_length, seq_length)
            if end - start == max_length:
                window = sentence[start: end]
                sentence_windows.append(window)
            elif end - start < max_length:  # 末端に達したら抜ける
                window = sentence[start: end]
                sentence_windows.append(window)
                break
        return sentence_windows


def read_span_dataset(filename_jsonl: str) -> list[list[TokenLabelPair]]:
    """ 文字列位置スパンで記録されたデータセットをトークナイズし、
    トークン-ラベルのペアからなるConll2003-likeな形式に変換する.
    NOTE: メモリ溢れ防止のためのデータ分割処理もここで行う.
    """
    conv = Span2TokenConverter()
    dataset: list[list[TokenLabelPair]] = []
    with open(filename_jsonl) as fp:
        lines = [json.loads(line) for line in fp if line.strip()]
        for data in lines:
            text = data["text"]
            spans = data.get("spans", None)
            if spans:
                spans = [
                    ChunkSpan(sp["start"], sp["end"], sp["label"]) for sp in spans
                ]
                token_labels = conv.get_token_label_pairs(text, spans)
                # window処理をかけて MemoryError を防止
                token_labels_windows = window_token_labels(token_labels)
                dataset.extend(token_labels_windows)
    return dataset


def train_valid_test_split(
    dataset: list,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> tuple[list, list, list]:
    train_valid, test = train_test_split(dataset, test_size=test_ratio)
    rel_valid_ratio = valid_ratio / (1 - test_ratio)
    train, valid = train_test_split(train_valid, test_size=rel_valid_ratio)
    return train, valid, test


def bio_sorter(x: str):
    if x.startswith('O'):
        return '1'
    else:
        return '-'.join(x.split('-')[::-1])


class QuasiDataset:
    train: list[dict[str, Union[int, list[str]]]]
    validation: list[dict[str, Union[int, list[str]]]]
    test: list[dict[str, Union[int, list[str]]]]
    label_list: list[str]
    id2label: dict[int, str]
    label2id: dict[str, int]

    def __init__(
        self,
        dataset: list[list[TokenLabelPair]],
        valid_ratio: float = 0.2,
        test_ratio: float = 0.2
    ):
        label_set = {d.label for ds in dataset for d in ds}
        label_list = sorted(label_set, key=bio_sorter)
        id2label = dict(enumerate(label_list))
        label2id = {l: i for i, l in id2label.items()}

        all_dataset = [
            {
                'id': i,
                'tokens': [t_l.token for t_l in token_labels],
                'labels': [t_l.label for t_l in token_labels],
                'ner_tags': [label2id[t_l.label] for t_l in token_labels]
            }
            for i, token_labels in enumerate(dataset)
        ]
        train, valid, test = train_valid_test_split(all_dataset, valid_ratio, test_ratio)
        self.train = train
        self.validation = valid
        self.test = test
        self.label_list = label_list
        self.id2label = id2label
        self.label2id = label2id

    @staticmethod
    def load_from_span_dataset(
        filepath: str,
        valid_ratio=0.2,
        test_ratio=0.2
    ):
        dataset_all = read_span_dataset(filepath)
        return QuasiDataset(dataset_all, valid_ratio, test_ratio)

    def export_token_label_dataset(
        self, delimiter: str = "\t"
    ):
        """ トークン-ラベルペアのデータをConll2003-likeな形式で出力する.
        """
        def _stringfy_sentences(dataset):
            sentences = []
            for d in dataset:
                token_labels = zip(d['tokens'], d['labels'])
                sentence = '\n'.join(
                    delimiter.join((tok, lb)) for tok, lb in token_labels
                )
                sentences.append(sentence)
            return sentences

        sentences = _stringfy_sentences(self.train)
        with open("train.conll", "wt") as fp:
            _data = "\n".join(sentences)
            fp.write(_data)
        sentences = _stringfy_sentences(self.validation)
        with open("validation.conll", "wt") as fp:
            _data = "\n".join(sentences)
            fp.write(_data)
        sentences = _stringfy_sentences(self.test)
        with open("test.conll", "wt") as fp:
            _data = "\n".join(sentences)
            fp.write(_data)
