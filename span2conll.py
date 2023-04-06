import json
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from data import ChunkSpan, Token, TokenLabelPair


def bio_sorter(x: str):
    if x.startswith("O"):
        return "1"
    else:
        return "-".join(x.split("-")[::-1])

def make_batch(
    data: list[dict[str, int | list[str]]], batch_size: int
) -> list[dict[str, list[int] | list[list[str]]]]:
    n_data = len(data)
    batched_data = []
    for i in range(0, n_data, batch_size):
        batch = data[i : i + batch_size]
        batched_data.append(
            {
                "id": [d["id"] for d in batch],
                "tokens": [d["tokens"] for d in batch],
                "labels": [d["labels"] for d in batch],
                "ner_tags": [d["ner_tags"] for d in batch],
            }
        )
    return batched_data


class WordTokenizerWithAlignment(metaclass=ABCMeta):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def tokenize_with_alignment(self, text: str) -> list[Token]:
        pass


class Span2WordLabelConverter:
    """span形式のデータを(re)tokenizeしてトークン-ラベルペアを得るためのクラス"""

    def __init__(
        self,
        tokenizer: WordTokenizerWithAlignment,
        max_length: int = 128,
        window_stride: int = 32,
    ):
        self.tokenizer: WordTokenizerWithAlignment = tokenizer
        self.max_length: int = max_length
        self.window_stride: int = window_stride

    def window_by_tokens(self, tokens: list) -> list[list]:
        """一文のデータが長い場合にストライド付き固定長分割を施す処理."""
        max_length = self.max_length - 2  # [CLS]と[SEP]を考慮
        window_stride = self.window_stride

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

    @staticmethod
    def _get_chunk_span(
        query_span: tuple[int, int], superspans: list[tuple[int, int]]
    ) -> tuple[int, int] | None:
        """トークンを包摂するチャンクについて、トークンの文字列スパンを包摂するチャンクのスパンを返す.
        NOTE: 一つのチャンクはトークン境界を跨がないと想定.
        """
        for superspan in superspans:
            if query_span[0] >= superspan[0] and query_span[1] <= superspan[1]:
                return superspan
        return None

    @staticmethod
    def _get_labels_per_tokens(
        spans_of_tokens: list[tuple[int, int]],
        tokenspan2chunkspan: dict[tuple[int, int], tuple[int, int]],
        chunkspan2tagtype: dict[tuple[int, int], str],
    ) -> list[str]:
        """トークン列に対応するラベル列をトークンスパンからラベルへのマップを基に構成"""
        label = "O"
        token_labels: list[str] = []
        prev_span = (0, 0)
        for token_span in spans_of_tokens:
            if token_span in tokenspan2chunkspan:
                chunkspan = tokenspan2chunkspan[token_span]
                tagtype = chunkspan2tagtype[chunkspan]
                if label == "O":
                    label = f"B-{tagtype}"
                elif label.startswith("I"):
                    if prev_span == chunkspan:
                        label = f"I-{tagtype}"
                    else:  # 同じtagtypeのチャンクが隣接
                        label = f"B-{tagtype}"
                else:
                    label = f"I-{tagtype}"
                prev_span = chunkspan
            else:
                label = "O"

            token_labels.append(label)
        return token_labels

    @classmethod
    def get_token_labels_from_spans(
        cls, tokens: list[Token], spans_of_chunks: list[ChunkSpan]
    ) -> list[TokenLabelPair]:
        """文字列スパンとトークンスパンから、トークン-ラベルペアを得る."""
        spans_of_tokens = [(token.start, token.end) for token in tokens]
        # トークンの文字列スパンからトークンを包摂するチャンクへのマップを構成.
        _span_tuples = [(span.start, span.end) for span in spans_of_chunks]
        _span2label = {(span.start, span.end): span.label for span in spans_of_chunks}
        tokenspan2chunkspan: dict[tuple[int, int], tuple[int, int]] = {}
        for original_token_span in spans_of_tokens:
            chunk_span = cls._get_chunk_span(original_token_span, _span_tuples)
            if chunk_span is not None:
                tokenspan2chunkspan[original_token_span] = chunk_span
        # トークン列に対応するラベル列を、トークンスパンをキーとするマップを基に構成.
        labels_per_tokens = cls._get_labels_per_tokens(
            spans_of_tokens, tokenspan2chunkspan, _span2label
        )
        token_labels = [
            TokenLabelPair(token.text, label)
            for token, label in zip(tokens, labels_per_tokens)
        ]
        return token_labels

    def _convert(
        self, sentence_text: str, spans: list[ChunkSpan]
    ) -> list[list[TokenLabelPair]]:
        """文字列と文字列スパンから、トークン列とトークン-ラベルペアを得る. ウィンドウ分割も行う.
        変換例.
        - text: "太郎の家", [ChunkSpan]: [(0, 2, "PERSON")]
        - [TokenLabelPair]: [("太郎", "B-PERSON"), ("の", "O"), ("家", "O")]
        """
        # 単語から窓内文字列への対応をとりながら単語分かち書き
        _tokens = self.tokenizer.tokenize_with_alignment(sentence_text)
        # 単語分かち書きした結果を元に、文字列スパンからトークンスパンへのマッピングを構成
        token_labels = self.get_token_labels_from_spans(_tokens, spans)
        # 長文のメモリ溢れ対策として、ストライド付きウィンドウ分割を行う.
        token_labels_windows = self.window_by_tokens(token_labels)
        return token_labels_windows

    def convert(
        self, filename_jsonl: str, export: bool = True
    ) -> list[list[TokenLabelPair]]:
        """文字列位置スパンで記録されたデータセットをトークナイズし、
        トークン-ラベルのペアからなるConll2003-likeな形式に変換する.
        メモリ溢れ防止のためのデータ分割処理もここで行う.
        """
        with open(filename_jsonl) as fp:
            lines = [json.loads(line) for line in fp if line.strip()]
            text_spans = [
                (
                    d["text"],
                    [
                        ChunkSpan(sp["start"], sp["end"], sp["label"])
                        for sp in d["spans"]
                    ],
                )
                for d in lines
            ]
        dataset = [
            token_label
            for text, spans in text_spans
            for token_label in self._convert(text, spans)
        ]
        if export:
            outpath = filename_jsonl.replace(".jsonl", ".conll")
            self.export_conll_format(dataset, outpath)
        return dataset

    @staticmethod
    def export_conll_format(
        dataset: list[list[TokenLabelPair]], filepath: str, delimiter: str = "\t"
    ):
        """トークン-ラベルペアのデータをConll2003-likeな形式で出力する."""
        sentences = []
        for token_labels in dataset:
            sentence = "\n".join(delimiter.join((tok_lb.token, tok_lb.label)) for tok_lb in token_labels)
            sentences.append(sentence)
        with open(filepath, "wt") as fp:
            _data = "\n\n".join(sentences)
            fp.write(_data)
