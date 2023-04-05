import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass

from data import ChunkSpan, TokenLabelPair


@dataclass
class CoNLLSentence(Iterable[TokenLabelPair]):
    """CoNLL2003風形式のトークン-ラベルデータを読み込むクラス。
    span形式のjsonにエクスポートする用途を想定している。
    """

    token_labels: list[TokenLabelPair]
    chunks: list[ChunkSpan]

    @property
    def text(self) -> str:
        return "".join([tl.token for tl in self.token_labels])

    def __iter__(self):
        for token in self.token_labels:
            yield token

    @staticmethod
    def __chunk_token_labels(
        tokens: list[TokenLabelPair],
    ) -> list[list[TokenLabelPair]]:
        chunks = []
        chunk = []
        for token in tokens:
            if token.label.startswith("B"):
                if chunk:
                    chunks.append(chunk)
                    chunk = []
                chunk = [token]
            elif token.label.startswith("I"):
                chunk.append(token)
            elif chunk:
                chunks.append(chunk)
                chunk = []
        return chunks

    @staticmethod
    def __chunk_span(tokens: list[TokenLabelPair]) -> list[tuple[int, int]]:
        pos = 0
        spans = []
        chunk_spans = []
        for tl in tokens:
            token_len = len(tl.token)
            span = (pos, pos + token_len)
            pos += token_len

            if tl.label.startswith("B"):
                # I->B
                if len(spans) > 0:
                    chunk_spans.append((spans[0][0], spans[-1][1]))
                    spans = []
                spans.append(span)
            elif tl.label.startswith("I"):
                spans.append(span)
            elif len(spans) > 0:
                # B|I -> O
                chunk_spans.append((spans[0][0], spans[-1][1]))
                spans = []

        return chunk_spans

    @classmethod
    def __build_chunks(cls, tokens: list[TokenLabelPair]) -> list[ChunkSpan]:
        _chunks = cls.__chunk_token_labels(tokens)
        _labels = [c_tokens[0].label for c_tokens in _chunks]
        _spans = cls.__chunk_span(tokens)
        return [
            ChunkSpan(
                start=s,
                end=e,
                label=lbl.split("-")[1],
            )
            for lbl, (s, e) in zip(_labels, _spans)
        ]

    @classmethod
    def from_conll(cls, path: str, delimiter: str = "\t"):
        # CoNLL2003 -> list[Sentence]
        sentences = []
        with open(path) as fp:
            for s in fp.read().split("\n\n"):
                tokens: list[TokenLabelPair] = []
                for token in s.split("\n"):
                    line = token.split(delimiter)
                    if len(line) >= 2:
                        token_text = line[0]
                        token_label = line[-1]
                        tlp = TokenLabelPair(token_text, token_label)
                        tokens.append(tlp)
                chunks = cls.__build_chunks(tokens)
                sentences.append(CoNLLSentence(tokens, chunks))
        return sentences

    def export_span_format(self) -> dict:
        # {"text": "", "spans": [{"start":0, "end":1, "label": "PERSON"}]}
        text = self.text
        spans = [
            {"start": c.start, "end": c.end, "label": c.label} for c in self.chunks
        ]
        return {"text": text, "spans": spans}

    def export_json(self) -> str:
        jd = self.export_span_format()
        return json.dumps(jd, ensure_ascii=False)

    def convert_conll_to_span(self, path: str):
        """CoNLL2003形式のデータをspan形式のjsonlに変換する."""
        sentences = self.from_conll(path)
        with open(path.replace(".conll", ".jsonl"), "wt") as fp:
            for s in sentences:
                fp.write(s.export_json())
                fp.write("\n")


if __name__ == "__main__":
    # read conll2003-like format
    path = sys.argv[1]  # train.conll
    CoNLLSentence.convert_conll_to_span(path)
