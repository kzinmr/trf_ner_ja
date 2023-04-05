from dataclasses import dataclass


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
