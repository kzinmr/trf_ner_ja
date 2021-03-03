from dataclasses import dataclass
from enum import Enum
from typing import List

IntList = List[int]
IntListList = List[IntList]
StrList = List[str]
StrListList = List[StrList]
PAD_TOKEN_LABEL_ID = -100
PAD_TOKEN = "[PAD]"


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class SpanAnnotation:
    start: int
    end: int
    label: str


@dataclass
class StringSpanExample:
    guid: str
    content: str
    annotations: List[SpanAnnotation]


@dataclass
class TokenClassificationExample:
    guid: str
    words: StrList
    labels: StrList


@dataclass
class InputFeatures:
    input_ids: IntList
    attention_mask: IntList
    label_ids: IntList


@dataclass
class CNNFeatureExample:
    words: StrList
    features: StrList
    characters: StrListList
    labels: StrList


@dataclass
class CNNInputFeature:
    word_ids: IntList
    feature_ids: IntList
    character_ids: IntListList
    label_ids: IntList
