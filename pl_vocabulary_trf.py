import os
import unicodedata
from itertools import product
from typing import Dict, List, Optional

import MeCab
import textspan
from tokenizers import (
    Encoding,
    NormalizedString,
    PreTokenizedString,
    Tokenizer,
)
from tokenizers.pre_tokenizers import PreTokenizer
from transformers import (
    AutoTokenizer,
    BertTokenizerFast,
    PreTrainedTokenizerFast,
)

from data import *


class LabelTokenAligner:
    """
    Align word-wise BIOLU-labels with subword tokens
    """

    def __init__(self, labels_path: str):
        with open(labels_path, "r") as f:
            labels = [l for l in f.read().splitlines() if l and l != "O"]

        self.labels_to_id = {"O": 0}
        self.ids_to_label = {0: "O"}
        for i, (label, s) in enumerate(product(labels, "BILU"), 1):
            l = f"{s}-{label}"
            self.labels_to_id[l] = i
            self.ids_to_label[i] = l

    @staticmethod
    def get_ids_to_label(labels_path: str, bilou: bool = True) -> Dict[int, str]:
        with open(labels_path, "r") as f:
            labels = [l for l in f.read().splitlines() if l and l != "O"]
        if bilou:
            ids_to_label = {
                i: f"{s}-{label}"
                for i, (label, s) in enumerate(product(labels, "BILU"), 1)
            }
        else:
            ids_to_label = {
                i: f"{s}-{label}"
                for i, (label, s) in enumerate(product(labels, "BI"), 1)
            }
        ids_to_label[0] = "O"
        return ids_to_label

    @staticmethod
    def align_tokens_and_annotations_bilou(
        tokenized: Encoding, annotations: List[SpanAnnotation]
    ) -> StrList:
        """Recover tokenwise BIOLU-labels by using char-token alignment info in Encoding.
        :param tokenized: output of PreTrainedTokenizerFast
        :param annotations: annotations of string span format
        """
        aligned_labels = ["O" for _ in tokenized.tokens]

        for anno in annotations:
            # char_ix -> token_ix for each annotations
            # [1,1,None,2,2,3,] for some 3-tokens that spans 6-characters
            annotation_token_indices = [
                tokenized.char_to_token(char_ix)
                for char_ix in range(anno.start, anno.end)
            ]
            # [1,2,3,]
            annotation_token_ix_uniq = sorted(
                list(filter(None, set(annotation_token_indices)))
            )
            if len(annotation_token_ix_uniq) == 1:
                token_ix = annotation_token_ix_uniq[0]
                prefix = "U"
                aligned_labels[token_ix] = f"{prefix}-{anno.label}"
            else:
                last_token_in_anno_ix = len(annotation_token_ix_uniq) - 1
                for num, token_ix in enumerate(annotation_token_ix_uniq):
                    if num == 0:
                        prefix = "B"
                    elif num == last_token_in_anno_ix:
                        prefix = "L"
                    else:
                        prefix = "I"
                    aligned_labels[token_ix] = f"{prefix}-{anno.label}"

        return aligned_labels

    def align_labels_with_tokens(
        self, tokenized_text: Encoding, annotations: List[SpanAnnotation]
    ) -> IntList:
        # TODO: switch label encoding scheme, align_tokens_and_annotations_bio
        raw_labels = self.align_tokens_and_annotations_bilou(
            tokenized_text, annotations
        )
        return list(map(lambda x: self.labels_to_id.get(x, 0), raw_labels))


class PicklableTagger:
    def __init__(self, mecab_option: str):
        self.option = mecab_option
        self.tagger = MeCab.Tagger(mecab_option)

    def __getstate__(self):
        return {"option": self.option}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __getnewargs__(self):
        return (self.option,)

    def __reduce_ex__(self, proto):
        func = PicklableTagger
        args = self.__getnewargs__()
        state = self.__getstate__()
        listitems = None
        dictitems = None
        rv = (func, args, state, listitems, dictitems)
        return rv

    def __call__(self, text):
        return self.parse(text)

    def parse(self, text):
        return self.tagger.parse(text).rstrip()


class MecabPreTokenizer:
    def __init__(
        self,
        mecab_dict_path: Optional[str] = None,
        space_replacement: Optional[str] = None,
    ):
        """Constructs a MecabPreTokenizer for huggingface tokenizers.
        - space_replacement: Character which is replaced with spaces.
            You might want to use it because MeCab drop spaces by default.
            This can be used to preserve spaces by replacing them with spaces later.
            Special characters like '_' are used sometimes.
        """

        self.space_replacement = space_replacement

        mecab_option = (
            f"-Owakati -d {mecab_dict_path}"
            if mecab_dict_path is not None
            else "-Owakati"
        )
        self.mecab = PicklableTagger(mecab_option)

    def tokenize(self, sequence: str) -> List[str]:
        text = unicodedata.normalize("NFKC", sequence)
        if self.space_replacement:
            text = text.replace(" ", self.space_replacement)
            splits = self.mecab.parse(text).strip().split(" ")
            return [x.replace(self.space_replacement, " ") for x in splits]
        else:
            return self.mecab.parse(text).strip().split(" ")

    def custom_split(
        self, i: int, normalized_string: NormalizedString
    ) -> List[NormalizedString]:
        text = str(normalized_string)
        tokens = self.tokenize(text)
        tokens_spans = textspan.get_original_spans(tokens, text)
        return [
            normalized_string[st:ed]
            for char_spans in tokens_spans
            for st, ed in char_spans
        ]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)


def custom_tokenizer_from_pretrained(
    tokenizer_file_or_name: str, cache_dir: Optional[str] = None
) -> PreTrainedTokenizerFast:
    """Load BertWordPieceTokenizer from tokenizer.json.
    This is necessary due to the following reasons:
    - BertWordPieceTokenizer cannot load from tokenizer.json via .from_file() method
    - Tokenizer.from_file(tokenizer_file) cannot be used because MecabPretokenizer is not a valid native PreTokenizer.
    """

    if os.path.exists(tokenizer_file_or_name):
        tokenizer_dir = os.path.dirname(tokenizer_file_or_name)
        pt_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            cache_dir=cache_dir,
        )

        # This is necessary for pt_tokenizer.save_pretrained(save_path)
        _tokenizer = Tokenizer.from_file(tokenizer_file_or_name)
        _tokenizer.pre_tokenizer = PreTokenizer.custom(MecabPreTokenizer())
        pt_tokenizer._tokenizer = _tokenizer

    else:
        # trf>=4.0.0: PreTrainedTokenizerFast by default
        # NOTE: AutoTokenizer doesn't load PreTrainedTokenizerFast...
        pt_tokenizer = BertTokenizerFast.from_pretrained(
            tokenizer_file_or_name,
            cache_dir=cache_dir,
        )
    return pt_tokenizer
