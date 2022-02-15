import os
import pickle
from typing import Dict, List

import fugashi
import unidic_lite
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from predictor_en import Decoder, Predictor, Token, TokenLabelPair


os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        window_stride: int = 5,
    ) -> List[List[Token]]:
        """一文のデータが長い場合にストライド付き固定長分割を施す処理."""
        tokens = self.tokenize_with_alignment(sentence)

        seq_length = len(tokens)
        if seq_length <= max_length:
            return [tokens]
        else:
            sentence_windows = []
            # max_length を窓幅, window_strideをストライド幅とする固定長に分割
            for start in range(0, seq_length, window_stride):
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

    def encode(self, sentence_text: str):
        """Tokenizationを用いてテキストをトークナイズ・Tensor化する。
        NOTE: デコーディングのために offset_mapping も保持する。
        NOTE: 長文の対策として、ストライド付きウィンドウ分割も行う。
        """
        # sentence -> [sentence_window(max_length, window_stride)]
        sentence_windows = self.pretokenizer.window_by_tokens(
            sentence_text, self.max_length - 2, self.window_stride
        )
        token_windows = ["".join(tok.text for tok in toks) for toks in sentence_windows]
        encodings = [
            self.tokenizer(
                sent,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                # stride=self.window_stride,  # fast-only
                # return_overflowing_tokens=True,  # fast-only
                add_special_tokens=True,
                return_token_type_ids=False,
                return_length=True,
            )
            for sent in token_windows
        ]
        dataset = [
            {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
            for enc in encodings
        ]
        offset_mapping = [
            [(tok.start, tok.end) for tok in toks] for toks in sentence_windows
        ]
        return dataset, offset_mapping


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
        self.decoder = Decoder(id2label, window_stride)

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
        dataset, offset_mapping = self.encoder.encode(sentence_text)
        outputs = self.predictor.predict(dataset)
        tokens_in_sentence = self.decoder.decode(sentence_text, offset_mapping, outputs)
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
    print(tokens_in_sentence)
    for t_l in tokens_in_sentence:
        print(t_l.token.text, t_l.label)
