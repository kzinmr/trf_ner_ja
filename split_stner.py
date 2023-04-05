import json
import random
import pickle

random.seed(42)


def train_valid_test_split(
    dataset: list, valid_ratio: float = 0.2, test_ratio: float = 0.2
) -> tuple[list, list, list]:
    random.shuffle(dataset)
    n = len(dataset)
    train_valid, test = (
        dataset[: int(n * (1 - test_ratio))],
        dataset[int(n * (1 - test_ratio)) :],
    )
    rel_valid_ratio = valid_ratio / (1 - test_ratio)
    _n = len(train_valid)
    train, valid = (
        train_valid[: int(_n * (1 - rel_valid_ratio))],
        train_valid[int(_n * (1 - rel_valid_ratio)) :],
    )
    return train, valid, test


#

from urllib.request import urlopen


def get_data_from_url(url: str):
    with urlopen(url) as response:
        return response.read().decode("utf-8")


url = "https://raw.githubusercontent.com/stockmarkteam/ner-wikipedia-dataset/main/ner.json"
res = get_data_from_url(url)
data = json.loads(res)
# with open("ner.json", encoding="utf-8", errors="ignore") as fp:
#     s = fp.read()
#     data = json.loads(s)

print(len(data), data[0])
# {'curid': '3572156',
#  'text': 'SPRiNGSと最も仲の良いライバルグループ。',
#  'entities': [{'name': 'SPRiNGS', 'span': [0, 7], 'type': 'その他の組織名'}

span_data = [
    {
        "text": d["text"],
        "spans": [
            {"start": ent["span"][0], "end": ent["span"][1], "label": ent["type"]}
            for ent in d["entities"]
        ],
    }
    for d in data
]

train, valid, test = train_valid_test_split(span_data, valid_ratio=0.1, test_ratio=0.2)

with open("train.jsonl", "wt") as fp:
    for d in train:
        js = json.dumps(d, ensure_ascii=False)
        fp.write(js)
        fp.write("\n")
with open("valid.jsonl", "wt") as fp:
    for d in valid:
        js = json.dumps(d, ensure_ascii=False)
        fp.write(js)
        fp.write("\n")
with open("test.jsonl", "wt") as fp:
    for d in test:
        js = json.dumps(d, ensure_ascii=False)
        fp.write(js)
        fp.write("\n")


label_set = {
    "O",
    "B-人名",
    "I-人名",
    "B-法人名",
    "I-法人名",
    "B-政治的組織名",
    "I-政治的組織名",
    "B-その他の組織名",
    "I-その他の組織名",
    "B-地名",
    "I-地名",
    "B-施設名",
    "I-施設名",
    "B-製品名",
    "I-製品名",
    "B-イベント名",
    "I-イベント名",
}

with open("label_set.pkl", "wb") as fp:
    pickle.dump(label_set, fp)
