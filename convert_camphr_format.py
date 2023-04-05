import json
import sys

input_filename = sys.argv[1]
assert input_filename.endswith(".jsonl")
with open(input_filename) as fp:
    jds = [json.loads(l) for l in fp if l.strip()]
    jds = [
        (text, sorted({(s, e, l) for s, e, l in entd["entities"]}, key=lambda x: x[0]))
        for text, entd in jds
    ]

span_dataset = [
    {
        "text": text,
        "spans": [{"start": s, "end": e, "label": l} for s, e, l in ents],
    }
    for text, ents in jds
    if len(text) > 5
]

output_filename = input_filename + ".new.jsonl"
with open(output_filename, "wt") as fp:
    for d in span_dataset:
        js = json.dumps(d, ensure_ascii=False)
        fp.write(js)
        fp.write("\n")
