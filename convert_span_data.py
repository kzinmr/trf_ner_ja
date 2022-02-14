import json
import sys

input_filename = sys.argv[1]
assert input_filename.endswith(".jsonl")
with open(input_filename) as fp:
    jds = [json.loads(l) for l in fp if l.strip()]
span_dataset = [
    {
        "text": text,
        "spans": [{"start": s, "end": e, "label": l} for s, e, l in entd["entities"]],
    }
    for text, entd in jds
]

output_filename = input_filename + ".new.jsonl"
with open(output_filename, "wt") as fp:
    for d in span_dataset:
        js = json.dumps(d, ensure_ascii=False)
        fp.write(js)
        fp.write("\n")
