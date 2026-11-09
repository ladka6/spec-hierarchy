"""Quick look at generated LayerSkip shards: counts, lengths, rambling, one sample.

  python scripts/inspect_lsdata.py /scratch-shared/$USER/hspec_ls/data_ft
"""
import glob
import sys

import torch
from transformers import AutoTokenizer

d = []
for f in sorted(glob.glob(f"{sys.argv[1]}/shard*.pt")):
    d += torch.load(f)
tok = AutoTokenizer.from_pretrained("facebook/layerskip-llama3-8B")
resp = [tok.decode(x["ids"][x["n_prompt"]:]) for x in d]
n_resp = sorted(len(x["ids"]) - x["n_prompt"] for x in d)
n_tot = [len(x["ids"]) for x in d]
print(f"{len(d)} seqs, response mean {sum(n_resp) / len(n_resp):.0f} median {n_resp[len(n_resp) // 2]} "
      f"max {n_resp[-1]}, total len mean {sum(n_tot) / len(n_tot):.0f}, kept at 2048: {sum(t <= 2048 for t in n_tot)}")
print(f"first token BOS: {sum(int(x['ids'][0]) == 128000 for x in d)} of {len(d)}")
print(f"responses with a new Question: turn: {sum(chr(10) + 'Question:' in s for s in resp)}")
print(f"responses ending in EOS: {sum(s.endswith('<|end_of_text|>') for s in resp)}")
print(f"responses ending at a Question: stop: {sum(s.rstrip(':').endswith('Question') for s in resp)}")
print(f"tokenizer has a chat template: {bool(tok.chat_template)} (must be False: Q/A format, as in eval)")
print("prompt start:", repr(tok.decode(d[0]["ids"][: d[0]["n_prompt"]])[:200]))
for i in (0, 1, 2):
    print(f"response {i} start:", repr(resp[i][:200]))
    print(f"response {i} end:  ", repr(resp[i][-120:]))

# loops among length-capped responses: share of distinct 8-grams in the last 512 tokens (1.0 = no repetition)
capped = [x["ids"][x["n_prompt"]:].tolist() for x in d if len(x["ids"]) - x["n_prompt"] >= 1024]
def distinct(t, n=8):
    g = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    return len(set(g)) / max(len(g), 1)
dist = sorted(distinct(t[-512:]) for t in capped)
if dist:
    q = lambda p: dist[int(p * (len(dist) - 1))]
    print(f"length-capped: {len(capped)}; distinct 8-gram share in last 512 tokens: p10 {q(.1):.2f} median {q(.5):.2f}; "
          f"looping (< 0.5): {sum(v < 0.5 for v in dist)}, heavy (< 0.2): {sum(v < 0.2 for v in dist)}")
    worst = min(capped, key=lambda t: distinct(t[-512:]))
    print("most repetitive tail:", repr(tok.decode(worst[-150:])))
