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
print("sample:", repr(resp[0][:400]))
