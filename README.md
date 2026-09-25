# spec-hierarchy

Three-stage speculative decoding on top of [DFlash](https://github.com/z-lab/dflash):

```
block-diffusion drafter  ->  middle verifier (cheap copy of the target)  ->  target
      (DFlash)               accepts / rejects draft blocks,                 checks all pending
                             feeds its hidden states to the drafter          tokens in one pass
```

The target is only called once a window of P mid-accepted tokens is pending. Under greedy
decoding every emitted token is still the target's argmax, so the output is identical to the
target's own greedy output.

## Why the window size matters

With per-token agreement beta = 1 - eps between middle model and target, and a target check
costing r times the per-token cost of the lower stages, tokens per unit cost are maximized at

    P* ~= ln(1 + eps * r) / eps

so long windows (64+) only pay off when beta >= ~0.98 and r ~ 100. Exp 1 measures beta,
exp 3 measures the real trade-off.

## Setup (Colab A100)

```bash
git clone https://github.com/ladka6/spec-hierarchy && cd spec-hierarchy
bash run_pilot.sh                 # installs deps, runs the test and all three experiments
# SKIP_INSTALL=1 bash run_pilot.sh   to rerun without reinstalling
```

Models (downloaded from Hugging Face on first use): `Qwen/Qwen3-8B` target,
`z-lab/Qwen3-8B-DFlash-b16` drafter, middle candidates `Qwen3-8B` in 4/8-bit plus
`Qwen3-4B / 1.7B / 0.6B`. Needs about 40 GB of disk and fits a 40 GB A100.

## Experiments

| script | question | output |
|---|---|---|
| `tests/test_cpu.py` | Are all decoders lossless? (tiny random models) | `ALL PASSED` |
| `scripts/exp1_agreement.py` | How often does each middle model agree with the target, and how long are the agreement runs? | `results/exp1_agreement.json` |
| `scripts/exp2_features.py` | Does the drafter keep its acceptance length when fed the quantized model's hidden states? | `results/exp2_features.json` |
| `scripts/exp3_pipeline.py` | End-to-end tokens/s of AR, DFlash, and three-stage with fixed and adaptive windows | `results/exp3_pipeline.json` |

Each script takes `--n` (prompts per dataset), `--max-new`, `--datasets`, `--mids`.
Middle model specs: `bnb4:<hf id>`, `bnb8:<hf id>`, `hf:<hf id>`.

Rough runtimes on an A100: exp1 ~15 min, exp2 ~15 min, exp3 ~30 min (mostly model loading
and the autoregressive baseline).

## What to look at

- exp1: `beta` and `E[acc|P]` for `bnb4:Qwen/Qwen3-8B`. Feasibility needs beta >= ~0.97.
- exp2: `tau` for `M>T` vs `T/T`. A small drop means the drafter can run on the
  middle model's features without retraining.
- exp3: `x_dflash` for the `3s-*` rows, and `match` should be ~1.0.

Note: bitsandbytes 4-bit kernels are slow at batch size 1, so exp3 speed numbers
understate what a proper W4 kernel (AWQ/Marlin) would give. exp1/exp2 are
kernel-independent.

## Layout

```
hspec/pipeline.py   decoders: ar, two-stage (DFlash with any verifier/feature model), three-stage
hspec/models.py     loading target, middle models, drafter
hspec/data.py       prompts (DFlash benchmark formats), chat template
scripts/            experiments
tests/test_cpu.py   losslessness test on tiny random models
```
