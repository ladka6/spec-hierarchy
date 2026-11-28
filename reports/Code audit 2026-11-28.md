# Code audit (28 November 2026)

Scope: the code behind the numbers in `Status 2026-11-24.md` and the meeting notes, read line by line:
`hspec/lagtrain.py`, the DFlash package (`dflash/model.py` 0.1.0), `hspec/models.py`, `hspec/data.py`,
`scripts/gen_traindata_vllm.py`, `gen_pairdata.py`, `exp27_refresh.py` (block helpers), `exp107_lift_diag.py`
(loader), `exp111`, `exp115`, `exp116`, `exp118`, `exp123`, `exp124`, `exp125`, `exp126`, `bench_band.py`,
`train_lag.py` (split only). Not read in this pass: `exp113`, `exp114`, `exp117`, and the scripts before exp111.

The audit was done by the same assistant that wrote most of this code, so it is not an independent review. Section 3
lists checks that do not depend on its reading.

## 1. Findings

### 1.1 The lag-trained drafter is evaluated on its own training sequences in exp117 / exp123 / exp124 (real issue)

`train_lag.py` trains on every sequence of `hspec_lagft/data` except 48 (`data[:48]` after a seeded shuffle).
exp116 / 117 / 118 / 123 / 124 take their "held-out" anchors from `seqs[fit_end:]` of `load_seqs`, which is held out
from the fits made inside those scripts and from the original z-lab drafter, but is part of the lag drafter's
training set. Every number produced with `drafter_lag16_fp32` in exp117, exp123 and exp124 is therefore measured
on sequences that drafter saw once in training: the chain depth table (one block 3.82, A(B), survival per cut, 1.65x
and 2.07x), the cut policy table with the lag drafter (one block 4.466, 7.40 / 6.65, the fork rows), exp117 with the
lag drafter.

Direction of the bias: optimistic. Size: unknown; one epoch, each sequence seen once with random anchors.
Not affected: everything with the original drafter (exp111, 115, 116, 118, exp123 / exp124 with the original
drafter), and exp125 / exp126 (they use `train_lag.py`'s split: trained on its training part, evaluated on its 48
validation sequences).

Fix: `--lag-val` in exp123 and exp124 takes the anchors from those 48 validation sequences. Commands in section 3.

### 1.2 exp126: arms are not on the same anchors (limits cross-arm reading)

In table B the random generator that draws the anchors is also used by the "shuffled" and "other positions"
interventions. Arms with those interventions (16, 64, 256) therefore draw other anchors than arms without (0, real).
Visible at lag 0, where the code is not used: 5.53 to 5.55 for the code arms, 5.70 and 5.65 for arm 0 and real.
Within an arm every intervention uses the same anchors, so "all / last 1 / last 2 / hidden" comparisons stand.
Comparisons between a code arm and arm 0 or real in exp126 (the hazard table, tau levels) are not paired. The claim
that the loss sits at the first token of the block also follows from exp125's first-token accuracy on common
anchors (tokens 0.72, real 0.85 at lag 8), so it does not rest on exp126 alone.

### 1.3 No interval on any number

The standard error of a mean tau over 754 blocks is about 0.18 for two unpaired sets (the lag 0 gap in 1.2 is an
example). exp125 arms share their anchors, so differences between arms are paired and tighter, but no interval was
ever computed, and blocks of one sequence are not independent (48 sequences). Statements that rest on differences
of 0.1 to 0.15 should be read as unconfirmed: LoRA on the band path (4.76 against 4.86), one stage against two
stages for tokens (4.54 against 4.38), the predicted code in self64 (+0.09), the differences among the four-layer
paths. Differences of 0.5 and more (tokens against real, the code sizes, the band) are well outside this.

Fix: exp125 now saves the taus of every block per sequence; `scripts/paired_ci.py` gives the paired difference with a
bootstrap interval over sequences. Existing runs did not save them.

### 1.4 exp111 "no features for the last m positions": wrong description in the notes

The script does not hide the last m tokens. It gives them to the ORIGINAL drafter as token inputs of the block (the
lag layout), which that drafter was never trained on. The drop (8.54 to 2.07 at m = 8) therefore mixes missing
target information with an input layout the drafter has not seen. The slide notes of 26 November said the drafter
"gets nothing at all for those tokens"; that is wrong. The number that isolates the information is the lag-trained
one (5.7 to 4.54 at lag 8 with tokens, 5.78 with real features).

### 1.5 Speed estimates use a constant drafter time

`bench_band.py` times a drafter call with 1 or 16 context features, 16 block positions, no cache of the earlier
context. In a chain the drafter's input grows with every stage (in-flight tokens are block inputs: g + 16 positions,
up to about 60 + 16 at stage 8 without features), and in real use it also attends to its cache of the prefix.
exp123 / exp124 charge 3.55 ms for every stage. Width is cheap on this GPU (verify: 20.4 ms for 16 tokens, 21.7 for
48), so the error is probably small, but it was not measured and it favours the chain. All speeds remain formulas
over measured pieces, not end-to-end runs.

### 1.6 Stored responses are assumed to be the target's greedy tokens

tau counts a match with the stored next token. `hspec_lagft/data` was generated with vLLM, the pair data with
batched left-padded `generate`; the scripts never check that the stored token is the argmax of the target they
load. A mismatch lowers every arm's tau equally and does not favour an arm. Evidence that it is small for the pair
data: exp115 reads the next token from the last layer at 0.995 to 0.997. Not checked for the vLLM data.
Check: `scripts/audit_data_greedy.py`.

### 1.7 Smaller points

- exp125 evaluation: anchors go up to T - 3, so blocks near the end of a sequence are shorter than 15 slots and cap
  tau. Same for every arm.
- exp125 checkpoints store the drafter in bf16. The effect is small: self64 gave 5.614 at lag 0 at the end of its
  run and 5.621 after reloading.
- exp125 / exp126 reproduce `train_lag.py`'s split only if the shard directory holds the same files as at lag
  training time. exp125 prints "22443 train / 48 val sequences"; the lag training log should show the same count.
- exp126 table D: "code variance explained" is a single ratio over all dimensions, dominated by the two to four
  largest directions of the code. The tau column is the meaningful one.
- exp126 "other positions": blocks with too few donor positions are skipped, so that row can have fewer blocks.

## 2. Checked and found correct

- `pack` (every tau comes through it): a block at anchor s with lag g reads features [0, s - g), tokens s - g .. s
  and MASK slots; positions and mask as documented. Confirmed by `tests/test_leakage.py` on tiny random models: no
  dependence on later tokens or on features at / after s - g, sensitivity to the anchor and to the last allowed
  feature, blocks packed together independent, and at lag 0 the packed call equals the call `dflash_generate`
  makes (features of all positions before the anchor, anchor token without a feature). 30 checks pass.
- The definition "features exist for [0, s), anchor s has none" matches `dflash_generate` (after a verify the
  features of the anchor and the accepted tokens are kept, the bonus token becomes the new anchor).
- exp125 layouts (real / code / state: second copy of the rows; path / self / xattn: rows per block): masks and
  positions read line by line; in-flight rows are positions s - g .. s - 1 only; real features of those positions
  cannot contain the anchor or later tokens (causal target). The producer arms read embeddings, the verified
  prefix, and for self the drafter's own earlier block. Not covered by the CPU test (the layout lives inside
  `main`); covered by `--audit` (section 3).
- exp125 tau: leading matches with the stored tokens + 1, per block; same anchors in every arm and at every
  evaluation (fixed seed).
- exp125 / exp126 fits (PCA, ridge starts, readouts) use training sequences only; evaluation on the 48 validation
  sequences.
- exp115: targets, offsets and held-out split; the readout through the head assumes `hidden_states[-1]` is after
  the final norm, which the 0.995 at the last layer confirms.
- exp116 `BandStack`: prefix keys / values cut at the first in-flight position, bridges read prefix states only,
  start-up assertion that the full stack reproduces the real tap (cosine > 0.99).
- exp123 / exp124 chain logic: kept tokens, banned token position, dead chains, retry as a stage, A(B), the
  speed formula as documented. With real features and a late cut the features come from the true tokens, but such
  a chain is already counted as dead.
- exp118, exp111 mean ablation, `load_draft`, `format_prompt`.

## 3. Checks to run (GPU)

    python tests/test_leakage.py                      # CPU, tiny models; passes here
    bash snellius/submit_audit.sh                     # exp125 --audit for nine arms, a few minutes each
    grep -h "\[audit\]" exp125-*.out | grep -v " ok " # every line must end in "all checks passed"
    sbatch snellius/job_audit_data.sbatch             # 1.6
    DRAFT=$HOME/hspec_ckpt/drafter_lag16_fp32 TAG=lag_val EXTRA="--lag-val --shallow" sbatch --export=ALL snellius/job_exp124.sbatch   # 1.1
    DRAFT=$HOME/hspec_ckpt/drafter_lag16_fp32 TAG=lag_val EXTRA="--lag-val" sbatch --export=ALL snellius/job_exp123.sbatch            # 1.1

`--audit` perturbs what the target computed at chosen positions and checks whether a block's output moves: nothing
at or after the anchor and no later token may matter; the position just before the in-flight span must matter; what
the target computed at the in-flight positions may matter only in the oracle arms (real, code, state), never in a
producer arm (tokens, path, self, xattn). Zero-initialised outputs are opened first so a leak cannot hide behind
them. None of the new scripts (`--audit`, `--lag-val`, `audit_data_greedy.py`) has been run on a GPU.
