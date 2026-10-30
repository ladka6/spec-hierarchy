# Synthesis: what we know, what failed, what is still open (30 October 2026)

Internal working document (experiment ids kept on purpose). Setup unless stated otherwise: Qwen3-8B target,
DFlash drafter z-lab/Qwen3-8B-DFlash-b16 (block 16), greedy decoding, offline measurements on cached greedy target
text (530 sequences: 420 train / ~104 held out). tau = accepted tokens per verification including the bonus token.

## 1. The gap (agreed with the supervisor)

Block drafters draft every position of a block from target features that stop at the anchor. The drafted positions
never get target information. The recent papers that address in-block quality (DFlash 2, D-Loop, DEdit, Adjacent
Causal Injection, DStill, FlashMap) only rearrange or refine information the drafter already has. DFlash 2's own
oracle shows the ceiling: even perfect selection from the top 16 candidates falls from 99.5% to 87.8% by position 6.
Of the 11 block-diffusion drafter papers in the taxonomy, only ReAnchor brings target-side information to drafted
positions, and only after a rejection, from states conditioned on a wrong prefix. DLoop (Oct 2026) chains drafting
stages using the drafter's own hidden states in place of target states (+5 to 41% speed, gain disappears at
concurrency 32).

## 2. What we measured (positive findings)

**Where acceptance is lost**
- DFlash per-token accuracy given a correct prefix is flat at about 0.84; errors compound, tau = (1 - p^B) / (1 - p).
  Top-k trees bound tau at 7.25 (top-2) and 8.64 (top-4).
- Redrafting at the stop positions (exp26): prefix tokens alone fix 49% of stops, features of the target's first 18
  layers 57%, real target features 80%. 38% of stops happen where the target is nearly certain; there real features
  fix 95.5%. The missing ingredient is the target's deep computation over the drafted tokens, not token dependency.

**How much target information is worth (exp71)**
- Real features tau 8.44. Removing the features of the last m context positions: 3.60 / 2.18 / 1.60 (m = 4 / 8 / 15).
- Only the deep taps matter (L25, L33), with a threshold near cosine 0.95. L1 / L9 / L17 can be near random.
- A 4-bit copy's features reach the real level: 8.42 / 8.38 / 8.30.

**The cheap copy (exp66, exp69, exp70 pending)**
- Greedy agreement with the target: 0.958 (AWQ) / 0.967 (GPTQ w4a16) in vLLM; runs >= 20 tokens 49 to 57%,
  >= 100 tokens 9 to 13%; conditional agreement rises along a run (0.967 to 0.984).
- It knows where it is wrong: its own top-1 margin flags disagreements with AUROC 0.961; the target's token is in its
  top-2 99.5% of the time; disagreements sit at the target's own near-ties (target margin 0.21 there vs 0.87 overall).
  Branching only at low copy margin (k = 2) gives effective agreement 0.988 at 7.7 branch points per 100 tokens.

**Feature refresh with the copy (October 5 report)**
- 4-bit copy 12.53 accepted per verify = real features 12.54; 3-bit 12.09; 2-bit RTN 8.44 (fails); plain DFlash 7.98.
- End to end (40 prompts): DFlash 5.86 -> 10.17 (+74%), DDTree-64 7.52. Gain split: ~13% from the copy's
  correction, ~25% from redrafting, ~61% from the copy's features.
- Speed: projected 1.24 to 1.40x with vLLM component costs; around break even (0.93 to 0.97x) in our HF code.

**Costs (vLLM, batch 1)**: drafter 2.6 ms; AWQ copy 3.2 ms (17 tokens), 3.7 (33), 9.1 (65); target verify 13.3 ms
(17 tokens), 14.6 ms (65). Verify cost barely grows with the number of tokens.

**The target's top layers correct errors (exp73)**: layer-k states with random error at cosine c, finished by the
target's own layers k..35 over the real context: k = 28, c = 0.9 -> tau 12.6 (of 16); c = 0.8 -> 11.2. The same error
on the last layer with no upper layers: 10.9 / 9.7. Random error is optimistic (the LM head alone tolerates it).

## 3. What failed, and why

| Attempt | Result | Why it failed |
|---|---|---|
| Small learned feature emulator (50M) | cosine 0.38, fixes fewer stops than tokens alone | emulating 33 layers of an 8B model needs far more capacity |
| Untrained pruned copy (24 / 18 layers) | gain gone at 18 layers | deep features depend on all earlier layers |
| Pruned copy "taps8" + LoRA, drafter loss (exp55) | tau 9.20 vs 11.90 for the 4-bit copy | training recovers part, not all |
| 2-bit copy: RTN / trained (exp55) | 8.44 (fails) / agreement 0.82 | too little precision left |
| Gating the extra work to uncertain rounds | lowers acceptance more than it saves | the gate misses useful rounds |
| Sequential restarts (M4) | tau 7.59 -> 9.93 for 1 to 4 restarts, best 1.14x; cheap features < 1x | each restart pays a drafter call; drafter accuracy caps the chain |
| Online correction of the copy from verified states (exp68) | error autocorrelation 0.39 (lag 1); estimators explain 12 to 20% of error energy; agreement unchanged | the copy's errors are position-specific near-ties, not a smooth bias |
| Probe for target features (exp67) | data-limited, memorising | 420 sequences |
| Lookahead placeholders in the verify pass, frozen target + gated LoRA (exp72) | offset 1: cosine 0.87 to 0.90, accuracy 0.77 (DFlash 0.93); falls to 0.10 by offset 15; tau 5.85 / 4.45 / 3.00 for m = 4 / 8 / 15 | a placeholder cannot know the token at its own position: it can only produce an expected state |
| DFlash states -> linear map -> target layers k..35 (exp74) | tau 5.63 to 6.49 vs DFlash 8.47 (no upper layers: 3.94) | DFlash's states are not target-like (cosine 0.63 to 0.80); realistic errors hurt more than random ones |
| Same, trained end to end (exp76, k = 28) | tau 7.28 vs fine-tuned control 8.41 | the drafter cannot learn target mid-layer states from its own information |
| True future summary added to the drafter (exp75, add / gated / scaled / kv) | no effect at any bottleneck size | **unresolved**: training loss tracks d = 0 to the 4th decimal, likely the summary never reaches the predictions (outlier dimensions); oracle-token check and per-dimension standardisation pending |

General caveat: with 420 training sequences, even a plain fine-tune of DFlash drifts (8.65 -> 8.41). Every
training-based negative result above is data-limited to some degree.

## 4. The principle the evidence supports

Target information about drafted positions comes only from target-like computation over the drafted tokens. Every
shortcut that tries to predict it without that computation (placeholders, maps from the drafter's states, small
emulators, summaries without the tokens) falls well short. What remains open is where that computation happens, who
performs it, and how its cost is paid.

## 5. Constraints for the methodology

From the supervisor:
- the gap above is agreed;
- a plain quantized copy that refreshes features is not publishable on its own (October 10);
- a multi-stage pipeline with a quantized intermediate verifier was liked as a frame (October 13), but not "just
  adding a quantized copy".

From Ege:
- a method (architecture) contribution, not an analysis or theory paper;
- one genuinely novel idea, not a combination of existing methods;
- compute cost is not a selection criterion; retraining (including the target, if needed) is acceptable;
- single-GPU experiments preferred for now.

## 6. Open questions to search for a methodology (not proposals)

1. **Who performs the target-like computation.** So far: a separate copy (works, ruled out as the contribution), the
   target's verify pass (only on verified tokens), or the drafter (cannot). Is there an architecture where the
   drafter and the target-like computation are the same network, so the computation is not an add-on?
2. **When it happens.** Before drafting (needs tokens that do not exist yet), between drafting stages (DLoop's slot,
   currently filled with the drafter's own states), or inside verification (ReAnchor's slot, only after rejections).
   Is there a slot where the cost is hidden or shared?
3. **How often it is paid.** Per round, per stage, or amortised over several rounds (one computation serving several
   blocks).
4. **What it is used for.** Features for the drafter (refresh), or decisions (where to branch, when to stop, when to
   call the big verifier), where the copy's calibrated margin (AUROC 0.96) is already a strong signal.
5. **The multi-stage verification frame.** The supervisor's preferred frame treats the intermediate model as a
   verifier, not a feature source. What is the research problem inside it that is not "add a quantized copy"?

## 7. Pending runs

- exp75 oracle (`--source tokens`) and deep source with per-dimension standardisation and shuffle evaluation: decides
  whether a true future summary helps at all.
- exp70: copy vs target distance under sampling (1 - TV) at T = 0.6 / 1.0.
