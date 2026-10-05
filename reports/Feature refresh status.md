# Feature-refreshed drafting: status (5 October 2026)

## 1. Problem

Block drafters such as DFlash predict a whole block of 16 tokens from the target's hidden features at a single anchor. The tokens the drafter produces inside the block never get target features, so the accuracy drops along the block (top-1 accuracy 0.93 at offset 1, 0.37 at offset 15, Qwen3-8B, T=0). Most of the recent work (D2SD, DEdit, xPress, Domino, DSpark, DFlash 2) treats this as a dependency problem and adds token conditioning, refiners or editing passes on top of the drafter.

We measured where the acceptance is actually lost (exp26). At the positions where the draft stops, we redraft with three kinds of information about the true prefix:

| Redraft with | Stops fixed (DDTree-64, T=0) | Stops fixed (T=1) |
|---|---|---|
| prefix tokens only | 49% | 41% |
| features of the target's first 18 layers | 57% | 51% |
| real target features | 80% | 68% |

Knowing the correct tokens gives only a part of the gain. The rest is the deep computation of the target over these tokens, which the drafter can not reproduce in one pass. 38% of the stops happen where the target itself is nearly certain (entropy below 0.1), and there the real features fix 95.5% of them. So the main limitation of block drafters is the missing target computation, not the missing token dependency. This also explains why the dependency fixes stay at single digit gains over a tree.

## 2. Method

If the drafter needs the target computation, the question is whether a cheap version of it is enough. One round of the method:

1. DFlash drafts block 1 from the real target features.
2. A low-bit copy of the target runs once over the drafted tokens, with its own KV cache. It gives approximate features for these tokens and its own next-token predictions.
3. The copy's predictions locate the first position where it disagrees with the draft and give the corrected token. The drafter restarts from there, conditioned on the copy's features (one batched drafter call, optionally with extra restarts at low-confidence positions).
4. The target verifies block 1 and the restarted chains as one tree. Every emitted token is the target's argmax, so the method is lossless.

The drafter and the target are not retrained.

## 3. Results

Feature quality of the copy (offline, T=0, restarts at the most likely stop positions, plain DFlash 7.98):

| Copy | Accepted tokens per verify |
|---|---|
| 4-bit | 12.53 (identical to real target features, 12.54) |
| 3-bit | 12.09 |
| 2-bit (round to nearest) | 8.44, fails |

Ablation of the copy-corrected restart (offline, T=0, all with 31 tree nodes or less):

| Variant | Accepted tokens |
|---|---|
| plain DFlash | 7.98 |
| copy correction only | 8.82 |
| correction + redraft from tokens only | 10.42 |
| correction + redraft with copy features | 14.25 |

About 13% of the gain comes from the correction itself (the known quantized intermediate verifier effect), about 25% from redrafting and about 61% from the copy's features. The same split holds at T=1.

End to end in the real decoding loop (greedy, 40 prompts from GSM8K, MATH-500, HumanEval, MT-Bench, Qwen3-8B):

| Method | Accepted tokens per verify |
|---|---|
| plain DFlash | 5.86 |
| refresh, copy-corrected restart | 10.17 (+74%) |
| refresh, + 2 confidence restarts | 10.79 (+84%) |
| DDTree-64 (reference) | 7.52 |

The gain is present on all four datasets, also on MT-Bench. The implementation is tested on CPU for exact equality with greedy decoding.

## 4. What did not work

- A small learned emulator of the features (50M parameters) reached cosine 0.38 and fixed less stops than tokens alone. Emulating 33 layers of an 8B model needs much more capacity.
- Pruning the target without training (keeping 24 or 18 of the layers) loses a lot. With 18 layers the gain is gone.
- Gating the extra work to only uncertain rounds lowers the acceptance more than it saves.

## 5. Speed

The extra copy pass and the second drafter call have to cost less than what the longer accepted chains save.

- With vLLM component costs (4-bit AWQ copy with Marlin kernels 3-6 ms, bf16 verify 13 ms) the projected speedup over plain DFlash is 1.24-1.40x.
- In our own HF code with CUDA graphs, every stage is measured: drafter 3.7 ms, copy 15.5-17.7 ms, verify 22.0-23.6 ms. Here the method is around break even (0.93-0.97x). The copy costs 0.70 of a verify in HF instead of 0.23-0.45 in vLLM, because the per-layer overhead outside the matrix multiplications does not shrink with quantization.

A measured speedup needs an implementation inside vLLM. A simpler version that verifies only the copy-corrected chain would fit vLLM's linear verification and avoid tree attention. The estimate is 2-4 weeks of engineering.

## 6. Points to discuss

1. The thesis core is there: a diagnosis, a method, ablations and a real loop. Is the vLLM implementation worth 2-4 weeks for a paper submission?
2. More target models (Qwen3-4B, a LLaMA model), T=1 with lossless sampling, and the baselines (DFlash 2, D2SD, DEdit) are needed in any case.
3. A full novelty check is still open. The closest work found so far is D2SD (redrafting with tokens), KVShot (target KV only for verified tokens) and the quantized hierarchy methods (QuantSpec, ML-SpecQD, Polybasic), none of which uses a target copy as the feature source for a separate drafter.

## 7. Future direction: injecting the future without a full copy

The copy shows that the drafter needs an approximation of the target's computation over the drafted tokens. A full low-bit copy is only the simplest way to get it. Directions to try:

1. **Drafter trained with the copy in the loop.** The 2-bit copy failed with the original drafter. A drafter fine-tuned on 2-bit or pruned-copy features may learn to tolerate their errors, which halves the copy cost again.
2. **Small correction on top of a cheap copy.** A light adapter that maps the features of a 2-bit or pruned copy to the real ones, trained with the drafter's KL like the faker. The copy gives the computation, the adapter only fixes its errors.
3. **Distilled feature generator from the target's own layers.** Keep a subset of the target layers and fine-tune them to produce the features DFlash reads, instead of a full copy. Untrained pruning failed, but it was never trained.
4. **Reuse the verification pass.** The target already computes real features for every drafted node it verifies. Verifying a longer or wider speculative continuation is almost free at batch 1, so the next round could start with real features further ahead, which injects the future from the verifier itself instead of from a second model.
5. **Hide the copy behind the verifier.** With two GPUs the copy pass for the next round can run while the target verifies the current one, if the next block is drafted from the predicted continuation. Then the copy cost leaves the critical path.

All five keep the main finding, that drafters need target computation and not only tokens, and try to make that computation cheaper or free.
