# Drafter depth / capacity vs block offset (DFlash family)

Date: 2026-10-04. Purpose: literature check for the thesis claim that a DFlash drafter is compute-limited at far block offsets, and for the candidate method "offset-dependent drafter compute".

Our finding (for reference): Qwen3-8B target, z-lab/Qwen3-8B-DFlash-b16 drafter, T=0. Top-1 accuracy falls from 0.93 (offset 1) to 0.37 (offset 15). Redrafting at a stop with the true prefix tokens fixes 49% of stops; with target features from the first 18 layers, 57%; with all 36 layers, 80%.

Source tags:
- [F] fetched and read this session.
- [N] from earlier verified notes in this folder (all_papers.csv, parallel_drafters.md).
- [K] from model background knowledge, NOT verified this session. Check before citing.
- [T] title/ID confirmed by web search only; content not read.

Access problems this session: arXiv started returning HTTP 429 (rate limited) after the DFlash fetches, and WebFetch then hit a session limit (resets 18:50 Europe/Amsterdam). Not read: DEdit (arxiv.org/html/2609.38510 and /abs), 2610.00888, PosS (openreview PDF returned 403), FastEagle, Future Lens, Gloeckle on alphaxiv. Items marked [K] or [T] should be re-checked when fetching works.

---

## 1. DFlash (arXiv 2602.06036) and released configs

### Released drafter config [F]
z-lab/Qwen3-8B-DFlash-b16 config.json (https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/blob/main/config.json):
- num_hidden_layers 5, hidden_size 4096, intermediate_size 12288, 32 heads, 8 KV heads, head_dim 128, all full attention.
- block_size 16, target_layer_ids [1, 9, 17, 25, 33], num_target_layers 36, mask_token_id 151669.
- HF card: "1B params", BF16, model.safetensors 2.1 GB. Arithmetic check (ours): per layer about 42M attention + 151M MLP = 193M, x5 = 0.97B, plus a 5x4096 -> 4096 fusion projection (84M) = about 1.05B. Embedding and LM head are shared with the target, so the 1B is almost all transformer depth.
- So the drafter has 5 layers against the target's 36. Each offset k must be produced by these 5 layers alone.

### Paper defaults [F]
- "5 layers" for most targets, "8 for Qwen3 Coder". Block size "16 (10 for LLaMA 3.1)".
- Target features: "5 layers uniformly selected between the second layer and the third-to-last layer".
- Loss weight per block position k: w_k = exp(-(k-1)/γ), γ = 7 for block 16, 5 for block 10, 4 for block 8. Motivation: "errors at early positions within a draft block invalidate all subsequent tokens".
- KV injection of target features into every draft layer is claimed to let acceptance "scale effectively with the number of draft layers".

### Ablations (Qwen3-4B target, T=0, 100K training samples; format speedup / τ) [F]
Table 6, number of draft layers (5 target features):

| Draft layers | Math500 | HumanEval | MT-Bench |
|---|---|---|---|
| 3 | 4.69x / 5.64 | 3.90x / 4.61 | 2.38x / 3.18 |
| 5 | 4.71x / 5.99 | 3.96x / 4.94 | 2.35x / 3.37 |
| 8 | 4.64x / 6.33 | 3.96x / 5.29 | 2.23x / 3.50 |

Paper text: "Deeper draft models are more expressive and achieve higher acceptance lengths, but they also incur higher drafting latency... the optimal number of layers depends on the deployment setting." τ rises monotonically 3 -> 5 -> 8 layers (+12% Math500, +15% HumanEval, +10% MT-Bench from 3L to 8L), but speedup is flat or falls.

Table 7, number of target hidden features (3-layer drafter):

| Features | Math500 | HumanEval | MT-Bench |
|---|---|---|---|
| 3 | 4.49x / 5.38 | 3.80x / 4.47 | 2.32x / 3.07 |
| 5 | 4.69x / 5.64 | 3.90x / 4.61 | 2.38x / 3.18 |

Table 8, train -> inference block size (8-layer drafter):

| Train -> infer | Math500 | HumanEval | MT-Bench |
|---|---|---|---|
| b16 -> b16 | 4.64x / 6.33 | 3.96x / 5.29 | 2.23x / 3.50 |
| b16 -> b8 | 3.87x / 5.09 | 3.39x / 4.44 | 2.12x / 3.18 |
| b8 -> b16 | 3.78x / 5.02 | 3.24x / 4.28 | 2.09x / 3.09 |
| b8 -> b8 | 3.97x / 5.21 | 3.53x / 4.61 | 2.22x / 3.29 |

"A model trained with a larger block size generalizes well to smaller inference-time block sizes", not the reverse. Block-8 model fully accepts the whole block 35.7% of the time.

Other ablations: KV injection vs input fusion (Table 9: DFlash KV τ 4.2 / 3.3x GSM8K vs input fusion 3.5 / 2.9x); random anchor sampling (Table 13: 5.64 vs 4.94 τ Math500).

Not in the paper (as far as the fetch showed): per-position acceptance curves, hidden-size ablation, any position-dependent architecture. Note the row overlap: Table 6 "3-L" = Table 7 "5-H", and Table 6 "8-L" = Table 8 "b16->b16"; these are the same runs.

### Other released drafters
- README model list [F, local /home/claude/dflash/README.md]: Qwen3 4B/8B, Coder-30B-A3B, Coder-Next, Qwen3.5/3.6 family, Gemma 4, GPT-OSS 20B/120B, Llama-3.1-8B, Kimi, MiniMax, GLM 5.1. Per-checkpoint layer counts were not fetched (session limit). From the paper: 5 layers except Qwen3-Coder (8). Earlier notes [N]: LLaMA-3.1-8B drafter is also about 1B.

## 2. DFlash 2 (https://inco.ai/blog/dflash2/) [F]

This is the most direct prior evidence for our hypothesis.

- Backbone stays 5 layers. Additions: a two-tap dynamic depthwise causal convolution, "Conv_k(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}", "inserted before and after each attention and feed-forward sublayer" (+16.5M params, about 3%), and a path selector (low-rank bilinear score over adjacent candidates, 256-dim embeddings, +2.0M params). Block 8 for Qwen3.8-27B, 16 for Muse Glimmer.
- Hypothesis stated in the blog: "a five-layer backbone may be too small to preserve dependencies across the block. If that is right, depth should help most at later positions. And it does!"
- Per-position Recall@1, Qwen3-4B, GSM8K, T=0:

| Position | 3L | 5L | 15L |
|---|---|---|---|
| 0 | 85.21 | 85.39 | 86.42 |
| 1 | 79.26 | 80.31 | 81.61 |
| 2 | 77.18 | 79.39 | 80.68 |
| 3 | 75.75 | 78.27 | 80.34 |
| 4 | 73.96 | 77.39 | 80.59 |
| 5 | 70.40 | 76.03 | 79.66 |
| 6 | 64.97 | 72.86 | 78.73 |

  Our arithmetic: 5L -> 15L gains +1.0 pt at position 0 and +5.9 pt at position 6. 3L -> 5L gains +0.2 pt at position 0 and +7.9 pt at position 6. The gain from depth grows with position.
- Oracle (top-16) recall 5L: 99.5% at position 0, 97.3% at 1, 87.8% at 6. "Even the oracle decays... No selector can fix that, because the candidates themselves are running out."
- With the convolution (5L + conv): about 79-80% recall@1 up to position 6 (77.61% at position 6), close to 15L.
- Cost: "Its convolutions add 3% parameters and 0.7% cycle latency; the ten extra layers of 15L add 15.2%." The 15.2% most likely means cycle latency (10 extra layers cannot be only 15% more drafter parameters); the sentence is ambiguous. The blog's verdict: ten extra blocks "add capacity everywhere, even at the early positions that had little left to gain, and erase much of the efficiency that makes DFlash attractive."
- Results: Qwen3.8-27B mean τ 4.80 vs DSpark 3.62; Muse Glimmer 5.70 vs DFlash 4.44; Qwen3.5-4B 5.97 vs 4.92. Throughput 2.7-3.4x AR (Qwen3.8-27B), 3.1-4.6x (Muse Glimmer).
- No scaling law and no mean-τ for 3L/5L/15L. Training setup for these variants: only "all drafters are trained under the same setup".

Relevance: DFlash 2 shows (a) far positions are capacity-limited and (b) uniform depth wastes compute on near positions. It then picks a cheap local-dependency fix (conv) instead of offset-targeted depth. Offset-dependent depth is the obvious unexplored middle option. Caveat for our story: their framing is "preserve dependencies", while our redraft experiment says dependency (true prefix) explains only 49% of stops vs 80% with full-depth features. The conv result partly argues the opposite (cheap dependency mixing closes most of the 15L gap at block 8). Our block 16 and offsets up to 15 are further out than their position 6.

## 3. Drafter scaling: depth, size, data

| Paper | Depth / size variants | What happens | Tag |
|---|---|---|---|
| DFlash (2602.06036) | 3 / 5 / 8 layers | τ up monotonically (5.64 -> 5.99 -> 6.33 Math500), speedup peaks at 5L (4.69x / 4.71x / 4.64x) | [F] |
| DFlash 2 blog | 3 / 5 / 15 layers | Recall@1 up mostly at late positions; 15L too slow | [F] |
| Scylla (2505.07858) | EAGLE-style, 1 / 2 / 5 layers = 138M / 275M / 413M | Log-linear acceptance vs drafter pretraining tokens and capacity; mean τ 6.32 vs EAGLE-2 3.94 (Vicuna-7B, T=0); bigger drafter costs more per step | [N] |
| EAGLE-3 (2503.01840) | 1 decoder layer, fixed | Speedup keeps rising with training data (data-scaling plot, up to 8x ShareGPT) once feature loss is removed; no depth scaling study | [N] + [K] for the data-scaling figure |
| P-EAGLE (2602.01469) | 1 vs 4 layers (parallel EAGLE) | LLaMA-3.1-8B AL 4-layer 3.92 HumanEval / 3.04 MT-Bench vs 1-layer 2.69 / 2.41; 4-layer slower per pass, slowdown at K=3 on MoE | [N] |
| JetSpec (2606.18394) | 5-layer causal parallel head (about 1B, est.) | No depth ablation recorded in our notes | [N] |
| HASS (2408.15766), GRIFFIN (2502.11018) | 1-layer EAGLE-style; GRIFFIN 0.41B-2.07B by target | Gains come from training alignment, not depth; no depth ablation in our notes | [N] |
| PARD (2504.18583) / PARD-2 (2605.08632) | separate small LMs: Llama3.2-1B, Qwen2.5-0.5B, R1-Distill-Qwen-1.5B | No depth or size sweep in our notes | [N] |
| Hydra++ (2402.05109) | 4-layer MLP heads + extra decoder layer | Deeper heads help but drafting latency grows with depth | [N] |
| FastMTP (2509.18362) | 1 recursive MTP head | Per-step acceptance about 80 / 56 / 36% for K=3 | [N] |

General pattern: acceptance length rises with drafter depth with diminishing returns; wall-clock speedup at bs 1 is flat or falls past a small depth (DFlash peaks at 5L, P-EAGLE gains only 1.10-1.36x over AR EAGLE-3). No paper reports acceptance vs depth broken down by offset except the DFlash 2 blog.

## 4. Position- or offset-dependent compute

Closeness scale: 0 unrelated, 1 same problem but different lever, 2 position-specialized parameters or partial depth-per-offset, 3 explicit extra depth only for far offsets in a parallel drafter.

| Work | What it does | Closeness | Tag |
|---|---|---|---|
| FastEagle (2509.20416) | Replaces EAGLE's per-step recurrence with a cascade of light layers in one forward pass, layer-wise supervision; as I recall, draft token k is read out after the k-th cascaded layer, so depth grows with offset by construction | 2-3 (depth increases with offset, but it is an AR-style cascade, not a block-diffusion drafter, and not a choice made for compute reasons) | [T] title/abstract via search; mechanism [K] |
| PosS (2506.03566) | Position-specialized draft layers: different drafter layers handle different draft positions (groups of positions) in EAGLE-2/HASS-style AR drafting, to cut error accumulation at later positions | 2 (per-position parameters, same depth per position) | [T]; mechanism [K] |
| DFlash 2 blog | Measures that depth helps late positions most; then rejects uniform depth on latency grounds and adds a conv | 2 as evidence, 1 as method | [F] |
| Adjacent Causal Injection, "Draft in Parallel, Condition Through Depth" (2609.36173) | Title suggests injecting adjacent-token causal information through drafter depth in a parallel drafter; not read | unknown, possibly 1-2 | [T] |
| "Match the Distribution, Not the Compute: Post-Training MTP Heads" (2610.00888) | Title suggests a counter-position: matching the distribution matters more than head compute; not read. Must check, could undercut or support our claim | unknown, possibly a direct counterpoint | [T] |
| xPress (2608.02438) | Light causal refiner over diffusion-draft candidates via parallel Jacobi iterations; +30% τ over DFlash | 1 (extra serial compute, uniform over positions) | [N] |
| DSpark (2607.05147), Domino, TreeFlash, DBLast, D2SD | Add a light sequential or dependency module to a DFlash backbone | 1 (dependency lever, not depth) | [N] |
| Depth-Adaptive Transformer (Elbayad et al., 1910.10073, ICLR 2020) | Per-token (and per-sequence) adaptive decoder depth with halting in AR MT | 1 (adaptive depth per token, not per offset in a parallel block) | [T] + [K] |
| Looped/recurrent-depth LMs: Ouro (2510.25741), LoopSpec (2609.17184), WaveFront Decoding (2609.23033), Depth-Asynchronous Self-Speculation (2609.34538) | Self-speculation inside looped target models; early loop iterations draft | 1 (looped depth is in the target, not in a parallel drafter) | [T], Ouro [N] |
| ReDrafter (2403.09919) | RNN drafter, serial per token | 0-1 | [T] |
| Gloeckle MTP (2404.19737) | n heads, one transformer layer each, same depth for every offset | 0-1 | [N] |
| DeepSeek-V3 MTP | Sequential depth-1 MTP modules, one per extra token; chaining modules gives offset k a k-module path | 1-2 (depth grows with offset only through sequential chaining) | [N] + [K] |
| FastMTP (2509.18362) | One shared MTP head applied recursively K=3; offset k gets k head passes (looped depth per offset, weight-tied) | 2 (weight-tied recurrent depth that grows with offset, but serial and AR) | [N] |
| Your LLM Knows the Future (2507.11851) | k mask tokens + gated LoRA in the full target; every mask position sees full target depth | 1 (full depth for all positions; shows capacity fixes much of the decay) | [N] |
| D-PACE (2605.18810) | Position-aware loss weighting for parallel drafters | 1 (loss, not compute) | [T] |

Summary: autoregressive and recursive drafters (EAGLE chain, FastMTP, DeepSeek MTP chaining, FastEagle cascade) all give later offsets more serial compute by construction. Among one-pass block/parallel drafters (DFlash, PARD, P-EAGLE, JetSpec, DSpark), I found no method that allocates extra depth, loops, or extra layers only to far block positions. DFlash 2 measured the need and chose a different fix.

## 5. Why accuracy decays with offset

- DFlash 2 blog [F]: depth gains concentrate at late positions; even the top-16 oracle decays (99.5% -> 87.8%), so the drafter's candidate set itself degrades with offset, not just the top-1 choice.
- DFlash paper [F]: only the error-propagation argument for loss weighting; no analysis of capacity per position.
- Future Lens (Pal et al., CoNLL 2023, 2311.04897) [T + K]: probes how well a single GPT-J-6B hidden state predicts tokens N=1..3 ahead; a learned-prompt causal intervention reaches about 48% top-1 for N=1 (as I recall), with accuracy dropping for larger N. Supports "single anchor state carries limited far-future information".
- "Do Language Models Plan for Future Tokens?" (Wu et al., 2404.00859) [T + K]: separates pre-caching from breadcrumbs; finds mostly breadcrumbs in small models, meaning hidden states hold future information mainly as a byproduct of the current step's computation. Supports that far tokens need fresh computation over the intermediate tokens.
- Serial-compute theory [T + K]: constant-depth transformers are limited to TC0-like parallel computation (Merrill and Sabharwal); CoT gives serial steps (Li et al., "Chain of Thought Empowers Transformers...", ICLR 2024); "Pause Tokens Strictly Increase the Expressivity of Constant-Depth Transformers" (2505.21024); "The Serial Scaling Hypothesis" (2507.12549). None is about drafters, but together they give the formal angle: a fixed-depth drafter emulating k serial target steps of depth L each is asking a depth-5 network to do something whose natural serial depth grows like k x L.
- Multi-Token Prediction Needs Registers (2505.10518) and "Predicting the Order of Upcoming Tokens" (2508.19228) [T]: MTP training papers; may report per-offset accuracy decay; not read.
- I found no paper that runs our kind of decomposition (true-prefix redraft vs deeper target features) to separate dependency limits from compute limits. That decomposition looks new.

## 6. DEdit (arXiv 2609.38510, "DEdit: Iterative Draft Editing for Speculative Decoding")

Not read: arXiv returned HTTP 429 on both the html and abs pages, and no mirror page was available. Title and ID confirmed by search [T]. Earlier notes (overlap_branch_redraft.md) also could not read it.

From the title alone: an iterative editing pass over a draft would be a uniform extra-compute or refinement method (like xPress), i.e., closeness 1-2 to compute-per-offset and close to "redrafting". It must be read before claiming novelty, because an editor that re-runs on the suffix after a predicted stop would be very close to our redraft experiment.

## Open questions (what looks unclaimed)

1. Offset-dependent depth in a one-pass block drafter: extra layers, looped layers, or a second pass applied only to block positions beyond some offset, with near positions exiting early. Not found in prior work (pending DEdit, 2609.36173 and 2610.00888).
2. A dependency-vs-compute decomposition of offset decay (our redraft with true prefix vs deeper features). Not found.
3. Scaling curves of per-offset accuracy vs drafter depth at block 16. DFlash 2 only shows positions 0-6, one target, T=0, Recall@1.
4. Cost model: whether extra far-offset depth pays off in wall clock depends on the acceptance of near positions (far positions only matter if near ones are accepted). DFlash's 8L speedup loss and DFlash 2's latency complaint are the baselines to beat.

## To re-check when fetching works
- DEdit full text (2609.38510).
- 2610.00888 and 2609.36173 (possible direct overlap or counter-claim).
- PosS and FastEagle mechanisms and numbers.
- Future Lens exact numbers.
- Per-checkpoint DFlash configs (Qwen3-4B, Coder-30B, LLaMA-3.1-8B) layer counts.
- EAGLE-3 data-scaling figure numbers; Scylla capacity scaling figure.
