# Early Exit and Dynamic Depth for Decoder-Only LLMs (2024 to Sept 2026): Map and Open Gaps

Legend: [V] = verified this session from the arXiv abstract/HTML/PDF (via fetch). [BG] = background knowledge, not re-fetched this session; arXiv ID and headline numbers should be checked before citing. Context numbers from our own measurements are marked [OURS].

## Q1. Cluster map: key papers, speedups vs quality, what each leaves open

### Takeaway
The field splits into (a) exit-trained models (LayerSkip, EE-LLM, EE-Tuning) that make intermediate layers usable, (b) routing/skipping retrofits of pretrained models (MoD, Router-Tuning, D-LLM, FlexiDepth) that report FLOP/layer savings but often no wall-clock gain, (c) KV-cache fixes for skipped layers (state propagation, KV masking, recompute, and 2026 KV-sharing via cheap exit layers), (d) looped/recurrent-depth models where adaptive depth is native and now has a working serving system, and (e) calibration/risk control, which is solid for classification-like outputs but weak for long generation. Evidence from 2026 says modern post-trained, MoE and SSM models are intrinsically less suited for early exit.

### Cited Findings

**Early-exit training / exit-capable models**
- LayerSkip (Elhoushi et al., Meta, arXiv 2404.16710, ACL 2024): layer dropout with increasing rates by depth + early-exit loss with shared LM head; enables self-speculative decoding (draft with early layers, verify with remaining layers, reuse KV of draft layers). Released LayerSkip checkpoints incl. Llama3-8B; code at facebookresearch/LayerSkip — [arXiv](https://arxiv.org/abs/2404.16710), [ACL](https://aclanthology.org/2024.acl-long.681/), [GitHub](https://github.com/facebookresearch/LayerSkip). Reported speedups roughly 1.3x to 2.2x depending on task (e.g., up to ~2.16x on CNN/DM summarization, ~1.8x coding) [BG: numbers from memory, verify in paper].
- EE-LLM (Chen et al., arXiv 2312.04916, ICML 2024): 3D-parallel large-scale early-exit training on Megatron; inference handles KV for exited layers either by KV recomputation batched with the next token or pipeline-based parallelism; code pan-x-c/EE-LLM [BG].
- EE-Tuning (Pan et al., arXiv 2402.00518): parameter-efficient addition of exit heads to a frozen pretrained LLM (Llama-2 up to 70B) with small compute budget [BG].
- CALM (Schuster et al., arXiv 2207.07061, NeurIPS 2022): per-token confident adaptive LM with calibrated local exits, state propagation of exited hidden states to skipped layers' KV; T5 encoder-decoder, background only [BG]. River-LLM classifies CALM with DAT/ELLM as "State Propagation" — [River-LLM](https://arxiv.org/html/2604.18396).

**Missing-KV-cache problem and fixes**
- River-LLM (Shen and Zou, SJTU, arXiv 2604.18396, ACL 2026 per a notes site): taxonomy of existing KV fixes into Batching Recompute (EE-LLM), Mono-Decreasing Exit (SkipDecode), State Propagation (DAT, CALM, ELLM), KV Masking (D-LLM). Proposes a "KV-Shared Exit River": lightweight 4-bit (W4A16) exit layers that produce KV for skipped backbone layers without recomputation; exit decision from state-transition similarity of consecutive hidden states — [arXiv HTML](https://arxiv.org/html/2604.18396), [abs](https://arxiv.org/abs/2604.18396), [ACL2026 note](https://en.papernotes.org/ACL2026/code_intelligence/river-llm_large_language_model_seamless_exit_based_on_kv_share/)
  - Models: Llama3.2-1B, Llama3.1-8B, Phi4-mini, Ministral3-8B. Wall-clock 1.53x to 2.16x on GSM8K, MATH, HumanEval; GSM8K 74.4 to 78.2 vs 78.2 baseline at tau=0.5, near-lossless at tau=0.7 — [River-LLM](https://arxiv.org/html/2604.18396)
  - Limitations stated: only up to 8B; minimal gain on prefill-heavy tasks (MMLU) — [River-LLM](https://arxiv.org/html/2604.18396)
- ADEPT (Yoo et al., arXiv 2601.03700, Jan 2026): token-level early exit in both prefill and generation, "decoupling sequential dependencies in skipped layers" so KV need not be computed for all layers; claims up to 25% efficiency improvement in generation, 4x speedup in downstream classification, up to 45% performance improvement (abstract-level; models not in abstract) — [arXiv](https://arxiv.org/abs/2601.03700)
- DREX (arXiv 2512.15705, Dec 2025): dynamic rebatching at each exit point (exiting requests proceed, others buffered and regrouped), memory-efficient state-copying for missing KV, copy-free rebatching buffer, SLA-aware scheduler that predicts whether rebatching is profitable; throughput +2 to 12% vs baselines; "eliminates involuntary exits" — [arXiv](https://arxiv.org/abs/2512.15705)
- "Robust and Efficient Early Exit for LLMs: Mitigating KV Cache Loss and Enhancing Exit Stability" (Springer chapter, 2025; details not fetched) — [Springer](https://link.springer.com/chapter/10.1007/978-981-95-1233-1_7)
- FlexiDepth sidesteps the problem by still computing K/V for skipped layers (skips query/attention output and FFN) — [FlexiDepth](https://arxiv.org/abs/2503.23798)
- TIDE sidesteps it by running all layers and just choosing which layer's output to emit (no real skipping) — [TIDE](https://arxiv.org/pdf/2603.21365)

**Layer skipping / dynamic depth in pretrained LLMs**
- Mixture-of-Depths (Raposo et al., arXiv 2404.02258, 2024): top-k token routing around blocks with fixed capacity, trained from scratch; static compute graph [BG]. Cited as MoD routing in the looped-LM batching paper — [CDB](https://arxiv.org/html/2608.09444v1)
- Router-Tuning / MindSkip (He et al., arXiv 2410.13184, EMNLP 2025): only a router is fine-tuned on a small dataset; "Attention with Dynamic Depths" skips attention; reports 21% speedup with 0.2% performance drop — [arXiv](https://arxiv.org/abs/2410.13184), [GitHub](https://github.com/CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths)
- FlexiDepth (Luo, Wang, Yan, arXiv 2503.23798, 2025): plug-in router + adapter per layer on frozen Llama-3-8B-Instruct; skips ~8 of 32 layers with 100.7% benchmark retention; also Llama-2-13B-Instruct and Qwen-2.5-3B-Instruct. Authors explicitly state "our implementation does not lead to improved throughput on the existing GPU hardware" — [arXiv](https://arxiv.org/abs/2503.23798), [project](https://luoxuan-cs.github.io/flexidepth/)
- vSkipper (Da, Ferhatosmanoglu, Kalyvianaki, Cambridge, arXiv 2609.37062, 29 Sep 2026): FlexiDepth's standard generation loop decodes 14.6 to 21.0% slower than the base model despite skipping 8/32 layers. vSkipper adds a virtualization layer in SGLang; on Llama-3-8B (A100-80GB, also H100/A6000) at the load knee: -36.8% mean E2E latency on GSM8K, -13.6% on BBH; under saturation +11.3% (GSM8K) and +7.4% (BBH) request throughput; routed mode only profitable above ~200k resident tokens; prefill routing needs >=1,536 prompt tokens — [arXiv HTML](https://arxiv.org/html/2609.37062), [abs](https://arxiv.org/abs/2609.37062)
- D-LLM (Jiang et al., NeurIPS 2024): per-token dynamic layer allocation with KV masking/eviction for skipped layers [BG]; named as the "KV Masking" category in [River-LLM](https://arxiv.org/html/2604.18396)
- SkipDecode (Del Corro et al., arXiv 2307.02628, 2023): monotonically decreasing exit depth across positions in a sequence so skipped-layer KV is never needed later and batching works with a unified exit point [BG]; listed as "Mono-Decreasing Exit" in [River-LLM](https://arxiv.org/html/2604.18396)
- FFN-SkipLLM (Jaiswal et al., arXiv 2404.03865, 2024): skips FFN blocks (not attention) by input-adaptive cosine similarity, which avoids the KV problem since attention KV is still computed; reports ~25 to 30% FFN blocks skipped with marginal quality change [BG; the arXiv 2404.03865 link appeared in search results] — [arXiv](https://arxiv.org/html/2404.03865v1)
- "Skip-It? Theoretical Conditions for Layer Skipping in Vision-Language Models" (arXiv 2509.25584): VLM-focused theory on when layers are skippable (information-theoretic redundancy); not decoder-only text LLMs, relevant as theory background — [arXiv](https://arxiv.org/abs/2509.25584), [OpenReview](https://openreview.net/forum?id=XO9XUU6n8M)
- "Skip a Layer or Loop It? Test-Time Depth Adaptation of Pretrained LLMs" (CoLa, arXiv 2507.07996, 2025): MCTS search over per-sample layer compositions (skip/repeat) of a frozen LLM; shows many samples solvable with shorter paths [BG; title appeared in search noise only]
- LiteLoRA-style skipping: I found no reliable source for a method named "LiteLoRA" in this context; not verified.
- Self-speculative layer-skip drafting without training, e.g. Draft&Verify (arXiv 2309.08168), SWIFT (arXiv 2410.06916), Kangaroo (arXiv 2404.18911) [BG]. These are the practical wall-clock route for skipping in dense models because verification guarantees lossless output.

**Looped / recurrent-depth models**
- Ouro / LoopLM (ByteDance, "Scaling Latent Reasoning via Looped Language Models", arXiv 2510.25741, Oct 2025): 1.4B and 2.6B looped models with learned exit gates — [Pith](https://pith.science/paper/2510.25741), [HF](https://huggingface.co/ByteDance/Ouro-1.4B)
- Huginn (Geiping et al., arXiv 2502.05171, 2025): 3.5B recurrent-depth model, training-free convergence-based exit [BG]; cited in [CDB](https://arxiv.org/html/2608.09444v1)
- Continuous Depth Batching (Schwethelm, Rückert, Kaissis, arXiv 2608.09444, 10 Aug 2026): first end-to-end serving for per-token adaptive depth in looped LMs (Ouro-1.4B, Huginn-3.5B); queue per stage (prefill/prelude/core/coda), refill freed slots, depth-aware KV cache ("last-exited" copy or "shared" overwrite that changes attention semantics), async CPU prep with lookahead gates cuts GPU idle 40% to ~0.67%. Results: 1.5x to 1.9x offline throughput, 45 to 90% lower online latency, 94 to 99% of theoretical FLOP bound. Notes Bae et al. (2025a) proposed the concept without implementation — [arXiv](https://arxiv.org/html/2608.09444v1)
- Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates (Popescu, Sáez de Ocáriz Borde, Liò, arXiv 2607.20519, Jul 2026): training a gate entangles readout learning with depth supervision of the trajectory; fixed-prior trained trajectories + entropy/confidence readouts match or beat learned gates; on MANO 99% accuracy at ~1.5 loops vs 4+ loops for learned gates; Ouro's pretrained gates "not uniformly Pareto-optimal" on MMLU/ARC/HellaSwag/CSQA — [arXiv](https://arxiv.org/html/2607.20519v1)

**Exit criteria and calibration**
- Fast yet Safe: Early-Exiting with Risk Control (Jazbec et al., arXiv 2405.20915, 2024): risk control (Learn-then-Test style) for tuning exit thresholds with guarantees — [arXiv](https://arxiv.org/html/2405.20915v2)
- Controlling the Risk of Corrupted Contexts via Early-Exiting (Wynn, Jazbec, ..., Nalisnick, arXiv 2510.02480, ICML 2026): LTT with context-aware loss; LLaMA-3-8B, LLaMA-2-7B and their LayerSkip variants; ~50% speedup, up to 81.8% fewer layers evaluated; authors note the per-token threshold "may be less effective for longer-generation tasks" — [arXiv](https://arxiv.org/html/2510.02480)
- Conformal Risk Control (Angelopoulos et al., arXiv 2208.02814) is the underlying tool — [arXiv](https://arxiv.org/abs/2208.02814)
- A risk-controlled early exit for diffusion LMs exists as an auto-generated FARS/Analemma draft (not peer reviewed, treat as low-reliability) — [PDF](https://lemma-public-asset.analemma.ai/online/fars/live/live_live_20260213/idea_808c134d-cfc5-4140-993a-632cfdc4dc0b/main.pdf)
- TIDE (Jaber and Jaber, RightNow AI, arXiv 2603.21365, Mar 2026): learned exit classifiers (2-layer MLP, bottleneck 128) trained on frozen hidden states with cosine-to-final as convergence label; DeepSeek-R1-Distill-8B and Qwen3-8B; tau=0.98 concentrates exits at penultimate checkpoint — [PDF](https://arxiv.org/pdf/2603.21365)
- River-LLM uses consecutive hidden-state transition similarity as exit signal — [River-LLM](https://arxiv.org/html/2604.18396)

**Serving systems with exits**
- HELIOS (Kumar et al., arXiv 2504.10724, MLSys 2026): adaptive model and early-exit selection for LLM serving — [arXiv](https://arxiv.org/abs/2504.10724), [PDF](https://lca.ece.utexas.edu/pubs/kumar_mlsys26.pdf) (details not fetched)
- DREX and vSkipper above.

**Decline of EE suitability**
- The Diminishing Returns of Early-Exit Decoding in Modern LLMs (Wei, Du, Yu, Tiwari, Li, Xu, Wang; arXiv 2603.23701, 24 Mar 2026): defines Early-Exit Adaptability Score EAS = mean over layers of A_l = S_l^alpha * w_l^(1-alpha), S_l = layer-to-final logit similarity, w_l = (L-l)/L skip ratio; oracle early-exit benchmark in OpenCompass — [PDF](https://arxiv.org/pdf/2603.23701), [abs](https://arxiv.org/abs/2603.23701)
  - EAS: Llama2-7B 0.52, Llama3-8B 0.46, Qwen2-7B 0.36, Qwen3-8B 0.51, GPT-OSS-20B 0.59. Note this is not monotone across generations (Qwen3 > Qwen2), so the "diminishing trend" is a family-level claim with exceptions — [PDF](https://arxiv.org/pdf/2603.23701)
  - Max skip ratio under 5% accuracy loss only 0 to 7.26% (oracle) in Table 1; no wall-clock data — [PDF](https://arxiv.org/pdf/2603.23701)
  - Post-trained models show "delayed logit alignment" vs base; MoE (Qwen3-30B-A3B) "reduces cross-layer alignment"; SSMs (Mamba-130M, Mamba2-7B, Mamba-Codestral-7B) "least suitable"; larger (>20B) models more suitable — [PDF](https://arxiv.org/pdf/2603.23701)
  - Limitations: training-dynamics analysis only on Pythia checkpoints; future work suggested: "early-exit-aware tuning techniques" and controlled small-scale pretraining — [PDF](https://arxiv.org/pdf/2603.23701)
- [OURS] logit-lens agreement with final layer at half depth: Qwen3-8B disagrees 96%, Llama3-8B 98.5%, LayerSkip-Llama3-8B 13%. This is consistent with 2603.23701 that vanilla models are poorly suited and that EE-aware training (LayerSkip) changes the picture drastically.

**Early exit in reasoning models (separate topic)**
- CoT early stopping (stop generating reasoning tokens) is a different axis from layer-depth exit; e.g., DiffAdapt (arXiv 2510.19669) is difficulty-adaptive token budgeting, not layer exit — [arXiv](https://arxiv.org/html/2510.19669v2). TIDE is the only per-layer exit work found that evaluates on a reasoning-distilled model (DeepSeek-R1-Distill-8B) — [TIDE](https://arxiv.org/pdf/2603.21365)

### Inferences
- The pretrained-retrofit line (FlexiDepth, Router-Tuning) mostly reports layer/FLOP savings; the only honest wall-clock numbers either come from self-speculation (lossless) or from dedicated serving systems published in 2025-2026 (DREX, vSkipper, CDB).
- The KV problem is now handled by five strategy families (recompute, monotone-decreasing, state propagation/copying, masking, and generate-KV-via-cheap-layers), plus "compute KV anyway" (FlexiDepth, FFN-only skipping). None of these come with a quality guarantee on the effect of approximate KV on future tokens.

### Gaps
- Exact LayerSkip, EE-LLM, D-LLM, MoD, SkipDecode, FFN-SkipLLM numbers were not re-fetched this session.
- "LiteLoRA-style skipping" could not be identified; the user may be referring to a paper I did not find.

## Q2. Real wall-clock state at batch 1 and under batching

### Takeaway
At batch 1, real early exit or skipping gives about 1.5x to 2.2x only with exit-trained or add-on exit modules plus careful KV handling (River-LLM, LayerSkip self-spec); naive retrofits can be slower than the base model. Under batching, gains shrink to single-digit or low double-digit percent in dense LLMs unless a dedicated scheduler is used; looped LMs are the exception (1.5x to 1.9x throughput with CDB).

### Cited Findings
- FlexiDepth: no throughput improvement on GPUs; vSkipper measured its generation loop 14.6 to 21.0% slower than base — [FlexiDepth](https://arxiv.org/abs/2503.23798), [vSkipper](https://arxiv.org/html/2609.37062)
- vSkipper serving gains on Llama-3-8B only appear at load: -36.8%/-13.6% E2E latency at the knee, +11.3%/+7.4% throughput at saturation; profitable only above ~200k resident tokens — [vSkipper](https://arxiv.org/html/2609.37062)
- DREX: +2 to 12% throughput with dynamic rebatching — [DREX](https://arxiv.org/abs/2512.15705)
- TIDE (no real skipping): +6.6% at batch 1 (R1-Distill-8B); at batch 8, +8.1% Qwen3-8B but -16.3% for R1-Distill; output_hidden_states overhead scales poorly — [TIDE](https://arxiv.org/pdf/2603.21365)
- River-LLM: 1.53x to 2.16x wall-clock (batch size not confirmed in fetched summary, likely batch 1); speedup ~10% below full-quantization baselines; minimal on prefill-heavy tasks — [River-LLM](https://arxiv.org/html/2604.18396)
- Router-Tuning: 21% speedup, 0.2% drop (setting/batch not confirmed) — [Router-Tuning](https://arxiv.org/abs/2410.13184)
- CDB for looped LMs: 1.5x to 1.9x offline throughput, 94 to 99% of FLOP bound — [CDB](https://arxiv.org/html/2608.09444v1)
- 2603.23701 gives oracle skip ratios only (<=7.26% under 5% loss), no wall-clock — [PDF](https://arxiv.org/pdf/2603.23701)

### Inferences
- For dense decoder-only LLMs, early exit as a standalone inference mode is being eaten by self-speculative decoding (lossless) and by quantization; River-LLM's own comparison to full quantization supports this.
- Batching heterogeneity is now a systems problem with three published answers (DREX rebatching, vSkipper virtualization, CDB queues); none targets self-speculative early-exit drafting in batched serving.

### Gaps
- No source found that reports LayerSkip self-speculative decoding throughput at batch sizes > 1 in a real serving engine (vLLM/SGLang). Worth a direct search.

## Q3. Acknowledged unsolved problems

### Takeaway
All five listed problems are explicitly acknowledged in 2025-2026 papers; KV for skipped layers and batching have partial systems fixes, while post-training degradation of exits and EE for MoE/SSM have diagnosis papers but essentially no methods.

### Cited Findings
- KV cache for skipped layers: four legacy strategies each with drawbacks, River-LLM's fix limited to <=8B and weak on prefill-heavy tasks — [River-LLM](https://arxiv.org/html/2604.18396); CDB's shared cache "modifies attention semantics" — [CDB](https://arxiv.org/html/2608.09444v1)
- Exit calibration for generation: risk-control per-token thresholds "may be less effective for longer-generation tasks" — [2510.02480](https://arxiv.org/html/2510.02480); learned looped-LM gates not Pareto-optimal, entropy readouts competitive — [2607.20519](https://arxiv.org/html/2607.20519v1); TIDE's conservative threshold pushes exits to penultimate layer — [TIDE](https://arxiv.org/pdf/2603.21365)
- Batching with heterogeneous depths: DREX, vSkipper, CDB (above); vSkipper profitable only at high load — [vSkipper](https://arxiv.org/html/2609.37062)
- Post-training degrading exits: "post-trained models generally exhibit delayed logit alignment"; future work calls for "early-exit-aware tuning techniques" — [2603.23701](https://arxiv.org/pdf/2603.23701)
- MoE/SSM: MoE reduces cross-layer alignment; SSMs least suitable — [2603.23701](https://arxiv.org/pdf/2603.23701). No MoE-specific early-exit method surfaced in search (results were generic MoE expert-skipping, e.g. ACL 2024 expert pruning/skipping) — [ACL 2024](https://aclanthology.org/2024.acl-long.334.pdf)

### Inferences
- The post-training gap is the most open: the diagnosis exists (2603.23701), LayerSkip shows EE-aware pretraining/CPT fixes it, but no paper found studies how SFT/RL/LoRA on an exit-trained model erodes exits, or how to regularize post-training to preserve them. [OURS] ~10% self-speculation speed loss after LoRA that does not accumulate with more steps is a direct data point on this gap.

### Gaps
- No quantitative study found on RL post-training (GRPO etc.) effects on intermediate-layer exits.

## Q4. What can be studied with 8B models on 1-4 A100s

### Takeaway
Most gap directions are feasible at 8B with LoRA or head-only training: recent papers themselves run at 1B to 8B on single A100s (River-LLM, vSkipper, TIDE, 2510.02480 with LayerSkip-Llama variants).

### Cited Findings
- River-LLM trained/evaluated at Llama3.1-8B and Ministral3-8B — [River-LLM](https://arxiv.org/html/2604.18396)
- vSkipper: Llama-3-8B FlexiDepth checkpoint on A100-80GB — [vSkipper](https://arxiv.org/html/2609.37062)
- TIDE: routers of d*128+128 params on frozen Qwen3-8B / R1-Distill-8B — [TIDE](https://arxiv.org/pdf/2603.21365)
- 2510.02480 used LLaMA-3-8B and LayerSkip variants — [2510.02480](https://arxiv.org/html/2510.02480)
- FlexiDepth: frozen backbone + router/adapter on Llama-3-8B-Instruct — [FlexiDepth](https://arxiv.org/abs/2503.23798)
- Router-Tuning: router-only fine-tuning on a small dataset — [Router-Tuning](https://arxiv.org/abs/2410.13184)
- EE-Tuning: exit heads added to frozen LLMs with small budget — [BG, arXiv 2402.00518]

### Inferences (feasible studies)
1. EE-preserving post-training: LoRA SFT/RL on LayerSkip-Llama3-8B with an auxiliary early-exit loss or layer-dropout during LoRA; measure half-depth agreement and self-spec acceptance before/after. Directly extends [OURS] ~10% loss finding and 2603.23701's call.
2. Cheap "EE-ification" of Qwen3-8B via LoRA + early-exit loss (continued training at small token budgets), measuring how many tokens are needed to move half-depth disagreement from 96% toward LayerSkip's 13%.
3. Calibrated exits for long generation: sequence-level risk control (e.g., bound on final-answer accuracy or on divergence from full-depth greedy output) rather than per-token thresholds, on LayerSkip-Llama3-8B.
4. Approximate-KV error analysis: compare state-copy, KV masking, River-style cheap KV layers on attention-output error over long contexts at 8B.
5. MoE: Qwen3-30B-A3B fits on 1-2 A100-80GB in bf16 (~60GB weights), so logit-lens/exit analysis is feasible; training is harder. SSM hybrids are small enough too.

### Gaps
- No source directly reports the token budget needed to make a vanilla 8B model exit-capable via LoRA.

## Q5. Novelty check for candidate gaps

### Takeaway
Closest work found for each candidate; none found that directly covers "early-exit preservation under post-training" or "sequence-level risk control for generative early exit." Limited search budget; treat as a first pass.

### Cited Findings
- "Preserving early exit ability after fine-tuning / LoRA on LayerSkip": search returned only LayerSkip itself and its derivatives; no paper found on exit-preserving fine-tuning — [LayerSkip](https://arxiv.org/abs/2404.16710). Closest diagnosis is 2603.23701 (post-training delays logit alignment) — [PDF](https://arxiv.org/pdf/2603.23701)
- "Risk control for generative early exit": closest are Fast yet Safe (2405.20915) and 2510.02480, the latter explicitly limited for long generation — [2405.20915](https://arxiv.org/html/2405.20915v2), [2510.02480](https://arxiv.org/html/2510.02480)
- "KV cache for skipped layers": crowded in 2025-2026 (River-LLM, ADEPT, DREX, CDB) — see Q1.
- "Batched heterogeneous depth serving": crowded (DREX Dec 2025, vSkipper Sep 2026, CDB Aug 2026, HELIOS MLSys 2026).
- "Learned halting vs readouts": 2607.20519 already shows simple readouts rival learned gates in looped models — [2607.20519](https://arxiv.org/html/2607.20519v1)
- "Early exit for MoE LLMs": no method paper found; only the diagnosis in 2603.23701.

### Inferences
- Least crowded and cheapest to study with our setup: (1) EE-preserving post-training on LayerSkip-8B, (2) sequence-level calibrated exit for generation, (3) early exit in MoE. KV-fix and batching directions are now crowded with systems papers from the last 10 months.
- Self-speculative decoding angle connects directly to the spec-hierarchy project: exit quality mainly matters as draft acceptance rate, where losslessness removes the calibration problem.

### Gaps
- Direct phrase searches not run for: "early-exit-aware fine-tuning", "RL post-training early exit", "early exit Mixture-of-Experts routing layer", "LayerSkip batched serving vLLM". Recommend running these before committing.
