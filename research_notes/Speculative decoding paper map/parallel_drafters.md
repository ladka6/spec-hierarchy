# Parallel / block / diffusion drafters for speculative decoding (paper map, as of 2026-10-03)

Reading notes. Network access: arxiv.org was only reachable through WebFetch, which returns a model-summarised view, not raw text. Numbers below are as the extractor returned them; anything that feeds a written claim should be checked against the PDF. Some entries reuse the team's earlier notes (`research_notes/critique/dflash.md`, `critique/ddtree.md`, `SD and early exit research gaps/drafter_side.md`); these are tagged [prior notes]. Tags: [abs] = only abstract/title seen; [html] = arXiv HTML read via extractor; [proj] = project page; [code] = repo read locally.

Baseline shorthand: "vs DFlash" means the paper reports DFlash as a baseline. T = sampling temperature. bs = batch size.

## Q1. Curated list (25 papers): per-paper record

### Takeaway
The line runs from Blockwise Parallel Decoding (2018) through PARD and diffusion-LM drafters (DiffuSpec, SpecDiff-2, DEER, 2025) to DFlash (Feb 2026, ICML 2026). DFlash then became the de facto backbone and baseline. Since May 2026 roughly 15 follow-ups fall into four groups: (a) restoring intra-block dependency (Domino, TreeFlash, JetSpec, DSpark, DBLAST, xPress, DARTree, D2SD); (b) trees over block marginals (DDTree, BASTION, CaDDTree, GRAFT); (c) serving/scheduling (DSpark, AdaFlash, DPara, LongSpark); (d) training objectives (PARD-2, AdaFlash, DBLAST, DFlow). Almost all evaluate on Qwen3-4B/8B (sometimes 14B or Coder-30B-A3B), and most use batch 1 with the HF Transformers backend.

### Cited Findings

**Foundations and early parallel drafters**

1. **Blockwise Parallel Decoding for Deep Autoregressive Models.** Stern, Shazeer, Uszkoreit. arXiv 1811.03115. NeurIPS 2018. Nov 2018. [abs]
   - Method: extra heads predict k future tokens in parallel. The base model then verifies them, and decoding backs off to the longest validated prefix.
   - Result: up to 2x fewer iterations than greedy with no quality loss, or up to 7x with a slight loss; up to 4x wall-clock. Tasks: MT and image super-resolution with Transformers (not LLMs). — [arXiv](https://arxiv.org/abs/1811.03115)
   - Code: none found. Stated limitations: none in the abstract (full text not read).

2. **PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation.** An et al. (AMD). arXiv 2504.18583. arXiv only (v4 Nov 2025). Apr 2025. [html]
   - Method: adapts a small AR model into a target-independent parallel drafter using mask tokens. Conditional Drop-token (COD) training is 3x more efficient.
   - Result: LLaMA3.1-8B 4.08x (311.5 tok/s) and Qwen2.5-7B 4.87x. Setting: single A100-40GB, bs 1 (tested up to 16), T=0, optimized Transformers and vLLM. Baselines: vanilla SD and EAGLE. The abstract says 3.67x / 264.88 tok/s on LLaMA3.1-8B, which is "1.15x faster than EAGLE-3". The two figures differ across versions. — [arXiv abs](https://arxiv.org/abs/2504.18583), [HTML v3](https://arxiv.org/html/2504.18583v3)
   - Code: https://github.com/AMD-AIG-AIMA/PARD
   - Stated limitation: "As the batch size increases, the bottleneck shifts from memory-bound to compute-bound... PARD achieves a speedup of 1.17x to 3.06x." — [HTML v3](https://arxiv.org/html/2504.18583v3)

3. **DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding.** Li et al. arXiv 2510.02358. Findings of ACL 2026. Oct 2025. [abs + ACL page]
   - Method: a pretrained (not target-aligned) diffusion LM drafts a token lattice. Causal-consistency path search extracts a sequence from it, and an adaptive draft-length controller sets its length.
   - Result: "up to 3x wall-clock speedup". Detailed settings not extracted. — [arXiv](https://arxiv.org/abs/2510.02358), [ACL Anthology](https://aclanthology.org/2026.findings-acl.1048/)
   - Code: none found. Stated limitations: not retrieved.

4. **SpecDiff-2: Scaling Diffusion Drafter Alignment for Faster Speculative Decoding.** Sandler, Christopher, Hartvigsen, Fioretto. arXiv 2511.00606. arXiv only. Nov 2025. [abs + html partial]
   - Method: a discrete-diffusion drafter with streak-distillation (a train-time objective for long accepted streaks) and test-time self-selection over several drafts sampled from the marginals.
   - Result: +55% average tok/s over prior baselines and up to 5.5x average speedup over AR. Targets: Qwen2.5-14B/72B, Llama 13B/70B. Hardware and T are in the appendix (not extracted). — [arXiv](https://arxiv.org/abs/2511.00606), [HTML v2](https://arxiv.org/html/2511.00606v2)
   - Code: none found. The paper has a "Future Work and Limitations" section (Sec. 9), but its text was not retrieved.

5. **DEER: Draft with Diffusion, Verify with Autoregressive Models.** Cheng et al. arXiv 2512.15176. arXiv only. Dec 2025. [abs + proj]
   - Method: a dLLM drafter aligned to the AR target with a two-stage training pipeline.
   - Result: up to 5.54x on HumanEval with Qwen3-30B-A3B; acceptance up to 32 tokens, about 5 on average. Setting: A100, T=0. Compared to EAGLE-3. — [arXiv](https://arxiv.org/abs/2512.15176), [project page](https://czc726.github.io/DEER/)
   - Code: https://github.com/czc726/DEER (checkpoints "coming soon"). Stated limitations: none stated.

6. **P-EAGLE: Parallel-Drafting EAGLE with Scalable Training.** Hui et al. (AWS). arXiv 2602.01469. arXiv only. Feb 2026. [abs]
   - Method: turns EAGLE into a parallel multi-token predictor via a learnable shared hidden state. Long-context training is made feasible through attention-mask precomputation and sequence partitioning.
   - Result: 1.10-1.36x over AR EAGLE-3 on GPT-OSS 120B/20B and Qwen3-Coder-30B, in vLLM. Hardware, bs and T not extracted. — [arXiv](https://arxiv.org/abs/2602.01469), [vLLM blog](https://vllm.ai/blog/2026-03-13-p-eagle)
   - Code: in vLLM; no standalone repo found. Stated limitations: not retrieved.

**DFlash and the tree line**

7. **DFlash: Block Diffusion for Flash Speculative Decoding.** Chen, Liang, Liu (z-lab). arXiv 2602.06036. ICML 2026, per the [z-lab project page](https://z-lab.ai/projects/dflash/). Feb 2026. [html, prior notes, code]
   - Method: a 5-layer block-diffusion drafter (block 16, one denoising step). Target hidden features from 5 layers are injected into the KV of every draft layer.
   - Result: Qwen3-8B averages are 4.86x / τ 6.49 at T=0 and 4.03x / 5.48 at T=1. EAGLE-3(16) gets 1.76x and EAGLE-3(60) 2.02x. Setting: H200, bs 1, Transformers, max 2048 tokens, thinking off. In SGLang on one B200, Qwen3-8B Math500 drops from 5.1x at concurrency 1 to 2.8x at concurrency 32. — [prior notes](../critique/dflash.md), [arXiv](https://arxiv.org/abs/2602.06036)
   - Code: https://github.com/z-lab/dflash; also integrated in SGLang Spec V2 and vLLM. — [LMSYS blog](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
   - Stated limitations: no Limitations section. The only forward-looking line: large blocks raise verify cost under compute-bound settings, and "We leave adaptive block-size scheduling to future work." — [prior notes](../critique/dflash.md)
   - **DFlash 2: no paper or arXiv ID found.** The z-lab/dflash README links a blog (inco.ai/blog/dflash2, which could not be fetched) and checkpoints for Muse-Glimmer-30B and Qwen3.8-27B. The local code shows `DFlash2DraftModel` with a grouped causal conv and a `CandidateSelector`, a first-order Markov rerank of top-k candidates. — [code] /home/claude/dflash/README.md, [HF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)

8. **DDTree: Accelerating Speculative Decoding with Block Diffusion Draft Trees.** Ringel, Romano. arXiv 2604.12989. arXiv only. Apr 2026. [html, prior notes, code]
   - Method: a best-first heap builds the top-B prefix tree from DFlash's per-position marginals. The tree is verified in one pass with an ancestor mask.
   - Result vs DFlash (T=0): Qwen3-8B MATH-500 goes from 5.56x / τ 7.79 to 7.52x / 10.73, and wins all 60 cells. Setting: 8xH200 data-parallel, bs 1, T∈{0,1}, HF/SDPA. The budget was chosen per cell on the test set (oracle). — [prior notes](../critique/ddtree.md), [arXiv](https://arxiv.org/html/2604.12989v1)
   - Code: https://github.com/liranringel/ddtree
   - Stated limitations: no limitations section found. Remark 1 says optimality holds only for the factorized surrogate Q, not for the target.

9. **BASTION.** KAIST/SAIT. arXiv 2605.29727. arXiv only. May 2026. [prior notes, html]
   - Method: query-dependent trees from block-diffusion marginals. The path-confidence product serves as the acceptance surrogate, and a roofline latency model drives tree growth.
   - Result: 6.61x over AR and 1.39x over DFlash, at T=0 and T=1. — [arXiv](https://arxiv.org/html/2605.29727v1)
   - Code: not found.
   - Stated limitations: batch size 1 only; needs stable runtime profiles for calibration. Model and GPU details not re-extracted.

10. **CaDDTree.** arXiv 2606.01813. arXiv only. Jun 2026. [prior notes, html]
    - Method: picks the per-round tree budget to maximise tok/s, using an offline latency profile (time per token is unimodal in budget) and drafter confidence.
    - Result: matches DDTree's oracle budget automatically, e.g. Qwen3-4B MATH-500 at 4.53 ms/tok for both, τ 10.37 vs 10.35. Setting: A100/A800, Qwen3-4B/8B. T=1 appears only in the appendix. — [arXiv](https://arxiv.org/html/2606.01813)
    - Code: not found. Stated limitations: none stated.

11. **GRAFT: Adaptive DLM-Based Draft Tree Construction with Target-Distilled Edge Scoring.** Ye et al. arXiv 2608.20375. Uses the AAAI template (2027 copyright); acceptance not confirmed. Aug 2026. [html]
    - Method: target-distilled edge scores fix parent-child compatibility in diffusion draft trees, and a state-aware budget sets tree size.
    - Result: 2.13-6.36x over AR. Setting: Qwen3-4B/8B and Coder-30B, A100-80GB, bs 1. Baselines: EAGLE-3, OPT-Tree, DFlash, DDTree. Its MAT on Qwen3-8B is about 8.06 vs DDTree's 8.18; the paper argues it wins on throughput, not length. — [arXiv](https://arxiv.org/html/2608.20375)
    - Code: not stated.
    - Stated limitation: "GRAFT still executes drafting, tree construction, and target verification sequentially at each round."

**Restoring intra-block dependency**

12. **Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding.** Huang et al. arXiv 2605.29707. arXiv only. May 28, 2026. [html]
    - Method: a DFlash backbone plus a GRU causal encoder and low-rank residual logit correction (+5.3% params).
    - Result: +16.6% τ and +12.3% end-to-end vs DFlash; up to 5.49x (Transformers) and 5.8x throughput (SGLang). Setting: Qwen3-4B/8B, A100, T=0, block 16. Baselines: EAGLE-3, DART, DFlash. — [arXiv](https://arxiv.org/html/2605.29707v1)
    - Code: https://github.com/jianuo-huang/Domino
    - Stated limitation: gains need long accepted prefixes; the correction matters less when many positions are rejected.

13. **TreeFlash: Parallel AR-Approximation for Faster Speculative Decoding.** Rheinboldt, Berdoz, Wattenhofer (ETH). arXiv 2606.03819. arXiv only. Jun 2026. [html]
    - Method: a SwiGLU/MLP layer conditioned on the drafter hidden state and the previous token approximates AR conditionals inside a tree. Decoding stays O(1) in depth.
    - Result: +12.4% block efficiency over DDTree at B=64, and +9.1% speedup. Setting: Qwen3-4B/8B and Coder-30B, GH200, bs 1, T∈{0,1}, PyTorch SDPA. — [arXiv](https://arxiv.org/html/2606.03819v1)
    - Code: https://github.com/ETH-DISCO/TreeFlash
    - Stated limitations: "single-batch setting using SDPA attention"; production optimizations "can conflict with large tree sizes"; "evaluated exclusively on Qwen-family models".

14. **D²SD: Accelerating Speculative Decoding with Dual Diffusion Draft Models.** Zhang et al. (PKU, Tsinghua, HKUST, UIUC, Ant). arXiv 2606.04446. arXiv only. Jun 2026. [html]
    - Method: drafter-1 confidence locates likely rejection points. A variable-prefix drafter-2 re-anchors at the top-K of those positions, and all branches are verified together with cascade attention.
    - Result: Qwen3-8B (γ=16, K=4, T=0) gets 4.98x / τ 7.05 vs DFlash 4.16x / 5.31. GPT-OSS-20B gets 6.15x vs 3.53x. At T=1, Qwen3-8B gets 4.01x. Setting: H200. Baselines: DFlash and EAGLE-3. — [arXiv](https://arxiv.org/html/2606.04446)
    - Code: https://github.com/catnanami/D2-SD
    - Stated limitations: no section. The ablation shows a third stage adds +0.62 τ but costs -4% speed.
    - Note: the extractor wrongly labelled this paper "arXiv:2602.06036", which is DFlash's ID. Its ID is 2606.04446, taken from the URL.

15. **JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting.** Hu et al. (incl. Hao Zhang). arXiv 2606.18394. arXiv only. v1 Jun 16, v4 Sep 27, 2026. [abs]
    - Method: a causal parallel draft head over fused hidden states gives branch-wise causal conditioning in a single forward pass.
    - Result: up to 9.64x on MATH-500 and 4.58x on chat. Setting: H100, Qwen3 dense and MoE. Baselines, bs and T not extracted (HTML fetch refused). — [arXiv](https://arxiv.org/abs/2606.18394)
    - Code: https://github.com/hao-ai-lab/JetSpec. Stated limitations: not retrieved.

16. **DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation.** Cheng et al. (PKU + DeepSeek-AI). arXiv 2607.05147. arXiv only. Jul 6, 2026. [html]
    - Method: a DFlash-style parallel backbone plus a light sequential module (semi-AR). A calibrated confidence head and a load-aware scheduler choose the verify length.
    - Result: τ +16.3/18.4/18.3% vs DFlash and +30.9/26.7/30.0% vs EAGLE-3 on Qwen3-4B/8B/14B. In DeepSeek-V4 production: 60-85% per-user speedup at matched throughput. — [arXiv](https://arxiv.org/html/2607.05147v1)
    - Code: https://github.com/deepseek-ai/DeepSpec
    - Stated limitation: under strict interactivity, "cannot eliminate the tension between throughput and latency". Also depends on throughput profiling.

17. **xPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding.** Wang et al. (IBM/UIUC). arXiv 2608.02438. arXiv only. Aug 3, 2026. [html]
    - Method: a lightweight causal refiner re-ranks diffusion-draft candidates through parallel Jacobi iterations.
    - Result: τ about +30% on average (up to +56%); end-to-end about 1.3x at T=0 and 1.46x at T=1 over DFlash. Setting: Qwen3-8B, H200, 7 benchmarks. Baselines: DFlash and a Markov head. — [arXiv](https://arxiv.org/html/2608.02438)
    - Code: https://github.com/Supercomputing-System-AI-Lab/xPress; vLLM PR #54448. — [search result](https://github.com/vllm-project/vllm/pull/54448)
    - Stated limitations: no section. The paper notes tree methods "fade quickly at large batch sizes".

18. **DBLast: Dependent Block Drafting for Stochastic Speculative Decoding.** Karimi, Gao, Hassanpour. arXiv 2608.05448. arXiv only. Aug 5, 2026. [html]
    - Method: a categorical latent mixture across block positions, trained with an acceptance-oriented objective, targeting T>0.
    - Result: over 12% macro τ gain over DFlash on Qwen3-8B in the high-entropy regime. Settings: T∈{0.7, 1.0, 1.5}, top-p∈{0.8, 0.95}, block 15. Benchmarks: GSM8K, MT-Bench, HumanEval, creative writing. GPU not extracted. — [arXiv](https://arxiv.org/html/2608.05448)
    - Code: not found.
    - Stated limitations: (1) effects of training-data stochasticity early in training; (2) the objective is a surrogate, not unbiased; (3) greedy-branch training proposals do not match stochastic inference.

19. **DARTree: Speculative Diffusion Decoding with Autoregressive Draft Trees.** Li, Luo, Shang, Shen (MBZUAI). arXiv 2608.13524. arXiv only. Aug 13, 2026. [html]
    - Method: extends pretrained AR correction heads from chains to trees through depth-wise parallel batching, followed by deferred best-first pruning.
    - Result: up to 9.73x over AR and τ up to 12.97 (+98.6% vs DFlash, +27.9% vs Domino). Setting: Qwen3-4B/8B, T∈{0,1}, 7 benchmarks. Baselines: DFlash, DDTree, Domino, EAGLE. GPU not given in the extract. — [arXiv](https://arxiv.org/html/2608.13524)
    - Code: not found. Stated limitations: none stated.

**Serving, training and context-cost follow-ups**

20. **AdaFlash: Adaptive Speculative Decoding via On-Policy Distilled Diffusion Drafters.** Qian et al. (Nanjing U. + Huawei). arXiv 2607.19223. arXiv only. Jul 21, 2026. [html]
    - Method: on-policy distillation with clipped reverse-KL, plus an adaptive length head.
    - Result: up to 5.3x over AR and +66% throughput over the prior SOTA at 128 concurrency. Setting: Qwen3-8B and Coder-30B, 8 benchmarks. Baselines: DFlash, EAGLE-3, MTP. GPU and T not extracted. — [arXiv](https://arxiv.org/html/2607.19223)
    - Code: not found.
    - Stated limitation, framed as motivation: bidirectional attention gives "high variance" at the domain and token level.

21. **DFlow.** arXiv 2609.06498. arXiv only. Sep 6, 2026. [prior notes]
    - Method: a relay module with boundary-token modulation reuses target hidden states at rejected positions. Trained with Multi-Round Self-Conditioned Training.
    - Result: τ +10.4/13.2/13.4% over DFlash (Qwen3-1.7B/4B/8B, T=0) and +9.0-12.4% at T=1. — [arXiv](https://arxiv.org/html/2609.06498)
    - Code: not found.
    - Stated limitations: 1.2% overhead; block fixed at 16; no long-context study; train-inference mismatch when combined with Domino.

22. **DPara: When Parallel Drafter Meets Parallel Speculative Decoding.** Liu et al. arXiv 2609.27396. arXiv only. Sep 23, 2026. [html]
    - Method: precomputes drafts for every possible acceptance boundary during verification (multi-anchor M-DFlash). Drafting fully overlaps verify, with no fallback.
    - Result: 3.21x (Qwen3-8B) and 3.52x (14B) over AR; +11.1% / 5.1% vs DSpark. Setting: H800, bs 1-16, greedy (sampling in the appendix). Baselines: DFlash, DSpark, EAGLE-3, SSD, PEARL. — [arXiv](https://arxiv.org/html/2609.27396v1)
    - Code: not found.
    - Stated limitations: the extracted quote (fallback up to 96% at bs 16) is about the SSD baseline, not DPara. DPara's own limitations were not retrieved.

23. **LongSpark: Efficient speculative decoding with a fixed-cost parallel drafter.** He, Liu, Shen, Li. arXiv 2609.37029. arXiv only. Sep 2026. [proj; arXiv HTML fetch was rate-limited (HTTP 429)]
    - Method: a block-diffusion drafter whose prefix encoding has fixed cost, using three fixed-size views (boundary state, recent window, global summary).
    - Result: 1.88-2.13x over AR at 1K-128K context. At 128K: 74.3 ms TPOT vs DSpark's 97.2 ms, drafter state 406x smaller than DSpark's, draft time about 3.5 ms flat. Under load (C128): +6-12% vs DSpark. Setting: Qwen3-4B/8B/14B. Baselines: DSpark, DFlash, EAGLE-3. — [project page](https://long-spark.github.io/)
    - Code: https://github.com/Hao-Yuan-He/LongSpark. Stated limitations: none found.

24. **PARD-2: A dual-mode speculative decoding framework with Confidence-Adaptive Token optimization.** An et al. (AMD). arXiv 2605.08632. arXiv only. May 2026, per the ID (the extractor said "2025", which conflicts with the ID). [html]
    - Method: Confidence-Adaptive Token (CAT) reweighting aligns the loss with acceptance length. One drafter serves both target-dependent and target-independent modes.
    - Result: target-dependent Qwen3-8B 5.81x / τ 6.98 vs DFlash 4.61x, PARD 4.39x and EAGLE-3 1.93x; LLaMA3.1-8B 5.19x (abstract: up to 6.94x). Setting: A100-40GB inference, vLLM, bs 1-64, T=0, K=16. — [arXiv](https://arxiv.org/html/2605.08632)
    - Code: https://github.com/AMD-AGI/PARD
    - Stated limitation: gains shrink at very high batch sizes.

25. **Accepted Prefixes Are Not All You Need: A Negative Result on PEFT-Based Block-Diffusion Drafting.** Javat, Kazakov (Bahçeşehir U.). arXiv 2607.12422. arXiv only. Jul 2026. [html]
    - Method: a LoRA adapter on the target backbone acts as a same-backbone block-diffusion drafter (D=16, 50k TULU-3 samples).
    - Result: Qwen3-0.6B reaches 34.05 tok/s (τ 2.88) vs FastMTP's 188.01 tok/s (τ 1.51); bf16, T=0, 256 tokens. Draft and verify each take about 50 ms because both are full-backbone passes. — [arXiv](https://arxiv.org/html/2607.12422)
    - Code: none.
    - Stated limitations: only Qwen3-0.6B; small training set; DFlash and DART not benchmarked; throughput depends on runtime and kernels.

### Inferences
Inferred weaknesses, one or two per paper. These are my reading, not claims made by the papers.
- Blockwise (1): MT and super-resolution only, and greedy only; the k heads are independent, which is the same problem every 2026 paper is still fixing.
- PARD (2): the drafter is a separate small AR-family model of its own, so a target-independent drafter only works within a model family (shared tokenizer). The two headline numbers differ across versions (3.67x vs 4.08x on LLaMA3.1-8B).
- DiffuSpec (3): it uses an off-the-shelf dLLM that is not aligned to the target, so it depends on dLLM availability per tokenizer. Its path search is serial overhead.
- SpecDiff-2 (4): evaluated on older Qwen2.5 and Llama-2-era targets, so it is not directly comparable to the Qwen3/DFlash baselines. Self-selection costs extra drafter samples.
- DEER (5): the headline is a single best case (HumanEval on the 30B-A3B MoE); T=0 only; weights not released.
- P-EAGLE (6): the modest 1.10-1.36x over EAGLE-3 is far below DFlash-class gains. No comparison with DFlash is reported in what I saw.
- DFlash (7): product-of-marginals drafting; fixed block 16, so speedup falls from 5.1x to 2.8x at concurrency 32; one drafter per target and per thinking mode.
- DDTree (8): the tree optimizes the surrogate Q, not the target; the budget is picked on the test set; bs 1 only, so the tree's verify cost is unpriced at batch >1.
- BASTION (9): bs 1 only; roofline profiles are tied to the hardware.
- CaDDTree (10): matching DDTree's oracle is a ceiling, not a gain over a tuned DDTree; T=1 is relegated to the appendix.
- GRAFT (11): shorter τ than DDTree at the same budget; A100 bs 1 only; the venue looks like a submission, not an acceptance.
- Domino (12): T=0 only; the GRU adds a sequential component whose cost grows with block length.
- TreeFlash (13): the gain over DDTree is +9% at bs 1. The tree-size tension with production kernels is unresolved.
- D2SD (14): two drafters means double the training and memory; the cascade-attention verify is wider, which gets costly at batch >1.
- JetSpec (15): the 9.64x headline is on MATH-500, the most predictable domain; the chat number (4.58x) is less than half of it.
- DSpark (16): the scheduler relies on throughput profiling and production traffic that cannot be reproduced. The open-sourced Qwen3 results are about τ, while the wall-clock claims come from DeepSeek-internal serving.
- xPress (17): Qwen3-8B on a single GPU type only. Jacobi iterations add latency that may not pay off once τ is already high.
- DBLAST (18): no end-to-end wall-clock was found in the earlier Pith read. The setting is temperature-heavy (T=1.5) and favors the method.
- DARTree (19): the 9.73x / τ 12.97 headlines are max values, and hardware is unspecified in what I saw. Depth-wise batching of AR heads adds draft passes that scale with depth.
- AdaFlash (20): on-policy distillation needs target rollouts per domain, which is expensive. The "66% over prior SOTA" figure is at one concurrency point.
- DFlow (21): fixed block 16; gains stack with Domino only after extra online rollout.
- DPara (22): precomputing drafts for all acceptance boundaries multiplies drafter compute by up to the block length, and that compute competes with verify at batch >1. The 3.2-3.5x is lower than DFlash's batch-1 numbers in other papers, so cross-paper comparison is unclear.
- LongSpark (23): fixed-size summaries may lose rare long-range tokens (e.g. copying from far context). The speedup (about 2x) is modest; the main win is at long context and under load.
- PARD-2 (24): T=0 only. The DFlash baseline (4.61x) is run by PARD-2's own authors on A100 vLLM, not taken from DFlash's paper.
- Negative result (25): a 0.6B target is the worst case for same-backbone drafting; it does not show the result holds at 8B and above. Single baseline.
- Cross-cutting: nearly all 2026 papers use Qwen3-4B/8B targets, bs 1 and T=0 as the headline. Baselines are inconsistent: DFlash numbers for Qwen3-8B range from 4.16x (D2SD, H200) to 4.86x (DFlash paper) to 4.61x (PARD-2). Few papers compare against DDTree, and none against each other's trees at batch >1. "Lossless" is not audited in bf16 (see prior notes).

### Gaps
- Arxiv HTML was refused or failed for: LongSpark (HTTP 429, rate limited); JetSpec and TreeFlash HTML on the first try (TreeFlash v1 later succeeded); 2605.30852 v1 PDF (empty). Settings for JetSpec, P-EAGLE, DiffuSpec and SpecDiff-2 (GPU, bs, T) are therefore missing.
- DFlash 2: no paper or arXiv ID found. The inco.ai blog could not be fetched (provenance block). Method details come only from the local code.
- BASTION, CaDDTree and DFlow details come from earlier notes and were not re-read today; the BASTION title was not re-confirmed.
- Venues: only DFlash (ICML 2026, from the project page), DiffuSpec (Findings ACL 2026) and Blockwise (NeurIPS 2018) are confirmed. GRAFT's AAAI status is unconfirmed. No OpenReview checks were done.
- No independent reproduction was found for any 2026 paper beyond the team's own DDTree/DFlash runs.

## Q2. Scope decisions, extras found, and ID verification

### Takeaway
All 26 candidate IDs resolve to the expected papers. "Efficient speculative decoding with a fixed-cost parallel drafter" (2609.37029) is titled LongSpark on arXiv. Speculative Pipeline Decoding is not a parallel drafter and should be treated as adjacent. Several further block-drafter papers exist (DScale, CAST, Adjacent Causal Injection, DeLS-Spec, HyperDFlash, DFlare, DFly, DART) and can be swapped in if the list needs extending.

### Cited Findings
- **Excluded as out of scope: Speculative Pipeline Decoding** (arXiv 2605.30852).
  - Title changed between versions: v1 "Higher-Accuracy Drafting with Hidden Latency via Pipeline Parallelism", v2 "Higher-Accruacy and Zero-Bubble Speculation via Pipeline Parallelism" (typo is in the original).
  - Method: the target is split into n pipeline stages, with a Transformer speculation module aggregating multi-depth features. This is self-speculation, not a block drafter.
  - Result: 2.99-3.04x on HumanEval (n=8, T=0) vs EAGLE-3's 2.10-2.94x. Setting: Qwen3.5-4B/9B, bs 1, n+1 GPU ranks.
  - Stated limitations: n=16 hurts; 5 or 9 ranks do not fit 4- or 8-GPU nodes; draft trees left to future work. — [arXiv v2 HTML](https://arxiv.org/html/2605.30852v2)
- **ID checks.** IDs and titles confirmed via arXiv abs/HTML or search for: 2504.18583, 2605.08632, 2602.01469, 2602.06036, 2604.12989, 2510.02358, 2511.00606, 2512.15176, 2608.05448, 2608.02438 (xPress), 2606.03819, 2605.29707, 2607.05147, 2606.18394, 2608.13524, 2606.04446, 2608.20375, 2607.19223, 2609.27396, 2609.37029, 2605.30852, 2607.12422, 1811.03115. 2606.01813, 2605.29727 and 2609.06498 were confirmed in earlier notes. — sources as cited in Q1
- **Extras, abstract or title only:**
  - DScale (arXiv 2609.37532, Sep 2026): a 112K-parameter predictor with dynamic verify-length allocation; +43.9-48.8% throughput over DFlash and +22-38% over DSpark at concurrency 8-32 (A100-40GB). — [arXiv](https://arxiv.org/abs/2609.37532) [prior notes]
  - CAST: Cost-Aware Speculative Trees from One-Pass Block Drafters (arXiv 2610.00321; title only, very recent). — [arXiv](https://arxiv.org/html/2610.00321)
  - Draft in Parallel, Condition Through Depth: Adjacent Causal Injection (arXiv 2609.36173; title only). — [arXiv](https://arxiv.org/html/2609.36173)
  - DeLS-Spec: Decoupled Long-Short Contexts for Parallel Speculative Drafting (arXiv 2607.07409; title only; overlaps with LongSpark). — [arXiv](https://arxiv.org/html/2607.07409)
  - HyperDFlash (arXiv 2606.26744; title only). — [alphaXiv](https://www.alphaxiv.org/abs/2606.26744)
  - DART: Diffusion-Inspired Speculative Decoding (arXiv 2601.19278; title only; used as a baseline by Domino). — [arXiv](https://arxiv.org/html/2601.19278v1)
  - ParallelSpec (arXiv 2410.05589; title only; an earlier parallel drafter). — [arXiv](https://arxiv.org/html/2410.05589v1)
  - DFlare (2606.02091) and DFly (2607.25852): IDs as cited in DFlow's related work, unverified. — [DFlow](https://arxiv.org/html/2609.06498)
- **Adjacent training papers that use DFlash as a baseline:** VAT (2608.30135), which reports DFlash +11.4% τ; Draft-OPD (2605.29343), which reports +13% vs DFlash. — [prior notes](../SD%20and%20early%20exit%20research%20gaps/drafter_side.md)

### Inferences
- If the team wants exactly 18-20 papers, the weakest candidates to cut are P-EAGLE (small gains, no DFlash comparison), DiffuSpec (superseded) and the negative result (a 0.6B toy). If more serving coverage is needed, DScale should replace one of them.
- Speculative Pipeline Decoding is useful only as a contrast. It hides draft latency by pipelining the target, while DPara hides it through overlap. The team's own hierarchy angle sits between the two.

### Gaps
- CAST, Adjacent Causal Injection, HyperDFlash, DFlare and DFly were not read; they may belong in the main list.
- No OpenReview or ICLR 2027 submission check was done.
