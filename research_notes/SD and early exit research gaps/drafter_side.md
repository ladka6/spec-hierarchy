# Drafter design and drafter training for LLM speculative decoding (2025 to Sep 2026): crowded vs open

Scope note: researched 2026-09-30. Source pages were read through a summarizing fetch tool, so exact numbers are as reported by the paper abstract/HTML; items marked [UNVERIFIED] are IDs or claims I did not open this session (from search-result titles or background knowledge). The hemingkx/SpeculativeDecodingPapers README could not be extracted (fetch returned only GitHub chrome), so the paper list below is built from targeted searches, not from that list.

## Q1. Which drafter-side directions are saturated (many 2025-2026 papers, diminishing gains)?

### Takeaway
Block-diffusion / parallel drafters conditioned on target hidden states (the DFlash line) became the dominant 2026 paradigm, and the space around it is now crowded: at least 8 papers from Apr to Sep 2026 add intra-block dependency modeling, tree construction, or budget/verify-length control on top of DFlash, each reporting +9% to +18% acceptance length over DFlash (not over DDTree). Tree construction/budgeting for diffusion drafters and "add causal dependency to a parallel drafter" are saturated; your own +5-7% over DDTree from dependency heads/block trees is consistent with this.

### Cited Findings

Cluster A: Parallel / block-diffusion drafters (backbone)
- DFlash (arXiv 2602.06036, Feb 2026): 5-layer block-diffusion drafter, 16-token blocks, target features injected into every draft layer's KV; exponentially decaying position loss weights; ~800K target-generated responses. Qwen3-8B: 4.86x at T=0, 4.03x at T=1; ~2.4x over EAGLE-3 at T=0. Paper does not enumerate limitations. — [arXiv 2602.06036](https://arxiv.org/html/2602.06036v1)
- P-EAGLE: Parallel-Drafting EAGLE with Scalable Training (arXiv 2602.01469, Feb 2026), integrated in vLLM. — [arXiv](https://arxiv.org/abs/2602.01469); [vLLM blog](https://vllm.ai/blog/2026-03-13-p-eagle) (gains not extracted)
- Concurrent/earlier parallel drafters cited by DFlash: DiffuSpec, SpecDiff-2, PARD, TiDAR. — [DFlash](https://arxiv.org/html/2602.06036v1). IDs [UNVERIFIED from background]: PARD 2504.18583, DiffuSpec 2510.02358, SpecDiff-2 2511.00606.
- DSpark (arXiv 2607.05147, Jul 2026; PKU + DeepSeek-AI): DFlash-based parallel backbone plus a light sequential module (semi-AR) plus a confidence head with a hardware-aware scheduler choosing verify length. +16.3-18.4% macro accepted length over DFlash (Qwen3 4B-14B), +26.7-30.9% over EAGLE-3; 60-85% per-user speedup in DeepSeek-V4 production. — [arXiv 2607.05147](https://arxiv.org/html/2607.05147v1)
- Domino (arXiv 2605.29707, May 2026): DFlash backbone + GRU causal encoder + low-rank logit correction; +16.6% acceptance length vs DFlash, +12.3% end-to-end, 5.3% extra params, up to 5.49x (Transformers), 5.8x (SGLang). — [arXiv 2605.29707](https://arxiv.org/html/2605.29707v1)
- TreeFlash (arXiv 2606.03819, Jun 2026): MLP conditioned on drafter hidden state + previous token to approximate AR distribution in a single pass; +12% block efficiency, +9% speedup over marginal tree drafting. — [arXiv 2606.03819](https://arxiv.org/abs/2606.03819)
- JetSpec (arXiv 2606.18394, Jun 2026, rev. Sep 27): causal parallel draft head over fused hidden states giving branch-wise causal conditioning; frames a "causality-efficiency dilemma"; up to 9.64x on MATH-500, 4.58x chat (H100, Qwen3 dense and MoE). — [arXiv 2606.18394](https://arxiv.org/abs/2606.18394)
- DBLAST (arXiv 2608.05448, Aug 2026): low-rank categorical latent mixture to correlate positions in a block, plus acceptance-oriented objective; >12% accepted length on Qwen3-8B at T=1.5. — [Pith summary](https://pith.science/paper/2608.05448)
- xPress: Parallel Refinement for Diffusion Drafters (arXiv 2608.02438, Aug 2026). — [arXiv](https://arxiv.org/abs/2608.02438) (content not extractable; gains unknown)
- D2 SD: Dual Diffusion Draft Models (arXiv 2606.04446, Jun 2026). — [arXiv](https://arxiv.org/html/2606.04446v1) (not read)
- DARTree: Speculative Diffusion Decoding with AR Draft Trees (arXiv 2608.13524, Aug 2026). — [arXiv](https://arxiv.org/html/2608.13524v1) (not read)
- DeLS-Spec: Decoupled Long-Short Contexts for Parallel Speculative Drafting (arXiv 2607.07409). — [arXiv](https://arxiv.org/html/2607.07409v1) (not read)
- Other block-diffusion drafters cited by DFlow: DFlare (2606.02091), DFly (2607.25852), "DFlash 2" (Inco AI blog, no arXiv). — [DFlow related work](https://arxiv.org/html/2609.06498) [IDs as reported by extractor, UNVERIFIED]
- DEER: search surfaced arXiv 2512.15176 for a DEER query but title not confirmed. [UNVERIFIED]

Cluster B: Tree construction / budget for parallel drafters
- DDTree (arXiv 2604.12989, Apr 2026): best-first heap over DFlash per-position marginals under a node budget, verified in one pass with ancestor-only mask; evaluated at T=0 in the main text. — [arXiv 2604.12989](https://arxiv.org/html/2604.12989v1)
- CaDDTree (arXiv 2606.01813, Jun 2026): per-round budget chosen to maximize tokens/sec using offline latency profiling (unimodal time-per-token in budget) and drafter confidence; matches DDTree-oracle budget automatically (Qwen3-4B MATH-500: 4.53 ms/tok both); T=1 only in appendix. — [arXiv 2606.01813](https://arxiv.org/html/2606.01813)
- BASTION (arXiv 2605.29727, May 2026; KAIST/SAIT): query-dependent tree from block-diffusion marginals, path-confidence product as acceptance surrogate, roofline latency model; 6.61x over AR, 1.39x over DFlash; T=0 and T=1; limitation: batch size 1 only, needs stable runtime profiles. — [arXiv 2605.29727](https://arxiv.org/html/2605.29727v1) (extractor said "2025", contradicted by the 2605 ID)
- GRAFT: adaptive DLM-based draft tree construction (arXiv 2608.20375). — [arXiv](https://arxiv.org/pdf/2608.20375) (not read)
- Draft Less, Retrieve More: hybrid tree construction (arXiv 2605.20104); SpecBlock (2605.07243). — [arXiv](https://arxiv.org/html/2605.20104); [awesomepapers](https://awesomepapers.io/llm-papers/papers/hf2605.07243) (not read)
- Background: EAGLE-2 dynamic trees (2406.16858), OPT-Tree (2406.17276). [UNVERIFIED IDs, background]

Cluster C: Verify-length / serving-level adaptation (high concurrency)
- DScale (arXiv 2609.37532, Sep 29 2026): 112K-param predictor, path-aware tiles, dynamic verify-length allocation; +43.9-48.8% throughput over DFlash, +22.2-37.7% over DSpark, +24.4-32.0% over Domino; Qwen3-4B/8B on A100-40GB, concurrency 8-32. Does not study T>0, drafter scaling, or calibration. — [arXiv 2609.37532](https://arxiv.org/abs/2609.37532)
- AdaFlash (arXiv 2607.19223, Jul 2026): on-policy distillation (reverse KL with clipping) plus adaptive length head; up to 66% throughput over DFlash at 128 concurrency. — [arXiv 2607.19223](https://arxiv.org/html/2607.19223)
- DPara (arXiv 2609.27396, Sep 23 2026): parallel SD without fallback by precomputing drafts for every acceptance boundary (M-DFlash, multi-anchor finetune); 3.21-3.52x over AR on Qwen3-8B/14B; +11-31% over DSpark. — [arXiv 2609.27396](https://arxiv.org/html/2609.27396v1)

Cluster D: Cross-round reuse
- DFlow (arXiv 2609.06498, Sep 6 2026): reuses target hidden states at rejected positions via a relay module with boundary-token modulation; Multi-Round Self-Conditioned Training; +10.4/13.2/13.4% τ over DFlash (Qwen3 1.7B/4B/8B, T=0), +9.0-12.4% at T=1. Limitations: 1.2% overhead, block size fixed at 16, no long-context study, train-inference mismatch when combined with Domino. — [arXiv 2609.06498](https://arxiv.org/html/2609.06498)

Cluster E: Multi-token prediction in the target
- "Your LLM Knows the Future" (arXiv 2507.11851, Jul 2025). — [arXiv](https://arxiv.org/abs/2507.11851) (not read)
- Parallel Token Prediction (arXiv 2512.21323, ICLR 2026). — [arXiv](https://arxiv.org/pdf/2512.21323) (not read)
- NeMo-RL SD paper supports native MTP heads as drafters during RL. — [arXiv 2604.26779](https://arxiv.org/html/2604.26779v1)
- FastMTP 2509.18362, DeepSeek-V3 MTP 2412.19437. [UNVERIFIED IDs, background]

Cluster F: Training objectives
- GTO, Group Tree Optimization (arXiv 2509.22134, Sep 2025): addresses draft policy misalignment between training and tree decoding. — [arXiv](https://arxiv.org/abs/2509.22134) (abstract not read)
- AdaSPEC selective KD (arXiv 2510.19779, Oct 2025). — [arXiv](https://arxiv.org/abs/2510.19779) (not read)
- VAT (arXiv 2608.30135, Aug 31 2026): verification head predicting per-position acceptance from simulated verification + weighting anchored at each sample's first rejection; EAGLE-3 +8.0% τ / +7.9% wall-clock, DFlash +11.4% τ / +8.7%; T=0 and T=1; Qwen3-4B/8B, Llama-3.1-8B. Cites concurrent PARD-2 and D-PACE. — [arXiv 2608.30135](https://arxiv.org/html/2608.30135)
- Acceptance-Aware Draft Model Training (arXiv 2609.24150, Sep 2026): EAL loss (greedy) gives only +0.2-0.5%; WTV loss (temperature-scaled TV overlap) gives ~15% at T=0.5 and T=1.0 over KL; plus GRPO stage with simulated acceptance length reward. Does not test EAGLE-3 or DFlash. — [arXiv 2609.24150](https://arxiv.org/html/2609.24150)
- Draft-OPD (arXiv 2605.29343, 2026): on-policy distillation with target-assisted rollout and replay from error positions; +23% over EAGLE-3, +13% over DFlash, >5x for thinking models. — [awesomepapers](https://awesomepapers.io/llm-papers/papers/2605.29343)
- Learning to Draft: adaptive SD with RL (arXiv 2603.01639). — [arXiv](https://arxiv.org/html/2603.01639v1) (not read)
- BudgetDraft: acceptance-aware multi-view training for sparse-KV SD (arXiv 2606.00144). — [awesomepapers](https://awesomepapers.io/systems-efficiency/papers/2606.00144) (not read)
- Background: HASS 2408.15766, GRIFFIN 2502.11018, EAGLE-3 2503.01840 [UNVERIFIED IDs]; DistillSpec 2310.08461 — [cited in EDA](https://arxiv.org/html/2603.09527v1)

### Inferences
- Saturated: (1) adding intra-block causal dependency to a parallel drafter (Domino, TreeFlash, DSpark, JetSpec, DBLAST, DARTree, D2 SD): 7+ papers in 5 months, all +9-18% over DFlash. Your +5-7% over DDTree suggests most of this gain overlaps with what DDTree's tree already captures; the incremental return over DDTree (rather than over DFlash chains) appears small, and none of these papers report gains over DDTree as baseline except CaDDTree (which matches it).
- Saturated: tree/budget construction for block-diffusion marginals (DDTree, CaDDTree, BASTION, GRAFT, DARTree, JetSpec) and concurrency-aware verify-length control (DSpark, AdaFlash, DScale, DPara). The DScale table shows these methods now compete on 20-50% throughput margins against each other, mostly via systems tricks.
- Training objectives: acceptance-aware losses are crowded too (VAT, 2609.24150, GTO, Draft-OPD, AdaFlash OPD, DBLAST's objective), with small gains at T=0 (EAL +0.2-0.5%, VAT ~8-11%) but larger at T>0 (WTV ~15%).
- Cross-round reuse: DFlow reuses verifier hidden states (not stale draft tokens) and gets 10-13%; your result that stale-token repair gives 0.28 of a fresh draft is consistent with the field moving to reusing target features instead of tokens. This direction has one strong paper (DFlow) plus DPara; not saturated but the obvious variant is taken.

### Gaps
- Could not read the hemingkx README, so coverage of 2025 EAGLE-family head papers (HASS, GRIFFIN successors) is thin.
- xPress, D2 SD, DARTree, GRAFT, DeLS-Spec, DFlare, DFly, P-EAGLE numbers not extracted.
- No paper found that reports gains relative to DDTree (the strongest tree baseline) for dependency heads; unclear whether reported +16% over DFlash survives with DDTree trees.

## Q2. What do the newest (2026) papers list as limitations / future work?

### Takeaway
Most 2026 drafter papers have no explicit limitations section in the HTML; stated or visible limitations cluster around: temperature=0-only main results, fixed block size 16, batch-size-1 evaluations, Qwen3-only targets, frozen targets, and short-context benchmarks.

### Cited Findings
- DFlow: overhead 1.2%; block size fixed at 16, larger blocks unexplored; no long-context scenarios; train-inference mismatch with Domino requiring online rollout. — [arXiv 2609.06498](https://arxiv.org/html/2609.06498)
- BASTION: batch size 1 only; depends on stable runtime profiles for calibration. — [arXiv 2605.29727](https://arxiv.org/html/2605.29727v1)
- DBLAST: soft mixture at train vs greedy-branch at inference is formally uncharacterized; no confidence intervals; no end-to-end wall-clock throughput. — [Pith](https://pith.science/paper/2608.05448)
- DSpark: room for improvement under strict interactivity constraints. — [arXiv 2607.05147](https://arxiv.org/html/2607.05147v1)
- DDTree: main results T=0; tree built from factorized marginals (surrogate objective). — [arXiv 2604.12989](https://arxiv.org/html/2604.12989v1)
- CaDDTree: T=1 only in appendix; no explicit limitations in text. — [arXiv 2606.01813](https://arxiv.org/html/2606.01813)
- DScale: does not study drafter size/block length scaling, T>0, or calibration. — [arXiv 2609.37532](https://arxiv.org/abs/2609.37532)
- VAT, AdaFlash, CaDDTree, DFlash: no explicit limitations/future work found by extractor. — [VAT](https://arxiv.org/html/2608.30135), [AdaFlash](https://arxiv.org/html/2607.19223), [DFlash](https://arxiv.org/html/2602.06036v1)
- AdaFlash assumes a frozen target; no target retraining. — [arXiv 2607.19223](https://arxiv.org/html/2607.19223)
- 2609.24150 (acceptance-aware training) not tested on EAGLE-3 or DFlash. — [arXiv 2609.24150](https://arxiv.org/html/2609.24150)
- Online Draft Co-Training (2609.07108): SD benefits "currently limited" for sparse MoE and linear-attention models. — [arXiv 2609.07108](https://arxiv.org/html/2609.07108)
- Scaling Laws for SD (2505.07858): "impact of RLHF on draft accuracy presents an intriguing open question"; dense only, MoE unexplored; AR small drafters only. — [arXiv 2505.07858](https://arxiv.org/pdf/2505.07858)
- NeMo-RL SD (2604.26779): speedup degrades with longer draft lengths despite higher acceptance; draft initialization strongly matters; online adaptation gives limited gains with good initialization; benefits shrink under async RL; large models more sensitive to policy lag. — [arXiv 2604.26779](https://arxiv.org/html/2604.26779v1)
- SpecRoll (2608.04962): only math, up to 14B; code/multilingual, longer training runs, larger deployments left open. — [arXiv 2608.04962](https://arxiv.org/html/2608.04962)
- TLT (2511.16665): impact of draft staleness under weight evolution not fully analyzed; 64-GPU max; bandit tuner hyperparameter sensitivity. — [arXiv 2511.16665](https://arxiv.org/html/2511.16665v2) (extractor-paraphrased, may be partly inferred)

### Inferences
- The consistent unaddressed axes across the DFlash family: block size beyond 16, long context, temperature > 0 as a first-class setting, non-Qwen3 targets, and the target changing over time.
- Several "limitations" in my extraction were paraphrased by the summarizer rather than quoted; only DFlow, SpecRoll, NeMo-RL and 2505.07858 had clearly quoted/explicit ones.

### Gaps
- Could not retrieve explicit limitation sections for Domino, JetSpec, TreeFlash, P-EAGLE, Draft-OPD.

## Q3. Is anyone studying parallel/diffusion drafters under sampling (T>0), drafter calibration, uncertainty for tree budgets, or drafter scaling laws?

### Takeaway
Sampling for block drafters: yes, recently (DBLAST Aug 2026, WTV loss Sep 2026), plus T=1 tables in DFlash/DFlow/VAT/BASTION. Calibration: DSpark introduces post-hoc Sequential Temperature Scaling; BASTION/CaDDTree/AdaFlash/VAT use drafter confidence or acceptance heads for budgets. Scaling laws: only for AR small drafters (2505.07858); no scaling-law study for block-diffusion drafters (layers, block length, data, target size) found.

### Cited Findings
- DFlash at T=1 loses ~17% speedup vs T=0 on Qwen3-8B (4.86x to 4.03x). — [arXiv 2602.06036](https://arxiv.org/html/2602.06036v1)
- DBLAST: block-diffusion drafters assume conditional independence across positions, and "the accepted draft length degrades as the entropy of the target sampling distribution increases"; latent-mixture fix gives >12% at T=1.5. — [Pith](https://pith.science/paper/2608.05448)
- WTV loss directly optimizes temperature-scaled distributional overlap; ~15% gain at T=0.5/1.0 over KL, but greedy-oriented EAL gives only 0.2-0.5%. — [arXiv 2609.24150](https://arxiv.org/html/2609.24150)
- DFlow gains hold at T=1 (9.0-12.4%). — [arXiv 2609.06498](https://arxiv.org/html/2609.06498)
- DSpark calibration: Sequential Temperature Scaling, 1D grid search on held-out data minimizing ECE while preserving rankings; confidence head drives verify length. — [arXiv 2607.05147](https://arxiv.org/html/2607.05147v1)
- BASTION: path-confidence product of drafter marginals as expected-acceptance surrogate for adaptive tree growth. — [arXiv 2605.29727](https://arxiv.org/html/2605.29727v1)
- CaDDTree: per-round budget depends on drafter confidence and profiled verify cost. — [arXiv 2606.01813](https://arxiv.org/html/2606.01813)
- VAT verification head predicts per-position acceptance; AdaFlash adaptive length head predicts acceptance ratio. — [VAT](https://arxiv.org/html/2608.30135); [AdaFlash](https://arxiv.org/html/2607.19223)
- "When Is a Draft Accepted? A Theory of Acceptance in SD" (arXiv 2606.30265). — [arXiv](https://arxiv.org/html/2606.30265) (not read)
- Scaling laws: log-linear acceptance vs pretraining tokens and drafter depth, for AR "Scylla" drafters 1-5 layers (138M-413M); no heads or diffusion drafters. — [arXiv 2505.07858](https://arxiv.org/pdf/2505.07858)
- JetSpec and DScale titles use "scaling" but refer to tree size / concurrency, not drafter-parameter scaling. — [JetSpec](https://arxiv.org/abs/2606.18394); [DScale](https://arxiv.org/abs/2609.37532)

### Inferences
- Tree-under-sampling is a real hole: DDTree/CaDDTree trees are built from marginals and evaluated mostly at T=0; the correct acceptance rule for a tree under stochastic verification (multi-candidate rejection sampling, e.g., SpecInfer/recursive rejection) interacts with the drafter's joint vs marginal distribution, and DBLAST studies chains/blocks, not trees. No paper found that evaluates DDTree-style trees at T in [0.6, 1.0] as a main result with the exact multi-draft acceptance rule.
- Calibration exists as an engineering component (DSpark STS) but nobody appears to measure whether drafter calibration is preserved after target fine-tuning or across temperatures, nor how miscalibration costs throughput in budget-adaptive trees.
- Block-diffusion drafter scaling (layers 1-8, block 8-32, training tokens, target size 1.7B-14B) is not characterized; DFlow explicitly leaves block size >16 unexplored.

### Gaps
- Did not verify whether DSpark reports ECE per temperature or per domain.
- Did not read 2606.30265 (acceptance theory); may contain relevant T>0 analysis.

## Q4. Joint drafter/target post-training, drafters for changing targets (RL, continual updates), and RL-rollout SD

### Takeaway
Drafter adaptation to a fine-tuned target is published (EDA 2603.09527 shared-private experts; Draft-OPD; AdaFlash OPD; vLLM LoRA-on-DFlash RFC), and drafter co-training during RL is now a busy systems area (TLT, FastGRPO, ReSpec, SPEC-RL, NeMo-RL SD, SpecRoll, Online Draft Co-Training). What is thin: principled continual-learning treatment of the drafter across a sequence of target versions (forgetting, a single drafter serving many target versions/adapters), and online learning theory with a non-stationary target.

### Cited Findings
- EDA, Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation (arXiv 2603.09527, Mar 2026): when target is domain-fine-tuned (Qwen2.5-Math/Coder/Meditron), direct transfer drops τ from ~4.7 to ~1.2; shared frozen FFN experts + trainable private experts (~27.5% params), self-generated data from fine-tuned target, Mahalanobis-based data selection (50% data); Math τ 4.79 vs full retrain 4.22; shared expert reused across targets. No explicit limitations; Qwen2.5 only; no continual/online comparison. — [arXiv 2603.09527](https://arxiv.org/html/2603.09527v1)
- vLLM RFC #52038: LoRA adapters on a single DFlash drafter reach "within ~2% of a fully-trained per-domain drafter" at ~28x smaller size; drafter adapters independent of target adapters. — [vLLM issue](https://github.com/vllm-project/vllm/issues/52038); analogous SGLang issue for DSpark — [issue](https://github.com/amdpilot-org/sglang/issues/2724)
- "Efficient and Scalable Speculative Decoding with Multi-..." (EMNLP 2025 main). — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.986.pdf) [UNVERIFIED: title truncated, content not read]
- Multi-Drafter SD with Alignment Feedback (ACL Findings 2026). — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1629.pdf) (not read)
- OnlineSPEC / When Drafts Evolve (arXiv 2603.12617, ICML 2026): drafter updates as online learning, speedup bounded by dynamic regret; Online-LR, Opt-Hydra, Ens-Eagle; up to 24% over prior SOTA; assumes fixed target distribution. — [arXiv 2603.12617](https://arxiv.org/html/2603.12617); [ICML](https://icml.cc/virtual/2026/poster/63017). Extractor listed "co-evolving draft-target pairs" as future work; this may be the summarizer's inference, not a quote [UNVERIFIED].
- DVI (arXiv 2510.05421, Oct 2025): self-speculative split of one LLM, LoRA drafter head trained online from verifier decisions; 2.16x on Spec-Bench with Vicuna-7B, greedy; does not address a changing target. — [arXiv 2510.05421](https://arxiv.org/html/2510.05421v1)
- OSD (2401.06706) background. — [cited in 2603.12617](https://arxiv.org/html/2603.12617)
- TLT (arXiv 2511.16665, ASPLOS'26): single-layer drafter trained on idle GPUs during long-tail rollouts via Spot Trainer; BEG bandit tuner; 1.7-2.1x end-to-end RL speedup over VeRL; yields a deployable drafter as by-product. — [arXiv 2511.16665](https://arxiv.org/html/2511.16665v2); [code](https://github.com/mit-han-lab/fastrl)
- SPEC-RL (arXiv 2509.23232, Sep 2025): speculative rollouts reusing prior-epoch trajectories. — [arXiv](https://arxiv.org/abs/2509.23232) (not read in detail)
- ReSpec (arXiv 2510.26475, Oct 2025): optimizing SD in RL systems. — [arXiv](https://arxiv.org/abs/2510.26475) (not read in detail)
- FastGRPO: concurrency-aware verification + online EAGLE-style drafter training; per SpecRoll it does not distinguish transient mismatch from persistent drift. — [SpecRoll related work](https://arxiv.org/html/2608.04962). ID 2509.21792 [UNVERIFIED]
- NeMo-RL SD (arXiv 2604.26779, Apr 2026): EAGLE-3 or native MTP drafters inside NeMo-RL/vLLM; 1.5-1.8x generation, 1.35-1.41x step at 8B; 1.24x async; projected 2.5x at 235B; online adaptation limited gains when initialization is good. — [arXiv 2604.26779](https://arxiv.org/html/2604.26779v1)
- SpecRoll (arXiv 2608.04962, Aug 2026): fast gradient-free trajectory-local hidden-state corrections gated by reliability + slow parameter updates only on sustained drift; 1.21-2.04x end-to-end over GRPO; beats FastGRPO in 15/15 settings. — [arXiv 2608.04962](https://arxiv.org/html/2608.04962)
- Online Draft Co-Training for long-context RL (arXiv 2609.07108, Sep 2026): co-trains EAGLE-3, DFlash, DSpark drafters under context/pipeline parallelism; 1.50-1.88x end-to-end at 8B-122B, 256K contexts. — [arXiv 2609.07108](https://arxiv.org/html/2609.07108). Its related-work IDs as extracted (e.g., EfficientRollout 2606.18967, MTP-RL, SpecForge 2603.18567) are [UNVERIFIED]; some extracted IDs were visibly garbled.
- Other RL-rollout drafters named by SpecRoll: RL-HFSpec, Distribution-Aware SD (suffix-tree from previous rollouts), EfficientRollout (quantized self-drafter). — [SpecRoll](https://arxiv.org/html/2608.04962) (the arXiv ID the extractor gave for Distribution-Aware SD, 2402.03300, is the DeepSeekMath/GRPO ID and is wrong)

### Inferences
- Your finding (LoRA post-training costs 10-35% DDTree speed; ~1 GPU-hour on-policy drafter retrain recovers it; joint drafter serves all targets, no forgetting on base) overlaps in spirit with EDA (shared drafter + per-target private params, self-generated data) and Draft-OPD/AdaFlash (on-policy training). To be novel it must be framed around what they do not cover: diffusion/tree drafters (EDA is architecture-generic, Qwen2.5, not DDTree), LoRA-adapted targets specifically (small drift regime vs EDA's full domain fine-tunes with τ collapse to 1.2), a single joint drafter with zero per-target params, and a quantified cost/benefit curve of drift size vs speed loss vs retrain budget.
- RL-rollout SD is crowded at the systems level (at least 8 papers). The open angle is the drafter-learning question inside it: characterizing drafter staleness as a function of policy KL drift, and whether gradient-free or continual-learning-style drafter updates (replay, regularization) beat periodic retraining. SpecRoll is the closest and only uses EAGLE-style heads on math.
- No paper found jointly optimizing the target's post-training loss with a drafter-acceptance term (i.e., making the target more "draftable"); the co-training papers keep the RL objective unchanged by design.

### Gaps
- Did not verify contents of SPEC-RL, ReSpec, FastGRPO beyond one-line characterizations.
- Did not find the EMNLP 2025 multi-* paper's full title.

## Q5. Candidate gaps (testable on 1-4 A100s with ~8B models) with novelty check

### Takeaway
The most defensible open directions are at the intersection of block-diffusion drafters with (a) stochastic tree verification, (b) calibration under drift/temperature, (c) drafter scaling laws, and (d) drafters for sequences of target versions. Intra-block dependency, tree budgeting, and generic acceptance-aware losses should be avoided.

### Cited Findings (closest existing work per gap)
- Gap 1: Block-diffusion draft trees under stochastic multi-draft verification (DDTree at T=0.6-1.0 with exact multi-candidate rejection; tree scoring that accounts for sampling rather than top-k marginals). Closest: DBLAST (chains/blocks at high entropy, not trees) — [Pith](https://pith.science/paper/2608.05448); WTV loss (training, not trees) — [arXiv 2609.24150](https://arxiv.org/html/2609.24150); BASTION/CaDDTree report T=1 but as secondary — [BASTION](https://arxiv.org/html/2605.29727v1), [CaDDTree](https://arxiv.org/html/2606.01813). Novelty: moderate-high. Risk: someone posts this within weeks given the density.
- Gap 2: Drafter calibration as a measured object: ECE of DFlash marginals vs temperature, domain, and after target LoRA fine-tuning; effect of miscalibration on budget-adaptive trees (CaDDTree/BASTION) throughput. Closest: DSpark STS — [arXiv 2607.05147](https://arxiv.org/html/2607.05147v1); VAT verification head — [arXiv 2608.30135](https://arxiv.org/html/2608.30135). Novelty: moderate (component exists; systematic study does not).
- Gap 3: Scaling laws for block-diffusion drafters (depth 1-8, block length 8-32, training tokens, target size Qwen3 1.7B-14B) at fixed verify budget. Closest: 2505.07858 (AR drafters only) — [arXiv](https://arxiv.org/pdf/2505.07858); DFlow leaves block >16 open — [arXiv 2609.06498](https://arxiv.org/html/2609.06498). Novelty: high; compute feasible for a small team on 4B/8B targets if training runs are short.
- Gap 4: Drafters for a sequence of target versions (continual target updates: LoRA SFT, then DPO/RL checkpoints): forgetting, one drafter vs per-version adapters, drift metric (policy KL on drafter's own distribution) predicting speed loss and retrain need. Closest: EDA — [arXiv 2603.09527](https://arxiv.org/html/2603.09527v1); vLLM LoRA-DFlash RFC — [issue](https://github.com/vllm-project/vllm/issues/52038); SpecRoll two-timescale — [arXiv 2608.04962](https://arxiv.org/html/2608.04962); OnlineSPEC assumes fixed target — [arXiv 2603.12617](https://arxiv.org/html/2603.12617). Novelty: moderate; directly extends your existing result and fits a continual-learning framing.
- Gap 5: Draftability-regularized target post-training (add a term that keeps the target's post-trained distribution close to what the frozen/joint drafter predicts, trading task quality vs SD speed). Closest: none found; co-training papers (TLT, 2609.07108) keep target objective unchanged — [TLT](https://arxiv.org/html/2511.16665v2), [2609.07108](https://arxiv.org/html/2609.07108). Novelty: high by my search, but not directly searched with that phrase [novelty check incomplete].
- Gap 6: Drafter staleness vs policy lag in RL with diffusion/tree drafters (DFlash/DDTree rather than EAGLE heads). Closest: 2609.07108 supports DFlash/DSpark co-training at systems level; NeMo-RL notes sensitivity to policy lag — [arXiv 2604.26779](https://arxiv.org/html/2604.26779v1). Novelty: moderate; a characterization study is feasible at 1.5-8B.

### Inferences
- Given your bounded results (dependency heads +5-7% over DDTree; stale-draft repair 0.28), Gaps 1, 3, and 4 are the strongest fits: they reuse your DFlash/DDTree/LoRA infrastructure and are not the crowded "better block drafter" race.
- Gap 4 needs positioning against EDA and Draft-OPD explicitly; the differentiator is small-drift LoRA regime, zero per-target parameters, diffusion trees, and forgetting measurements.

### Gaps
- Direct phrase searches were run for: drafter scaling laws, continual drafter/multi-target, multi-LoRA drafter, tree budget calibration, high-temperature parallel drafter. Not run for: "draftability" / "speculation-friendly fine-tuning", "calibration of draft model after fine-tuning", "block size scaling diffusion drafter". These should be checked before committing.
