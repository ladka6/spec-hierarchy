# Speculative decoding without a separate trained drafter (self-spec, retrieval/n-gram/Jacobi, in-model MTP)

Scope date: October 2026. Verification legend:
- [V] = read full paper or proceedings PDF this session.
- [A] = verified from the arXiv abstract page or search-result title only.
- [P] = verified in earlier team notes (research_notes/SD and early exit research gaps/*.md), not re-read this session.
- [K] = details from background knowledge, NOT re-verified. Check these before you cite them.
- "Inferred" weaknesses are my own reading. The authors do not state them.

arXiv IDs below were confirmed by search results or fetched pages unless they are marked [K]. Note: the arXiv API (export.arxiv.org) was blocked from the sandbox, so I could not batch-check metadata.

Recommended core list (20): Draft&Verify, LayerSkip, Kangaroo, SWIFT, CLaSp, DEL, SpecEE, HiSpec, QuantSpec, QSpec, TriForce, MagicDec | Lookahead, CLLMs, REST, PLD, SuffixDecoding, SAM-Decoding, Token Recycling, Cacheback | Gloeckle MTP, DeepSeek-V3 MTP, FastMTP, "Your LLM Knows the Future". That is 24. If you need to cut to 20, drop DEL, QSpec, Cacheback and SpecEE: they are the least central, and SpecEE is lossy.

## (a) Self-speculation: layer skipping / early exit / quantized or sparse-KV self-drafts

### Takeaway
Training-free layer skipping (Draft&Verify, SWIFT, CLaSp) is lossless but tops out at about 1.3 to 1.7x at batch 1, with gains growing with model size. Methods that train exits or adapters (LayerSkip, Kangaroo, HiSpec on EE models) reach about 1.7 to 2.2x. Quantized and sparse-KV self-drafts (QuantSpec, TriForce, MagicDec) are the strongest in the long-context, memory-bound regime.

### Cited Findings
- **Draft & Verify (Self-Speculative Decoding)**. arXiv 2309.08168. ACL 2024. Sep 2023.
  - Method: draft by skipping intermediate layers of the target. The skip set is chosen by Bayesian optimization, with no training.
  - Result: up to 1.99x on LLaMA-2.
  - Authors' limitation: none recorded.
  - Inferred weakness: the skip set is searched offline per model and task. SWIFT puts prior search cost at 7.5 to 20 h ([SWIFT](https://proceedings.iclr.cc/paper_files/paper/2025/file/d74d002a9154b4cc433a234feb27c5f4-Paper-Conference.pdf)).
  - Inferred weakness: the static skip set degrades under domain shift. SWIFT shows a static set dropping from 1.47x to 1.01x across tasks (same source).
  - Sources: [P][K]. [arXiv](https://arxiv.org/pdf/2309.08168.pdf), [list](https://github.com/hemingkx/SpeculativeDecodingPapers). Code [K]: github.com/dilab-zju/self-speculative-decoding.
- **LayerSkip**. arXiv 2404.16710. ACL 2024. Apr 2024 (Meta).
  - Method: train with layer dropout plus a shared early-exit loss (pre-training or continual pre-training), then self-speculate with early layers as the draft and the remaining layers as verifier, sharing the KV cache.
  - Result: up to 2.16x on CNN/DM and 1.82x on coding.
  - Code [K]: github.com/facebookresearch/LayerSkip.
  - Authors' limitation [P]: it needs the training recipe, so it is not drop-in for arbitrary checkpoints.
  - Inferred weakness: the training changes the base model, and quality versus the original checkpoint must be re-checked.
  - Inferred weakness: no exit-preserving fine-tuning method was found, so downstream fine-tuning can break the exits ([early_exit notes]).
  - Sources: [P]. [arXiv](https://arxiv.org/abs/2404.16710).
- **Kangaroo**. arXiv 2404.18911. NeurIPS 2024. Apr 2024.
  - Method: a shallow sub-network plus a small trained adapter acts as the draft. A second early exit stops drafting when confidence is low.
  - Result: 1.68x on Spec-Bench, with 88.7% fewer extra parameters than Medusa.
  - Code [K]: github.com/Equationliu/Kangaroo.
  - Inferred weakness: it is not training-free, since the adapter must be trained.
  - Inferred weakness: the speedup is close to that of training-free SWIFT and CLaSp, and below external-drafter methods.
  - Sources: [P]. [arXiv](https://arxiv.org/pdf/2404.18911).
- **SWIFT**. arXiv 2410.06916. ICLR 2025. Oct 2024.
  - Method: on-the-fly optimization of the skipped-layer set from context, plus confidence-based early stopping of drafting. No training.
  - Result: 1.3 to 1.6x wall-clock.
    - LLaMA-2-13B: 1.28 to 1.53x.
    - LLaMA-2-70B: 1.39 to 1.62x.
    - Acceptance rate: 90 to 100%.
  - Setting: RTX A6000, batch size 1, greedy and T=0.6.
  - Code: github.com/hemingkx/SWIFT.
  - Authors' notes:
    - The optimization phase costs about 2 min per run.
    - Gains are smaller for small models.
    - Layer sparsity is task-dependent.
    - Results are sensitive to the context window.
  - Inferred weakness: about 1.3x on 7B to 13B is weak against EAGLE-class drafters.
  - Inferred weakness: batch-1-only evaluation, so the gains likely vanish at larger batch, where verification becomes compute-bound.
  - Sources: [V]. [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/d74d002a9154b4cc433a234feb27c5f4-Paper-Conference.pdf).
- **CLaSp**. arXiv 2505.24196. ACL 2025. May 2025.
  - Method: re-optimizes the skipped layers every step, using dynamic programming over the hidden states from the last verification pass. No training.
  - Result: 1.3 to 1.7x wall-clock on Spec-Bench.
    - MT-Bench scales with size: 1.24x on 8B up to 1.73x on 405B.
    - T=0: 1.56 to 1.75x. T=1: 1.36 to 1.59x.
  - Setting: A800-80GB, batch size 1, FP16 (INT8 for 405B), LLaMA2/3 models.
  - Code: none in the paper.
  - Authors' limitation: only A800 GPUs and LLaMA models were tested, and integration with other SD methods was not explored.
  - Inferred weakness: the per-step DP adds overhead that eats most of the gain on 8B (1.24x).
  - Inferred weakness: there is no public code.
  - Sources: [V]. [ACL](https://aclanthology.org/2025.acl-long.1525.pdf).
- **DEL (Context-aware dynamic exit layer)**. arXiv 2504.05598. COLM 2025. Apr 2025.
  - Method: picks the exit layer and the speculation length dynamically, inside LayerSkip-style self-SD.
  - Speedup numbers: not re-verified.
  - Inferred weakness: it inherits LayerSkip's need for EE-trained checkpoints.
  - Inferred weakness: a later paper (ConfLayers, 2604.14612) reports DEL "often below 1x" on MI300X ([ConfLayers](https://arxiv.org/html/2604.14612)).
  - Sources: [P]. [arXiv](https://arxiv.org/abs/2504.05598).
- **SpecEE**. arXiv 2504.08850. ISCA 2025. Apr 2025.
  - Method: a lightweight predictor uses speculative tokens as features to decide early exit.
  - Result: 2.25x in the cloud setting and 2.43x in the PC setting, on Llama2-7B.
  - Code [K]: github.com/infinigence/SpecEE.
  - Inferred weakness: it is LOSSY. It is an early-exit engine, not lossless SD, so output can differ from the full model.
  - Inferred weakness: the predictor is trained per model.
  - Sources: [P]. [ACM](https://dl.acm.org/doi/10.1145/3695053.3730996).
- **HiSpec**. arXiv 2510.01336. Oct 2025. Authors: Kumar, Sanghavi, Das.
  - Venue: the fetched v2 HTML indicates ICML, but this is unconfirmed (it may just be the template). Treat it as arXiv until checked.
  - Method: three levels on early-exit models: a shallow layer drafts, an intermediate EE layer verifies early, and the full model does periodic final verification.
  - Result: average 1.28x and up to 2.01x throughput versus single-layer speculation.
    - Llama2-70B acceptance rate: 39.7% to 58.1%.
    - An earlier v1 read recorded a best of 2.08x on an EE-trained Llama3-8B, so the numbers differ between versions.
  - Setting: 4x H100, 8 models.
  - Code: none found.
  - Stated limitations: none in the fetched text.
  - Inferred weakness: it needs EE-trained or post-trained models, and post-trained EE models have few exits.
  - Inferred weakness: it was compared only to self-drafting baselines (LayerSkip, SWIFT, Lookahead), not EAGLE-3 or DFlash.
  - Inferred weakness: the batch size is not clearly reported.
  - Sources: [V]. [arXiv HTML v2](https://arxiv.org/html/2510.01336v2), [v1 notes](https://arxiv.org/html/2510.01336v1).
- **QuantSpec**. arXiv 2502.10424. ICML 2025 (PMLR v267). Feb 2025.
  - Method: self-draft with the same architecture, using a hierarchical 4-bit quantized KV cache and 4-bit weights. Target long context.
  - Result: over 90% acceptance and about 2.5x.
  - Inferred weakness: the gains depend on the long-context, memory-bound regime and shrink for short prompts.
  - Inferred weakness: it needs custom quantized kernels.
  - Sources: [P]. [PMLR](https://proceedings.mlr.press/v267/tiwari25b.html), [arXiv](https://arxiv.org/html/2502.10424v1).
- **QSpec**. arXiv 2410.11305. EMNLP 2025. Oct 2024.
  - Method: draft with low-precision activation and weight quantization (e.g. W4A4), and verify with higher-precision weight-only quantization (W4A16) of the same weights.
  - Venue verified [P] ([ACL Anthology](https://aclanthology.org/2025.emnlp-main.240/)). arXiv ID and numbers are [K].
  - Inferred weakness: the "target" is itself a quantized model, so it is lossless only relative to W4A16, not FP16.
  - Inferred weakness: it needs hardware with fast low-bit activation kernels.
- **TriForce**. arXiv 2404.11912. COLM 2024. Apr 2024.
  - Method: a two-level hierarchy. A tiny model drafts for the target running on a retrieval-selected sparse KV cache, and the full-KV target verifies that.
  - Result: 2.31x on Llama2-7B-128K on an A100.
  - Also [K]: a large speedup in an offloading setting on 2x RTX 4090. Code: Infini-AI-Lab/TriForce.
  - Strictly, it still uses a small model at the bottom level, so it is a hybrid.
  - Inferred weakness: it targets long-context, batch-1 or offloading scenarios.
  - Sources: [P]. [arXiv](https://arxiv.org/pdf/2404.11912.pdf).
- **MagicDec**. arXiv 2408.11049. ICLR 2025. Aug 2024.
  - Method: self-speculation with a sparse StreamingLLM-style KV cache as the drafter. It shows that for long sequences, SD helps even at moderate-to-large batch, because KV loading dominates.
  - Result [K]: about 2x on LLaMA-2-7B-32K and about 1.84x on LLaMA-3.1-8B at batch 32 to 256 on 8x A100.
  - Code: github.com/Infini-AI-Lab/MagicDec.
  - Inferred weakness: the claims hold only in the long-context regime. At short context and large batch, the original compute-bound argument returns.
  - Sources: [P]. [arXiv](https://arxiv.org/abs/2408.11049).

### Inferences
- Training-free self-drafts pay a large fraction of target depth per draft token (typically 40 to 60% of layers kept). That caps speedup well below external one-layer drafters. This is the common downside to analyze across this group.
- Almost all (a)-papers report only batch size 1, except MagicDec. Speedup under batched serving (vLLM/SGLang) is largely unreported for SWIFT, CLaSp, Kangaroo and HiSpec.

### Gaps
- DEL and QSpec exact numbers and settings, and the Kangaroo and LayerSkip hardware details, were not re-read this session.
- HiSpec's venue (ICML?) is unconfirmed, and the paper had no code link.
- Draft&Verify's code link and limitations section were not re-read.

## (b) Retrieval, n-gram and suffix drafting; Jacobi and lookahead

### Takeaway
Model-free drafting (PLD, REST, SuffixDecoding, SAM, Token Recycling, Cacheback) gives about 1.6 to 2.3x on general chat at batch 1. It gives much more (up to about 5x) on repetitive, code or agentic workloads, where the output copies its context. Results depend heavily on workload and datastore. Jacobi and lookahead methods trade extra FLOPs for parallelism. CLLMs require fine-tuning the target.

### Cited Findings
- **Lookahead Decoding**. arXiv 2402.02057. ICML 2024. Feb 2024. Authors: Fu, Bailis, Stoica, Zhang.
  - Method: Jacobi-style parallel n-gram generation plus a verification branch over an n-gram pool, with no draft model or datastore.
  - Result: up to 1.8x on MT-bench, and 4x with strong scaling on multiple GPUs for code completion.
  - Code: github.com/hao-ai-lab/LookaheadDecoding.
  - Stated limitations: none in the abstract.
  - Inferred weakness: it spends large extra FLOPs per step (window x n-gram size), so it degrades at larger batch or on compute-bound hardware.
  - Inferred weakness: the 4x figure needs multiple GPUs.
  - Sources: [A]. [arXiv](https://arxiv.org/abs/2402.02057), [GitHub](https://github.com/hao-ai-lab/LookaheadDecoding).
- **CLLMs (Consistency LLMs)**. arXiv 2403.00835. ICML 2024 (OpenReview id 2wqHY6OwNB). Mar 2024.
  - Method: fine-tune the target on Jacobi trajectories so that it converges to the fixed point in fewer Jacobi iterations.
  - Result [K]: about 2.4 to 3.4x on GSM8K, Spider and code tasks.
  - Code [K]: github.com/hao-ai-lab/Consistency_LLM.
  - Inferred weakness: it modifies target weights, so it is not lossless relative to the original model.
  - Inferred weakness: it needs per-domain trajectory generation and training.
  - Sources: [A]. [arXiv](https://arxiv.org/abs/2403.00835). The OpenReview PDF returned 403.
- **Jacobi decoding (Santilli et al., "Accelerating Transformer Inference for Translation via Parallel Decoding")**. arXiv 2305.10427. ACL 2023.
  - This is the origin of parallel Jacobi decoding.
  - The ID and details are [K] and were not verified this session.
  - Include it as background only.
- **REST**. arXiv 2311.08252. NAACL 2024. Nov 2023.
  - Method: draft tokens retrieved from a datastore via suffix-array match on the current context, verified with tree attention.
  - Setting for all results: 1x A6000, batch size 1.
  - Results on HumanEval with CodeLlama (2.7M-sample, 27 GB Stack datastore):
    - 7B: 2.36x greedy, 2.12x nucleus.
    - 13B: 2.27x greedy, 2.17x nucleus.
  - Results on MT-Bench with Vicuna (12 GB UltraChat datastore):
    - 7B: 1.69x greedy, 1.62x nucleus.
    - 13B: 1.77x greedy, 1.71x nucleus.
  - Code: github.com/FasterDecoding/REST.
  - Authors' limitations:
    - "performance... directly influenced by the accuracy and completeness of the datastore".
    - "Lack of in-context abilities" (e.g. personalized variable names).
  - Inferred weakness: a datastore of 12 to 27 GB plus CPU retrieval cost (96 cores in their setup).
  - Sources: [V]. [arXiv PDF](https://arxiv.org/pdf/2311.08252).
- **Prompt Lookup Decoding (PLD)**. GitHub only, no paper. Author: Apoorv Saxena (@apoorvumang). About Nov 2023.
  - Method: string-match the last n-gram against the prompt and copy what followed it as the draft.
  - Claimed "2x-4x" in the author's tweet, for input-grounded tasks.
  - Now built into HF transformers and vLLM as "n-gram speculation". vLLM added attribution via PR #59013 in Sep 2026.
  - Inferred weakness: near-zero gain when the output does not copy from the prompt (open-ended chat).
  - Inferred weakness: no peer-reviewed evaluation.
  - Sources: [A]. [repo](https://github.com/apoorvumang/prompt-lookup-decoding) (repo URL from vLLM PR), [vLLM PR](https://github.com/vllm-project/vllm/pull/59013), [tweet](https://x.com/apoorv_umang/status/1728831397153104255).
- **SuffixDecoding**. arXiv 2411.04975. NeurIPS 2025 spotlight. Nov 2024.
  - Method: model-free drafting from suffix trees over prompts and previous outputs, with adaptive speculation length. Targets agentic and repetitive workloads.
  - Result [K]: up to about 5x on agentic benchmarks.
  - Code [K]: integrated in Snowflake ArcticInference (vLLM plugin).
  - Inferred weakness: the benefit is concentrated in repetitive agentic or SQL workloads, with modest gains on open chat.
  - Sources: [P]. [arXiv](https://arxiv.org/abs/2411.04975), [project](https://suffix-decoding.github.io/).
- **SAM-Decoding**. arXiv 2411.10666. ACL 2025. Nov 2024.
  - Method: static (corpus) and dynamic (generated text) suffix automata find the exact longest suffix match in O(1) average per step. It falls back to or combines with generative drafters by match length.
  - Results with Vicuna-7B-v1.3, RTX A6000, batch size 1, greedy, fp16:
    - Spec-Bench: 1.84x alone, 2.27x with Token Recycling, 2.58x with EAGLE-2.
    - HumanEval: 2.29x.
  - Authors' limitations:
    - The combination rule is "a very heuristic approach".
    - "existing datasets do not well reflect the performance of retrieval-based methods in real usage".
    - The Python SAM implementation stores redundantly and is inefficient.
  - Inferred weakness: the standalone gain is modest. The headline 2.58x relies on EAGLE-2, which is a trained drafter.
  - Sources: [V]. [ACL](https://aclanthology.org/2025.acl-long.595.pdf).
- **Token Recycling ("Turning Trash into Treasure")**. arXiv 2408.08696. ACL 2025. Aug 2024.
  - Method: keep an adjacency matrix of top-k candidate tokens from past decoding steps, BFS a draft tree from it, and verify with tree attention. The matrix is updated continuously. Less than 2 MB of extra memory.
  - Results on a single A100-80GB, batch size 1, greedy:
    - Spec-Bench (Vicuna): 2.03x on 7B, 1.89x on 13B, 1.91x on 33B.
    - MBPP (CodeLlama): 2.26 to 2.34x.
    - Still about 2.0x at T=0.3, degrading at higher T.
  - Code: github.com/Luowaterbi/TokenRecycling.
  - Authors' limitation: the static tree structure.
  - Inferred weakness: it is sensitive to temperature.
  - Inferred weakness: the "+25% over Medusa" claim is against an old baseline. No EAGLE-2/3 comparison.
  - Sources: [V]. [ACL](https://aclanthology.org/2025.acl-long.338.pdf).
- **Cacheback**. arXiv 2511.21699. EMNLP 2025. Authors: Ma, Gim, Zhong (Yale).
  - Method: drafts only from LRU cache tables of token n-grams: a dynamic table plus a frozen table built from a corpus.
  - Setting: Spec-Bench (modified for fairness among stateful methods), on 1 to 4x RTX 4090.
  - Result on Vicuna-7B: average 1.86x, 2.42 mean accepted tokens.
  - Across Vicuna 7B, 13B and 33B, speedups range from about 1.29 to 1.84x by task.
  - It beats PLD, Lookahead, REST and Token Recycling, and is comparable to SAM.
  - Code: github.com/zyma98/Spec-Bench/tree/cacheback.
  - Authors' limitations:
    - The effect of the frozen-table corpus is unstudied.
    - Variation across GPUs and models is unexplored.
    - Evaluation is Spec-Bench only.
    - It is sensitive to hyperparameters, with no theory or auto-tuning.
  - Inferred weakness: many hand-set hyperparameters (LL, LC, FL, FC, TDL, CRT).
  - Inferred weakness: tested only on old Vicuna models.
  - Sources: [V]. [EMNLP](https://aclanthology.org/2025.emnlp-main.1581.pdf), [arXiv](https://arxiv.org/abs/2511.21699).

### Inferences
- Most of this group evaluates only on Vicuna or CodeLlama at batch 1 on Spec-Bench. That says little about modern instruct or reasoning models or about batched serving. SAM-Decoding's authors themselves say the benchmarks misrepresent real usage.
- These methods are complementary to trained drafters. SAM+EAGLE-2 and SuffixDecoding hybrids suggest that their best use is as a fallback or booster, not a replacement.

### Gaps
- SuffixDecoding's exact numbers and settings were not re-read this session.
- CLLMs' speedups and training cost were not re-read (the OpenReview PDF returned 403).
- PLD has no formal evaluation. Its creation date is inferred from the tweet only.

## (c) Multi-token prediction built into the target, used for speculation

### Takeaway
MTP heads trained with or into the target give self-speculation with no separate model:
- Gloeckle et al.: 3x on code at 7B.
- DeepSeek-V3: one MTP module, about 1.8x TPS.
- FastMTP: about 2x on a 7B in SGLang.
- Apple's "Your LLM Knows the Future": up to about 5x tokens per step, which is an acceptance-based figure, not wall-clock.

All of them require training.

### Cited Findings
- **Better & Faster LLMs via Multi-token Prediction**. arXiv 2404.19737. Apr 2024. Gloeckle et al., FAIR/Meta.
  - Venue: ICML 2024 [K]. The fetched v1 is an arXiv preprint.
  - Method: n independent output heads on a shared trunk with a shared unembedding, trained from scratch. The extra heads are used for self-speculative decoding.
  - Results with 7B models (setting: 4,200 sequences of 512 tokens, batch size 42, xFormers):
    - 4-token prediction: 3.0x on code, 2.7x on text, about 2.5 of 3 drafts accepted on code.
    - Byte-level 8-byte model: 6.4x.
  - Authors' limitations:
    - Worse than baseline for small models (it helps above about 3B).
    - How to choose n automatically is open.
    - Degradation on NLP multiple-choice benchmarks with 4-token prediction.
    - The optimal vocabulary size likely differs.
  - Inferred weakness: it needs pre-training with MTP and cannot be added to existing checkpoints.
  - Inferred weakness: the speedup is measured at batch 42 in a research stack, not a serving engine.
  - Sources: [V]. [arXiv PDF](https://arxiv.org/pdf/2404.19737).
- **DeepSeek-V3 MTP**. arXiv 2412.19437 (DeepSeek-V3 Technical Report). arXiv only. Dec 2024.
  - Method: a sequential MTP module of depth 1 is trained with the model. At inference it is repurposed as a speculative drafter.
  - Result [K]: second-token acceptance about 85 to 90%, about 1.8x TPS.
  - Inferred weakness: depth-1 MTP limits drafts to 1 extra token.
  - Inferred weakness: the gain is for one specific model family. FastMTP reports that vanilla MTP gives only 1.21x on MiMo-7B ([FastMTP](https://arxiv.org/pdf/2509.18362)).
  - Sources: [K]. ID listed as unverified in prior notes. [arXiv](https://arxiv.org/abs/2412.19437)
- **FastMTP**. arXiv 2509.18362. Sep 2025. Tencent.
  - Method: fine-tune a single shared-weight MTP head, applied recursively, on self-distilled data, with language-aware dynamic vocabulary compression.
  - Setting: MiMo-7B-RL, A10 24GB, SGLang, single batch, T=0, 7 benchmarks, K=3.
  - Result: 2.03x average versus AR, and +82% over vanilla MTP (1.21x). It is 1.81x without vocabulary compression.
    - Acceptance per draft position: about 80%, 56% and 36%.
    - Average accepted length: 2.66.
  - Training: about 1 day on one H20 server.
  - Code: github.com/Tencent-BAC/FastMTP.
  - Authors' note: smaller gains (1.84 to 2.07x) on general NLP tasks.
  - Inferred weakness: it depends on a base model that already ships an MTP head (MiMo).
  - Inferred weakness: vocabulary compression is language-specific.
  - Inferred weakness: evaluated on one small GPU at batch 1.
  - Venue: none found (arXiv only as far as I could tell).
  - Sources: [V]. [arXiv PDF](https://arxiv.org/pdf/2509.18362), [GitHub](https://github.com/Tencent-BAC/FastMTP).
- **Your LLM Knows the Future**. arXiv 2507.11851. Jul 2025. Samragh et al., Apple. Venue: none found.
  - Method: append k mask tokens, use gated LoRA that is active only on mask positions (preserving NTP outputs), add a 2-layer sampler head, and use "quadratic decoding".
  - Setting: Tulu3-8B, k=8, SFT-stage training on 8x A100.
  - Result: about 1.5 to 5.2x acceptance-based speedup (nearly 5x on math and code, about 2.5x on chat). This is NOT wall-clock.
  - Code: none.
  - Authors' limitations:
    - Applied only at the SFT stage.
    - The gated LoRA cannot be fused into base weights.
    - Diminishing returns with more masks.
    - Single model family.
  - Inferred weakness: the gap between acceptance-based and wall-clock speedup is unquantified. Unfused LoRA plus tree attention adds latency.
  - Sources: [V]. [arXiv PDF](https://arxiv.org/pdf/2507.11851).

### Inferences
- The MTP line blurs the "no separate drafter" boundary. All of it needs training (from scratch or by fine-tuning). The fair comparison is against EAGLE-style trained heads, which few of these papers run.
- Reported speedups use very different metrics (batch 42 throughput, acceptance-based tokens per step, batch-1 SGLang wall-clock). A thesis comparison should re-measure under one setting.

### Gaps
- DeepSeek-V3 MTP inference numbers and venues for FastMTP and "Your LLM Knows the Future" are unconfirmed (as far as I found they are arXiv only).
- No public code was found for the Apple paper.
- Not covered this round (possible extensions): ANPD/adaptive n-gram, PLD+, LogitSpec, Ouroboros, PaSS, Speculative Streaming, EESD (2406.03853), SpecPV (2512.02337), ConfLayers (2604.14612).
