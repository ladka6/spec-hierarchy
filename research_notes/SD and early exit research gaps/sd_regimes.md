# Speculative decoding by regime and workload (2025 to Sept 2026): SOTA, open problems, A100-testable gaps

Scope note: ~18 tool calls. Items marked **[verified]** were confirmed via search/fetch this session. Items marked **[unverified ID]** are cited from prior knowledge; the arXiv ID or numbers were not re-checked here and should be confirmed before use. Novelty checks were done with direct searches of the gap wording; they are not exhaustive. Several 2026 papers were only seen as search-result titles (abstract not read); they are flagged "title only".

## Per-regime state of the art, open problems, and candidate gaps (objective)

### Takeaway
Every regime listed now has multiple 2025-2026 papers. The least crowded, most A100-friendly openings are: (a) an online draft-budget policy for MoE targets at batch > 1 (explicitly left open by 2609.22156, which itself ran on a single A100 80GB); (b) statistically calibrated (risk-controlled) lossy step acceptance for reasoning models; (c) SD for *sequential* hybrid linear-attention targets (Qwen3.5-style), where self-speculation collapses. Agent action-level speculation and long-context KV-compressed drafting are crowded.

### Cited Findings

**1. Reasoning models / long CoT**
- SpecReason (Pan et al., arXiv 2504.07891, NeurIPS 2025): a small model proposes reasoning steps, the large model scores them and only regenerates rejected ones; exploits that reasoning steps tolerate semantic, not token-exact, equivalence. Code: ruipeterpan/specreason. [verified title/venue] — [arXiv](https://arxiv.org/abs/2504.07891), [GitHub](https://github.com/ruipeterpan/specreason)
- SpecGuard ("From Tokens to Steps: Verification-Aware SD for Efficient Multi-Step Reasoning", arXiv 2604.15244, 16 Apr 2026): step-level verification from model-internal signals (attention grounding + log-prob) plus a self-consistency selector; Qwen2.5-Math 7B target / 1.5B draft; up to +3.6% accuracy and ~11% lower latency vs standard SD and reward-guided SD. Relies on an external sentence transformer for selection. [verified] — [arXiv](https://arxiv.org/html/2604.15244)
- SpecGuard positions itself against SpecReason and Reward-guided SD (RSD, Liao et al. 2025; RSD arXiv 2501.19324 [unverified ID]) — [arXiv](https://arxiv.org/html/2604.15244)
- SparseSpec (arXiv 2512.01278, 1 Dec 2025; Zhao, ..., Kasikci, Han, Stoica): self-speculation for reasoning models using sparse attention (PillarAttn) as the drafter, unified draft/verify scheduler, delayed verification, dynamic KV management; up to 2.13x throughput. Motivation: long CoT makes decoding memory-bound on KV reads. [verified] — [arXiv](https://arxiv.org/abs/2512.01278)
- "Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes" (arXiv 2607.26627, July 2026) exists; fetch returned no text, so content is unknown. This is the paper most likely to overlap any "lossy but bounded" proposal and must be read. [title only] — [arXiv](https://arxiv.org/abs/2607.26627)
- Other 2026 titles in this space (title only): "Acceptance Dynamics Across Cognitive Domains in SD" (2604.14682) — [arXiv](https://arxiv.org/html/2604.14682); "Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding" (2605.26558) — [arXiv](https://arxiv.org/html/2605.26558v1); "Acceptance-Aware Draft Model Training for SD" (2609.24150) — [arXiv](https://arxiv.org/abs/2609.24150); "Adversarial Prompts for Acceptance Collapse in SD" (2607.21804) — [arXiv](https://arxiv.org/html/2607.21804v1)
- Also relevant, not re-verified: Speculative Thinking (2504.12329), Lookahead Reasoning (2506.19830). [unverified IDs]

**2. Agents and tool calls**
- SuffixDecoding (arXiv 2411.04975, NeurIPS 2025 spotlight): model-free drafting from suffix trees over prompts and previous outputs; targets repetitive agentic workloads. [verified] — [arXiv](https://arxiv.org/abs/2411.04975), [project](https://suffix-decoding.github.io/)
- Speculative Actions (arXiv 2510.04371): lossless framework that predicts agent actions/tool calls with a faster model and executes speculatively. [verified title] — [arXiv](https://arxiv.org/html/2510.04371v1)
- 2026 follow-ups (titles verified via search): "Act While Thinking: Pattern-Aware Speculative Tool Execution" (2603.18897) — [arXiv](https://arxiv.org/html/2603.18897v1); "Speculative Interaction Agents: Asynchronous I/O and Speculative Tool Calling" (2605.13360) — [arXiv](https://arxiv.org/abs/2605.13360); "Cost-Aware Speculative Execution for LLM-Agent Workflows" (2606.07846) — [arXiv](https://arxiv.org/pdf/2606.07846)
- Speculate with Memory (arXiv 2607.12236, 14 Jul 2026, Salesforce): adds online memory (contrastive transition table, episodic retrieval, confusion tracker) to agent speculation; 19-39% relative gain in action-prediction accuracy (WebArena, VWA), up to 2.5x on observation prediction (ALFWorld). Limitations: evaluated by offline replay, not live deployment; web element-index prediction hard; memory interference across domains. [verified] — [arXiv](https://arxiv.org/html/2607.12236)
- AgentSpec: not located this session. [unverified]

**3. Long context**
- MagicDec (arXiv 2408.11049, ICLR 2025): shows that for long sequences and moderate-to-large batch, SD with a sparse-KV (StreamingLLM-style) self-drafter improves both latency and throughput because KV loading dominates. [verified] — [arXiv](https://arxiv.org/abs/2408.11049), [GitHub](https://github.com/Infini-AI-Lab/MagicDec)
- LongSpec (arXiv 2502.17421): lossless long-context SD with a memory-efficient draft (constant KV), efficient tree verification and position handling. [verified title] — [arXiv](https://arxiv.org/abs/2502.17421)
- QuantSpec (arXiv 2502.10424, ICML 2025 / PMLR v267): self-speculation with a hierarchical 4-bit quantized KV cache and 4-bit weights as drafter. [verified] — [PMLR](https://proceedings.mlr.press/v267/tiwari25b.html), [arXiv](https://arxiv.org/html/2502.10424v1)
- arXiv 2505.20776 appeared in the long-context search; likely SpecExtend, not confirmed. [unverified] — [arXiv](https://www.arxiv.org/pdf/2505.20776)
- 2026 KV-compression-as-drafter titles: "VeriCache: Turning Lossy KV Cache into Lossless LLM Inference" (2605.17613) — [arXiv](https://arxiv.org/html/2605.17613v1); "A Sparse Glimpse of the Whole: Train-Free Self-Speculative Decoding" (2607.27735) — [arXiv](https://arxiv.org/html/2607.27735v1). [title only]
- TriForce (2404.11912), ASPIRE, OWL: not re-checked this session. [unverified]

**4. Batching / serving under load**
- "An Interpretable Latency Model for SD in LLM Serving" (arXiv 2605.15051): decomposes latency into load-independent and load-dependent parts for prefill, draft and verify; SD speedups diminish as load grows; effective batch size is set by the server, not the user; validated on vLLM across drafter/verifier sizes, lengths, request rates, draft lengths and acceptance rates; extended to MoE. Limitation: vLLM-only validation. [verified] — [arXiv](https://arxiv.org/abs/2605.15051)
- Goodput-based SD serving (arXiv 2406.14066, SmartSpec, later TurboSpec naming [naming unverified]): picks draft length per step from a goodput estimate. [verified title] — [arXiv](https://arxiv.org/html/2406.14066v2)
- "Batch SD done right", SpecServe, MineDraft: not re-checked. [unverified]

**5. MoE targets**
- Utility-driven SD for MoE (arXiv 2506.20675): SD can slow MoE targets because verifying k tokens activates more distinct experts; proposes utility (progress vs. verification cost) to enable/disable speculation. [verified title] — [arXiv](https://arxiv.org/abs/2506.20675)
- "The Limits of Speculation: Bounding SD in MoE" (arXiv 2609.22156, Sept 2026): SSP formulation and offline oracle; Qwen3-Coder-30B-A3B + SGLang EAGLE-3 on **a single A100 80GB**, B=1; speedup plateaus at ~2.34x beyond draft length 6, mean accepted length caps ~2.1; verification time predictable from unique-expert count (MAPE 0.26-0.59%); optimal policy reduces to a constant cost-slope threshold. Stated limitations: B=1 only (expert sharing across sequences changes costs at B>1), math-only data, **no online policy** (explicitly future work). [verified] — [arXiv](https://arxiv.org/html/2609.22156)
- Same paper cites SP-MoE (2510.10302, SD + expert prefetching), cost-aware MoE SD (2607.12696, "Less Experts, Faster Decoding"), adaptive per-token verification for MoE (2605.00342). [verified as citations] — [arXiv](https://arxiv.org/html/2609.22156), [2607.12696](https://arxiv.org/abs/2607.12696), [2605.00342](https://arxiv.org/html/2605.00342v1)

**6. Quantized/pruned drafters and hybrid SSM targets**
- QSpec (EMNLP 2025): complementary quantization schemes, low-precision activation drafting verified with higher-precision weight-only quantization of the same model. [verified venue] — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.240/)
- Mamba Drafters (arXiv 2506.01206, Findings EMNLP 2025): external Mamba models as drafters for transformer targets. [verified] — [arXiv](https://arxiv.org/html/2506.01206)
- Component-Aware Self-SD in Hybrid LMs (arXiv 2605.01106): uses the SSM/linear-attention subgraph as a zero-cost internal drafter. Parallel hybrids (Falcon-H1) reach alpha=0.68 at k=2; sequential hybrids (Qwen3.5) only alpha=0.038 (18x gap), consistent from 0.5B to 3B. Wall-clock speedup **below 1.0x** due to Python implementation. Cites STree (tree verification for hybrids with external drafters) and RAD. [verified] — [arXiv](https://arxiv.org/html/2605.01106v1)
- "SD meets quantization", SpecMamba, STree IDs: not re-checked. [unverified]

**7. Temperature > 0, multi-draft, and lossy/semantic acceptance**
- FLy, Training-Free Loosely SD (arXiv 2511.22972; OpenReview JjoTg34YiU): entropy gate plus deferred token window to accept semantically correct mismatches; >99% of target accuracy retained, 2.81x avg on Llama-3.1-70B, 5.07x on 405B, 1.62x faster than EAGLE-3 out of domain. The fetched abstract says nothing about T>0 behaviour. [verified] — [arXiv](https://arxiv.org/abs/2511.22972), [OpenReview](https://openreview.net/forum?id=JjoTg34YiU)
- Related 2026 lossy/calibrated verification: "Calibrated SD: Frequency-Guided Candidate Selection" (2604.13634) — [arXiv](https://arxiv.org/html/2604.13634); MARS margin-aware verification (no arXiv ID found) — [ResearchGate](https://www.researchgate.net/publication/408334795_MARS_Unleashing_the_Power_of_Speculative_Decoding_via_Margin-Aware_Verification); 2607.26627 (see regime 1). [title only]
- SpecTr (2310.15141), Judge Decoding (2501.19309), Medusa typical acceptance (2401.10774): pre-2025 or not re-checked. [unverified IDs]

### Candidate gaps with novelty check

| # | Gap | Closest existing work found | Crowdedness | A100 feasibility |
|---|---|---|---|---|
| G1 | Online cost-slope policy for MoE draft length at B>1 (expert overlap across batch) | 2609.22156 (states it as future work); 2607.12696, 2605.00342, 2506.20675 may already contain online heuristics at B=1 | Medium; moving fast (4 papers in 15 months) | High: Qwen3-30B-A3B + EAGLE-3 fits one A100 80GB; SGLang off the shelf |
| G2 | Risk-controlled lossy step acceptance for reasoning (conformal/LTT-calibrated threshold with a stated accuracy-loss bound) | SpecReason (fixed score threshold), SpecGuard 2604.15244, 2607.26627 (unread) | Medium-high; must read 2607.26627 first | High: Qwen3-8B target, 0.6-1.7B draft, HF or vLLM |
| G3 | SD for sequential hybrid linear-attention targets (Qwen3.5-style): state rollback/checkpoint cost on rejection, drafter choice | 2605.01106 (<1.0x wall clock, alpha 0.038 for sequential), STree, Mamba Drafters | Low-medium | Medium: needs recurrent-state checkpointing; vLLM hybrid support must be checked |
| G4 | Position/phase-dependent acceptance along long CoT (e.g., reflection tokens) to drive adaptive draft length | 2604.14682 (acceptance by domain), SparseSpec | Medium | High: measurement study with HF/vLLM logs |
| G5 | Temperature > 0 semantic/loose acceptance with a distribution-level guarantee | FLy 2511.22972, Calibrated SD 2604.13634, MARS | Medium-high | High |
| G6 | Agent speculation evaluated online (not replay) | Speculate with Memory 2607.12236 (replay only), Speculative Actions 2510.04371, 2603.18897, 2605.13360, 2606.07846 | High | Medium (needs agent env infra, not GPU) |
| G7 | Long-context KV-compressed self-drafting | MagicDec, QuantSpec, LongSpec, SparseSpec, VeriCache, Sparse Glimpse | High | High, but novelty low |

### Inferences
- G1 is the cleanest opening: the gap is stated verbatim in a Sept 2026 paper whose own setup was a single A100. Risk: concurrent 2607.12696 / 2605.00342 must be read to confirm they do not already give an online B>1 policy.
- G3 has the clearest negative result to build on (alpha 0.038, <1.0x) but engineering risk is higher.
- For Ege's hierarchical SD work, G1 and G3 are natural fits: a multi-level draft hierarchy changes the cost-slope per level (MoE), and an SSM subgraph could be an intermediate level for hybrid targets. This is my inference, not a claim from any source.

### Gaps
- Could not read 2607.26627 (empty text on fetch); it is the main novelty risk for G2 and G5.
- Did not verify: AgentSpec, TriForce, ASPIRE, OWL, SpecExtend, Batch SD done right, SpecServe, TurboSpec naming, MineDraft, SpecMamba, STree IDs, "SD meets quantization", SpecTr, Judge Decoding. Specific numbers for these are absent from these notes on purpose.
- Did not check the hemingkx/SpeculativeDecodingPapers list or the Xia/Hu surveys due to the tool budget.

## Which regimes still lack a convincing method, and why?

### Takeaway
MoE at batch > 1, sequential hybrid SSM targets, and lossy verification with guarantees lack convincing methods; the evidence is in the authors' own limitation sections. Serving under load has models and heuristics but only vLLM-validated ones.

### Cited Findings
- MoE: "No online policy provided"; analysis only at B=1; inter-sequence expert sharing at B>1 changes overhead; math-only evaluation. — [2609.22156](https://arxiv.org/html/2609.22156)
- MoE: speedup plateaus ~2.34x and accepted length ~2.1 even with larger budgets on Qwen3-30B-A3B. — [2609.22156](https://arxiv.org/html/2609.22156)
- Hybrid SSM: wall-clock speedups below 1.0x; sequential hybrids get alpha=0.038. — [2605.01106](https://arxiv.org/html/2605.01106v1)
- Serving: SD speedups diminish with load; effective batch size is emergent; validated only on vLLM. — [2605.15051](https://arxiv.org/abs/2605.15051)
- Agents: best recent memory-augmented speculation evaluated offline only. — [2607.12236](https://arxiv.org/html/2607.12236)
- Reasoning: step-level verifiers still depend on extra components (target as judge in SpecReason, sentence transformer in SpecGuard). — [2604.15244](https://arxiv.org/html/2604.15244), [2504.07891](https://arxiv.org/abs/2504.07891)

### Inferences
- Long context is the most "solved" regime for lossless SD (many KV-sparse/quantized self-drafters); what remains is mostly engineering and batch/load interaction, not a missing method.
- Lossy verification for reasoning has many methods but, as far as found, no method with a calibrated accuracy-loss bound; whether 2607.26627 closes that is unknown.

### Gaps
- Did not read limitation sections of SparseSpec, LongSpec, QuantSpec, FLy in full.

## Which gaps can be studied on single-node A100s without building a serving engine?

### Takeaway
G1 (MoE policy), G2 (calibrated step acceptance), G4 (acceptance dynamics along CoT) and G5 (T>0 loose acceptance) are doable with SGLang/vLLM/HF off the shelf on 1-4 A100s. G3 needs custom state rollback code. Batch/load studies can reuse vLLM plus the 2605.15051 latency decomposition.

### Cited Findings
- 2609.22156 ran its whole study on one A100 80GB with SGLang EAGLE-3 and CUDA graphs. — [arXiv](https://arxiv.org/html/2609.22156)
- SpecReason has a public PoC repo; SpecGuard used 7B/1.5B models. — [GitHub](https://github.com/ruipeterpan/specreason), [2604.15244](https://arxiv.org/html/2604.15244)
- 2605.15051 validated its model with vLLM measurements, so the same measurement setup is reproducible. — [arXiv](https://arxiv.org/abs/2605.15051)
- 2605.01106 used research-grade Python and 0.5-3B models, showing the acceptance-rate question is studyable cheaply, but wall-clock claims need kernels. — [arXiv](https://arxiv.org/html/2605.01106v1)

### Inferences
- Acceptance-rate and accuracy results can be done in HF; any latency/throughput claim should use vLLM or SGLang, since HF timings will not convince reviewers.
- FLy's headline numbers are at 70B/405B; reproducing at 8B on A100 is feasible but speedups will be lower because the target/draft cost ratio is smaller (inference).

### Gaps
- Did not verify current vLLM/SGLang support for SD on Qwen3.5-style hybrid targets.

## Where do 8B-scale results transfer credibly?

### Takeaway
Acceptance-rate, accuracy and memory-bound (long-context, long-CoT) findings transfer reasonably from 8B; MoE results can be run on a real MoE (30B-A3B) on one A100; batch/load speedup numbers transfer least.

### Cited Findings
- Hybrid self-speculation acceptance gap was scale-invariant from 0.5B to 3B. — [2605.01106](https://arxiv.org/html/2605.01106v1)
- MoE verification cost is driven by unique-expert count, a property of the architecture more than scale, and was measured on a 30B-A3B model on one A100. — [2609.22156](https://arxiv.org/html/2609.22156)
- SD speedups depend on load and effective batch size, both of which change with model size and hardware. — [2605.15051](https://arxiv.org/abs/2605.15051)
- FLy's largest gains appear at 405B (5.07x) vs 70B (2.81x), showing speedups grow with target size. — [2511.22972](https://arxiv.org/abs/2511.22972)

### Inferences
- Report acceptance length and accuracy as the primary metric at 8B, and wall-clock only as secondary with the engine named.
- MoE work (G1) avoids the transfer problem entirely because the target is a real MoE.

### Gaps
- No source found that systematically studies SD speedup transfer from 8B to 70B+; this is itself a possible (but low-impact) measurement contribution.
