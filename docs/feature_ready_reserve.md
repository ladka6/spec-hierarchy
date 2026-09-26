# Feature-ready job reserve

This experimental decoder keeps the existing DFlash weights and uses a bounded reserve
of alternative anchors to overlap drafting, middle verification, and target verification.
It measures actual wall-clock time; it does not use the virtual clock in `hier.py`.

## Algorithm

1. The target chooses the first token. The middle supplies prompt features.
2. A ready job contains a token prefix, an anchor token, middle features for everything
   **before** that anchor, and immutable middle/draft cache snapshots.
3. The draft worker extends the highest-priority ready job. While the middle verifies
   that proposal, the draft worker can extend a different ready job.
4. Middle verification produces its greedy continuation and, at low-confidence positions,
   alternate anchors using its second-choice token. Features before an alternate anchor
   are already valid, so DFlash can draft after it without another middle forward.
5. The target pulls an available continuation when idle. Middle-approved paths have
   priority, but a raw draft is eligible: middle approval is not a target dependency.
   `target_window` caps the check size; it is not a minimum fill barrier.
6. Only target results advance the committed prefix. Incompatible branches and late
   results are discarded. Compatible alternative continuations survive. When none survive,
   the middle rebuilds missing features from a retained snapshot of the common prefix.

The target may fall back to a one-token autoregressive step while proposals are absent.
This avoids forced target idleness but can outrun lower stages; benchmark both settings.
`--no-target-fallback` waits for a proposal instead. Neither policy uses artificial delays.

The reserve contains at most `reserve_size + 1` live paths, including the leading path.
At most one job per stage is in flight. Pruned in-flight work cannot be interrupted and
may retain snapshots until its completion. `max_ahead` stops launching drafting jobs
when their anchors are that far ahead; a block already launched may overshoot this bound.
Alternatives are ranked by products of middle top-2/top-1 probability ratios. This is a
heuristic priority, **not** a calibrated estimate of target agreement.

The initial implementation verifies chains independently. It does not batch alternative
branches or add middle tree attention. Those optimizations should be measured separately.

## Placement and resource costs

Each stage has an independent worker. CUDA work is synchronized before publishing its
result so that timings include device execution and transferred features are ready.
Distinct-device workers may overlap. Stages on the same device share an explicit lock;
their wait is recorded instead of assuming free concurrency. Consequently the two-GPU
configuration tests overlap with the target, while three GPUs test all three stages.

When the drafter is placed away from the target, it gets a local copy of the target's
embedding and output-head modules. Account for that memory in capacity planning. The
middle must have the target's hidden width, depth, and vocabulary (for example, a quantized
copy). No drafter retraining is needed. Current support is batch size one, greedy decoding,
CPU or CUDA, and single-device models (no model sharding).

Cache snapshots are copied before mutation. This is deliberately conservative and may be
expensive. Feature movement, cache copies, Python scheduling, and outstanding work drained
at completion all count toward measured runtime. No speedup is assumed or claimed.

## Run the paired ablation

Use the repository's existing model dependencies and pinned DFlash install from
`run_pilot.sh`. Add `pytest` if using the pytest commands below.

Two GPUs:

```bash
python scripts/exp9_reserve.py \
  --target-device cuda:0 --mid-device cuda:1 --draft-device cuda:1 \
  --reserves 0 1 2 4 8 --n 5 --max-new 256 --repeats 3 --trace
```

Three GPUs:

```bash
python scripts/exp9_reserve.py \
  --target-device cuda:0 --mid-device cuda:1 --draft-device cuda:2 \
  --reserves 0 1 2 4 8 --n 5 --max-new 256 --repeats 3 --trace \
  --output results/exp9_reserve_3gpu.json
```

`--prompt "Explain binary search."` uses a literal prompt instead of a benchmark dataset.
Repeat with `--no-target-fallback` and a different output file. Each prompt gets a target
AR correctness reference; every configuration must match exactly. Configuration order is
shuffled with a recorded seed. Output is saved after every run, including a failed match.

The `sync` baseline serializes the same backend with no reserve, checking after each
middle round. `reserve-0` allows independent workers but no alternate paths. Larger
reserves isolate the effect of preparing alternative jobs. This sync baseline is not the
fixed-window implementation in `pipeline.py`, and these ablations do not replace external
DFlash/DDTree/SSD performance baselines.

## Snellius launcher

On a login node, after checking out this branch:

```bash
# Once, if the existing environment/model/dataset cache is not already prepared:
bash snellius/setup.sh

# One job with three GPUs, one model per device:
bash snellius/submit_r9.sh

# Or use two GPUs, with the middle and draft stages sharing one device:
bash snellius/submit_r9.sh 2

# Optional: submit both placements as separate jobs:
bash snellius/submit_r9.sh both

# Short first run across all four datasets:
HSPEC_N=1 HSPEC_MAX_NEW=64 HSPEC_REPEATS=1 bash snellius/submit_r9.sh 3
```

Each job runs both target policies (`fallback.json` and `wait.json`), the synchronous
baseline, and reserve sizes 0/1/2/4/8 over gsm8k, math500, humaneval, and mt-bench.
Defaults are five prompts per dataset, 256 output tokens, and three repetitions.
The launcher prints the unique results directory and job ID; per-policy logs, timing
traces, and environment details are saved there. Failures propagate to SLURM.

The launcher requests one `gpu_a100` node with 18 CPU cores per requested GPU, following
the [SURF partition allocation](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius%2Bpartitions).
It reuses `setup.sh`'s Python modules, `$HSPEC_VENV` (default `~/venvs/hspec`), and
`$HF_HOME` (default `/scratch-shared/$USER/hf`), and runs offline on compute nodes.
It does not install packages or download checkpoints in the batch job.

Set `HSPEC_ACCOUNT` if your allocation needs an explicit SLURM account, `HSPEC_TIME`
to override the eight-hour limit, or `HSPEC_OUT` to choose the output parent directory.
`DRY_RUN=1 bash snellius/submit_r9.sh both` prints submission commands without submitting.
`squeue -u "$USER"` shows the queued/running jobs.

## Metrics and interpretation

- `confirmed_tok_s`: tokens committed after prefill / measured decode time including drain.
  The first token is excluded because prefill already produced it.
- `time_to_final_token`, `drain_seconds`: user-visible completion versus the cost of waiting
  for in-flight speculative calls to finish. Prefill and setup are reported separately.
- `stage_idle_fraction`: fraction of decoding time with no outstanding job for that stage.
  This includes policy/backpressure idle time; it is not a hardware utilization counter.
- `resource_wait_seconds`: time a dispatched worker waits for its shared-device lock.
  `service_seconds` includes transfers, cache operations, and device execution.
- `mean_ready_queue`, `ready_queue_empty_fraction`: availability of runnable feature-ready jobs.
- `branch_hits`, `reusable_tokens`: a prepared middle-verified branch matched a target
  correction, and the number of continuation tokens already available behind it.
- `recovery_delays`: correction receipt to submission of another nonempty target proposal.
  Fallback AR calls do not count. If decoding ends first, no completed delay is recorded.
- `compatible_fraction`: proposed token positions that match the eventual final prefix,
  including duplicate proposals in the denominator. This is a compatibility proxy, not
  attribution of which computation actually caused a token to be committed. Speculation
  beyond the final output counts as unused. Alternative-anchor construction is not included
  in this token counter; all its compute is included in middle service time.
  `unique_compatible_fraction` additionally counts each matching position only once in
  the numerator, so repeated compatible proposals cannot inflate that measure.
- `confirmation_bursts`: timestamp and size of each confirmed output burst, allowing
  inter-burst latency analysis. The function currently returns the completed output rather
  than streaming callbacks.
- `peak_allocated_bytes`: per-device PyTorch peak memory, including resident weights.
- `allocated_gpu_seconds_per_token`: assigned GPU count times decode wall time / timed
  output tokens. This measures allocation cost, not active kernel time or energy.

Keep model weights, devices, target policy, prompts, and output lengths fixed when comparing
reserve sizes. A useful result reduces time per confirmed token; lower idle time alone is
insufficient. Scheduler traces can establish overlap, but CUDA profiling is still needed
to understand kernel efficiency and memory-bandwidth contention.

## Tests

```bash
# Standard-library scheduler tests: no model packages required.
python -m unittest discover -s tests -p test_reserve_scheduler.py -v

# Real tiny Qwen/DFlash models: no checkpoint downloads required.
python -m pytest tests/test_reserve_scheduler.py tests/test_reserve_torch.py -q

# Existing decoder regression suite.
python tests/test_cpu.py
```

The tests cover worker overlap, same-resource serialization, prepared-branch recovery,
late results, unverified-draft target submission, stop tokens, short generation limits,
snapshot isolation, alternative-feature validity, and greedy equality. A three-CUDA-device
test is skipped when that hardware is unavailable. CPU tests do not establish GPU speedups.
