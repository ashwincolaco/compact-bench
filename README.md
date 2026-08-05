# COMPACT-Bench

**One budget axis for memory compaction in LLMs and agents.**

Methods that compact a model's memory — KV-cache eviction, quantization, prompt
compression, gisting, agent summarization — are evaluated in separate literatures
on incompatible axes (compression ratio at fixed perplexity, tokens saved at fixed
accuracy, recall at fixed state size). COMPACT-Bench puts them on one axis,
**bytes-per-token-of-history (BPT)**: the bytes of retained memory state per token
of original context. A KV evictor keeping 25% of tokens and a 4-bit quantizer land
on the same point; an agent's summary can be priced on the same scale.

Reference implementation for the benchmark proposed in
*What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction in LLMs
and Agents* (Colaco & Lahjouji, 2026).

## The four tasks

| Task | Question it answers | Metrics |
|---|---|---|
| `frontier` | how much accuracy does each method buy at a given BPT budget? | accuracy vs BPT |
| `reversibility` | at *equal* storage budget, does recoverable (raw, bounded) storage beat lossy (summary) storage? | recall vs budget |
| `attribution` | does the system **know what it dropped**? (post-compaction answerability self-report vs actual correctness) | AUROC, overclaim rate |
| `confidence` | is post-compaction confidence **calibrated**? | ECE, AUROC vs budget |

## Install

```bash
pip install -e .            # needs a CUDA torch; see below
```

Dependencies: `torch`, `transformers`, `kvpress` (NVIDIA), `datasets`,
`matplotlib`. Install a CUDA-matched torch first if pip's default doesn't fit your
system. Default model is `Qwen/Qwen2.5-1.5B-Instruct`, chosen to run on an 8 GB
consumer GPU (~3 GiB VRAM in fp16); pass `--model` to scale up.

## Run

```bash
compactbench frontier                          # accuracy-vs-budget sweep
compactbench reversibility                     # equal-budget recoverable vs lossy
compactbench attribution                       # self-knowledge of loss
compactbench confidence                        # calibration under compression
compactbench figures                           # render figures from runs/*.json
```

Every task takes `--model`, `--trials`, `--out`, and task-specific sweep flags;
`compactbench <task> --help` lists them. Results are JSON (per-record + summary).

## Reference results (Qwen2.5-1.5B-Instruct, RTX 4060)

**Frontier** (needle retrieval in natural filler, 1,395 generations): the full
cache scores 1.00; every KV method collapses to ~0 below ~25% of the full budget,
with query-conditioned and sink-aware policies holding accuracy to smaller budgets
and random eviction collapsing first.

**Reversibility** (equal token budget, compaction on overflow only):

| budget | irreversible (summary) | reversible (raw, bounded) |
|---|---|---|
| 10%  | 0.22 | 0.11 |
| 25%  | 0.47 | 0.25 |
| 50%  | 0.31 | 0.44 |
| 75%  | 0.44 | 0.69 |
| 100% | 1.00 | 0.92 |

The two strategies trade places: recoverable storage converts budget into recall
monotonically and leads once the budget is adequate; under tight budgets the lossy
summary leads because it spans the whole history. Reversibility dominates
*conditionally*, not universally.

**Attribution / confidence** (preview): at 75% compression the reference model
answers incorrectly while reporting ~0.95 confidence (ECE 0.95 vs 0.07 for the
full cache) — compacted systems do not know what they dropped. Full sweeps in
`runs/` after `compactbench attribution && compactbench confidence`.

## Design notes and caveats

- **BPT accounting** (`compactbench/bpt.py`): eviction keeping fraction *f* costs
  *f*·(full KV bytes/token); *b*-bit quantization costs *b*/16; prompt or summary
  compression to *m* of *n* tokens costs *m*/*n*; soft/gist tokens are priced by
  their stored vectors. Physical bytes upper-bound information content; the axis
  is a budget, not an entropy estimate.
- **Query-conditioned presses** (e.g. SnapKV) compress differently for the
  attribution probe than for the answer question, since each call sees a different
  trailing query. The attribution task therefore measures self-knowledge *under
  the system's own compaction policy*, not on a frozen retained set.
- Reference scale is deliberately small (a 1.5B model, one consumer GPU) so anyone
  can reproduce; the protocol is model-agnostic and the harness takes `--model`.

## Citation

```bibtex
@article{colaco2026keepforget,
  title   = {What to Keep, What to Forget: A Rate--Distortion View of Memory
             Compaction in LLMs and Agents},
  author  = {Colaco, Ashwin Gerard and Lahjouji, Nada},
  year    = {2026},
  note    = {arXiv preprint}
}
```

MIT license.
