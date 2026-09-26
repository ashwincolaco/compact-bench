# COMPACT-Bench

**One budget axis for memory compaction in LLMs and agents.**

Methods that shrink what a language model keeps in memory are developed in
separate literatures, and each reports its savings in its own unit. KV-cache
eviction reports the share of tokens kept, quantization reports bits per entry,
prompt compression reports a token ratio, and agent memory reports the size of its
store. COMPACT-Bench prices all of them in one unit, **bytes of retained state per
token of history (BPT)**, computed from the model's cache dimensions, and then asks
four questions at matched budgets: how much accuracy a method keeps, whether exact
or lossy retention does better, whether the system can tell what it discarded, and
whether its stated confidence reflects what its memory still supports.

This repository is the reference implementation for an anonymous ICLR 2027
submission, together with the scripts that regenerate every table and figure in it.

- **Interactive results:** [project page](https://anonymous.4open.science/w/compact-bench-6A71/index.html)
  (source in `docs/`)
- **Paper:** [`docs/paper.pdf`](docs/paper.pdf)

## What it found

Across 21 settings of 16 open models from 0.5B to 24B parameters:

| Finding | Evidence |
|---|---|
| Once every stored byte is counted, the mechanisms differ widely at the same budget, and which one wins depends on the model. | On Qwen3-1.7B a 2-bit KV cache keeps 0.98 accuracy at 22% of the full cache, while eviction, prompt compression, and summaries keep at most 0.56 below 30%. On Qwen2.5-1.5B the same quantizer fails below 8 bits. |
| Tolerance to KV eviction follows model generation, not size. | In five Qwen2.5/Qwen3 pairs of matched size, the newer model keeps half its accuracy at 17 to 33 points less of its cache. Every model released from November 2024 onwards collapses at 33–69% of its budget, every earlier one at 76–86%. |
| Content-based scorers fall below random eviction only because they cannot see the question. | Appending the question before compression lifts SnapKV and TOVA by a median 0.52 in accuracy (up to 0.98); methods that ignore the question move by a median 0.02. |
| Eviction that scatters its losses leaves fragments that the model takes for the fact. | On the six models whose self-report passes both uncompacted controls, SnapKV's wrong answers are claimed as still answerable in up to 74% of cases, against at most 11% for StreamingLLM, which removes a fact whole. In those SnapKV failures 70% of the key's cache slots survived, against 24% when the model correctly reported the loss. |
| Whether exact memory beats a summary depends on how much the questions need. | On documents with four times as many facts, the point where an exact raw buffer overtakes a summary moves from 41% to 13% of the budget on Qwen2.5-1.5B. |

## Tasks

| Task | Question it answers | Main metrics |
|---|---|---|
| `frontier` | How much accuracy does each KV method keep at each budget? `--query-aware` puts the question in the context before compression; `--variant multikey` adds three same-format distractor needles. | accuracy vs BPT, collapse point |
| `mechanisms` | How do eviction, KV quantization, eviction followed by quantization, LLMLingua-2, and the model's own summary compare at matched bytes? | accuracy vs measured BPT |
| `reversibility` | At equal storage, does an exact but bounded raw buffer beat a rolling summary? `--dense` uses 48 facts per document instead of 12. | accuracy vs budget |
| `audit` | Does the system know what it lost? A yes/no probe and the answer with a stated confidence read the same compressed cache, next to uncompacted present and absent controls, with the survival of the fact's key and value tokens logged. | overclaim, stated confidence, token survival |
| `attribution`, `confidence` | The earlier two-call versions of the audit, kept for reference. | AUROC, overclaim, ECE |

## Install

```bash
pip install -e .                 # install a CUDA build of torch first
pip install -e ".[quant]"        # bitsandbytes, for 4-bit NF4 weights
pip install -e ".[mech]"         # LLMLingua-2, for the mechanisms task
```

Dependencies: `torch`, `transformers`, `kvpress` (NVIDIA), `datasets`,
`matplotlib`. The default model, `Qwen/Qwen2.5-1.5B-Instruct`, runs on an 8 GB
GPU in fp16. Every task takes `--model`, `--quant {fp16,bf16,nf4,prequant}`,
`--trials`, and `--out`; `compactbench <task> --help` lists the sweep flags.
`prequant` loads a checkpoint that was saved already in 4-bit.

## Quick start

```bash
compactbench frontier      --model Qwen/Qwen3-1.7B --quant bf16
compactbench mechanisms    --model Qwen/Qwen3-1.7B --quant bf16
compactbench audit         --model Qwen/Qwen3-1.7B --quant bf16
compactbench reversibility --model Qwen/Qwen3-1.7B --quant bf16 --dense
```

Each run writes one JSON file with its full configuration and a record or
aggregate per cell, so every statistic can be recomputed as a simple aggregate.

## Reproducing the paper

The runs came in three batches, all on the replication subset (contexts of 2,000
and 4,000 tokens, three needle depths, 8 trials, 48 generations per cell).

```bash
./scripts/scale_sweep.sh                            # batches 1 and 2, by setting tag
python scripts/queue.py scripts/jobs_v3.txt q.log   # batch 3, exactly as run
python scripts/v3_report.py runs                    # the paper's tables
python scripts/v3_report.py runs table              # the per-setting appendix table
python scripts/v3_figures.py runs figures           # mechanisms and collapse-vs-size figures
compactbench scale --runs runs/scale                # the appendix frontier figures
```

`scripts/queue.py` runs a job file one line at a time, skips jobs whose output
already exists, and re-reads the file after each job, so an interrupted batch
resumes where it stopped. The per-generation records behind the paper ship with
the submission's supplementary material under `runs/`.

## Project page

`docs/` is a static page (plain HTML, CSS, and one script, with no dependencies
and no external requests) that lets a reader explore the results: the mechanisms
on one axis, the collapse point by model and size with linked accuracy curves, the
question-aware ablation, the audit with its controls, and reversibility. Every
chart has a table view.

```bash
python scripts/export_site_data.py runs docs/data.js   # regenerate the data
python -m http.server -d docs 8000                       # preview locally
```

To publish it after the review period, enable GitHub Pages from `main` / `docs`.

## Implementation notes

- **BPT accounting.** Eviction that keeps a fraction *f* of tokens costs *f* of
  the full cache. A *b*-bit quantized cache also pays for its per-group scales and
  zero points and for any entries kept at full precision, so the quantizer here
  costs (*b* + 1.48)/16 of the full cache: 34% at 4 bits, not 25%. Prompt
  compression and summaries cost the measured ratio of kept to original tokens.
  See `compactbench/bpt.py` and `compactbench/presses.py`.
- **Question-agnostic compression.** KVPress compresses the context during
  prefill and appends the question afterwards, so every press, SnapKV included,
  commits to a retained set before it sees the question. `--query-aware` changes
  only what the scorer sees.
- **One cache per audit.** `audit` asks its probe and its answer from the same
  compressed cache through KVPress's multi-question path, so every press, the
  unseeded random press included, gives both questions the same retained set.
  A completion without a stated confidence is recorded as missing, not as 50.
- **KV quantization.** `QuantPress` quantizes keys per channel and values per
  token in groups of 32 and keeps the largest 1% of each layer's entries in fp16.
  Without those outliers, 4-bit keys break Qwen2.5, whose first layer holds key
  magnitudes near 300.
- **OLMo-2.** KVPress rebuilds queries for scoring without OLMo-2's QK-norm and
  mixes its float32 rotary tables with bf16 queries. `compactbench/models.py`
  corrects both inside the scoring hook only.
- **Multi-GPU.** KVPress cannot compress a model sharded across GPUs, so models
  that do not fit one card at 16 bits run in 4-bit NF4 on a single GPU.
- **Reversibility summary cap.** The summary may use its whole budget: generation
  is capped at the budget plus 40 tokens and then truncated to the budget.

## Citation

Citation details will be added after the review period.

## License

MIT.
