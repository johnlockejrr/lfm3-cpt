# lfm3-cpt

Continued pre-training (CPT) of [LiquidAI LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) on raw Turkish text with
TRL + LoRA, turned from the Liquid cookbook notebook into a CLI project. It has a live Rich training dashboard,
metric logging, before/after evaluation, hardware profiles and tests.

The recipe is the notebook's. You train on Turkish Wikipedia. Perplexity is measured on held-out Wikipedia and on
out-of-distribution FineWeb-2 web pages. Grammar is checked with the TurBLiMP minimal-pair benchmark, before and after
training. At the end the adapter is merged into a standalone checkpoint.

## Install

Install PyTorch for your GPU **first**, then this package (torch is not pinned to a build in `pyproject.toml`).

**NVIDIA (RTX 3090):**

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CUDA build you use
pip install -e ".[dev]"
```

**AMD Strix Halo (gfx1151, ROCm):** install the ROCm PyTorch build that already works on your machine (an
official `rocm*` wheel or AMD's gfx1151 nightlies), then `pip install -e ".[dev]"`. Before training, check that
`lfm3-cpt info` reports `backend ROCM`, `bf16 True`, and the amount of GPU-visible memory you expect. On an APU that
amount depends on your BIOS/kernel memory split.

## Quick start

```bash
lfm3-cpt info                              # backend, GPU, versions, suggested profile
lfm3-cpt -p rtx3090 prepare                # stream + cache the corpora in data/tr (once)
lfm3-cpt -p rtx3090 train --merge          # baseline eval → train → final eval → merge
lfm3-cpt report runs/lfm3-1.7b-cpt-tr      # before/after table again, any time
```

On the Strix Halo use `-p strix-halo`. For a smoke test on any machine, use
`-p cpu-debug --set model_id=<small local model>`.

## Commands

| command | what it does |
|---|---|
| `info` | CUDA/ROCm detection, memory, bf16/TF32, library versions, suggested profile |
| `show-config [--toml]` | resolved config; `--toml` output can be saved and passed back with `-c` |
| `prepare` | streams Wikipedia / FineWeb-2 / TurBLiMP and caches them; reused while data settings are unchanged |
| `train` | full pipeline with live dashboard; `--resume`, `--merge`, `--skip-baseline`, `--max-steps`, `--lr`, … |
| `evaluate [MODEL]` | perplexity + TurBLiMP for a base model, an adapter dir or a merged model; `--json-out` |
| `merge ADAPTER` | exact merge into a fresh bf16 base (CPU by default); `--push-to-hub REPO` |
| `generate [MODEL]` | continue Turkish prompts for a qualitative check; `--chat` uses the chat template |
| `report RUN…` | summary of one or several finished runs |

Every config key can be set from any level: `defaults < --profile < -c file.toml < --set key=value < command options`.

```bash
lfm3-cpt -p strix-halo -s train_articles=20000 -s lora_gate_proj=true train --lr 1e-4
lfm3-cpt -p rtx3090 show-config --toml > my-run.toml   # edit, then: lfm3-cpt -c my-run.toml train
```

## Profiles

| profile | rows × accum | eval batch | checkpointing | notes |
|---|---|---|---|---|
| `t4` | 2 × 4 | 2 | on | the original notebook (fp16) |
| `rtx3090` | 4 × 2 | 4 | on | bf16 + TF32 for the fp32 LoRA matmuls |
| `strix-halo` | 8 × 1 | 8 | **off** | plenty of memory, modest compute: trade memory for speed; no pinned memory on an APU |
| `cpu-debug` | 1 × 1 | 1 | on | tiny, 2 steps, fp32 |

All GPU profiles keep the notebook's effective batch of 8 rows × 2,048 tokens per optimizer step, so the
hyper-parameters and results are comparable across machines. If the Strix Halo runs out of GPU-visible memory, use
`-s gradient_checkpointing=true` or a smaller batch.

## Outputs (`runs/lfm3-1.7b-cpt-tr/`)

- `config.toml`: the exact resolved config of the run
- `metrics.jsonl`: every Trainer log line, with perplexities and peak memory
- `baseline.json`: the pre-training evaluation, cached, so `--resume` does not recompute it
- `checkpoints/`: adapter checkpoints every `save_steps` (the last `save_total_limit` are kept)
- `adapter/`: the final LoRA adapter + tokenizer
- `results.json`: config, device, data stats, baseline/final metrics, grammar flips, training stats
- `merged/`: standalone model (with `--merge`)

## Changes from the notebook

- **Exact merge in bf16.** The notebook merged into the model it had trained. On a T4 that model was loaded in
  fp16, so the "merged" checkpoint silently became fp16. `merge` loads a fresh bf16 copy of the base model (on CPU,
  so it never competes with the GPU) and merges with `safe_merge=True`.
- **Crash in the examples loop.** `next(...)` raised `StopIteration` whenever one of the 4 "most improved"
  phenomena had no wrong→right pair, for example when it had not improved at all. `compare_grammar` skips those
  phenomena and also counts pairs that training *broke*.
- **Token-weighted perplexity.** The Trainer's `eval_loss` is an average of per-batch means. Before/after
  perplexity now uses the exact total NLL divided by the total number of tokens, over the same wrapped 2,048-token
  rows as training.
- **OOD tracked during training.** Wikipedia and web pages are both passed as eval sets, so `metrics.jsonl` shows
  both curves, not just before/after.
- **The grammar test leaves the tokenizer alone.** The notebook set `tokenizer.padding_side = "right"` globally.
  Padding is now done by hand, on the right. That matters for LFM3: its causal short convolutions are only safe with
  right padding. Sentences are also sorted by length, so batches carry less padding.
- **LoRA targets are checked** against the model's real module names. Missing ones are reported, instead of
  failing deep inside PEFT.
- **Checkpoints and resume** (`--resume`). The notebook had `save_strategy="no"`.
- **Non-reentrant gradient checkpointing**, the recommended mode with PEFT.
- **Optional `--train-embeddings`.** This is the notebook's "next step" for a new language: the (tied) input
  embedding is fully trained via `modules_to_save` with `ensure_weight_tying`. It adds about 130M trainable
  parameters, so consider a lower `--lr` with it.
- **Cached, reproducible data.** Corpora are stored without EOS (it is added at use time), with a signature of the
  data settings in `metadata.json`. Changing only training settings does not trigger a re-download.

The model, LoRA rank/alpha, target modules, learning rate, schedule, wrapped packing and data sizes match the
notebook by default.

## Tests

```bash
pytest                 # ~1 min on CPU, fully offline (a tiny Llama + BPE tokenizer is built on the fly)
pytest -m "not slow"   # skip the end-to-end training/CLI tests
```

The end-to-end tests run the real pipeline: prepare, baseline, a few LoRA steps, checkpoints, resume, final eval,
merge, and then the `evaluate`/`merge`/`generate`/`report` commands on the result.
