# Quickstart

Exact commands, in order. Run them from the repository root: the paths in `examples/*.toml` are relative to it.
The examples use the RTX 3090 profile; on the AMD Strix Halo replace `-p rtx3090` with `-p strix-halo`.

Background for every step is in [docs/LANGUAGES.md](docs/LANGUAGES.md) and the [README](README.md).

---

## 0. Install and check

```bash
git clone https://github.com/johnlockejrr/lfm3-cpt.git
cd lfm3-cpt
python -m venv .venv
. .venv/bin/activate
```

PyTorch first, for your GPU:

```bash
# NVIDIA (RTX 3090)
pip install torch --index-url https://download.pytorch.org/whl/cu128
# AMD (Strix Halo, gfx1151): install the ROCm torch build that already works on that machine instead
```

Then the project:

```bash
pip install -e ".[dev]"
pytest -q                      # offline, CPU, under a minute
lfm3-cpt info                  # backend (CUDA / ROCM), memory, bf16, suggested profile
```

`lfm3-cpt info` should show `bf16 True` on both machines. Use the profile it suggests.

---

## 1. Smoke test (5 minutes): three Latin passages

This downloads LFM3-1.7B on first use (about 3.5 GB), trains 10 steps on a tiny public-domain sample, and runs
every part of the local-corpus pipeline. It proves your install works; the numbers mean nothing.

```bash
lfm3-cpt -p rtx3090 -c examples/latin-sample.toml tokenizer-check
lfm3-cpt -p rtx3090 -c examples/latin-sample.toml prepare
lfm3-cpt -p rtx3090 -c examples/latin-sample.toml train --merge
lfm3-cpt -p rtx3090 -c examples/latin-sample.toml rank \
    --text "Arma virumque ___, Troiae qui primus ab oris" \
    --candidate cano --candidate canto --candidate vidi
lfm3-cpt -p rtx3090 -c examples/latin-sample.toml rank \
    --input examples/latin-gaps.jsonl --output runs/latin-sample/gaps-ranked.jsonl
lfm3-cpt report runs/latin-sample
```

Clean up afterwards: `rm -rf data/latin-sample runs/latin-sample`.

---

## 2. Turkish: the full recipe (CPT → SFT → chat)

```bash
lfm3-cpt -p rtx3090 prepare --sft              # Wikipedia, FineWeb-2, TurBLiMP, Aya conversations
lfm3-cpt -p rtx3090 train --merge              # stage 1: CPT, before/after evaluation, merged model
lfm3-cpt -p rtx3090 sft --merge                # stage 2: Turkish SFT on top of the CPT model
lfm3-cpt report runs/lfm3-1.7b-cpt-tr          # both stages
lfm3-cpt chat                                  # talk to runs/lfm3-1.7b-cpt-tr/sft/merged
```

Both stages in one command: `lfm3-cpt -p rtx3090 train --sft --merge`.

More data than the notebook (CPT benefits from it):

```bash
lfm3-cpt -p rtx3090 -s train_articles=30000 -s data_dir=data/tr-30k train -o runs/tr-30k --merge
```

---

## 3. Modern Hebrew from Wikipedia

```bash
lfm3-cpt -p rtx3090 -c examples/hebrew-wiki.toml prepare
```

Read the tokenizer lines in the table that `prepare` prints (`chars_per_token`, `tokenizer_verdict`). The example
config already trains the embeddings. If the verdict is "good", you can switch that off with
`-s train_embeddings=false`.

```bash
lfm3-cpt -p rtx3090 -c examples/hebrew-wiki.toml train --merge
lfm3-cpt -p rtx3090 -c examples/hebrew-wiki.toml generate --prompt "ירושלים היא"
```

For stage 2, first check which languages the Aya dataset has:

```bash
python -c "from datasets import load_dataset; print(sorted(set(load_dataset('CohereLabs/aya_dataset', split='train')['language_code'])))"
```

If `heb` is listed:

```bash
lfm3-cpt -p rtx3090 -c examples/hebrew-wiki.toml sft --filter language_code=heb --merge
```

Otherwise use any Hebrew chat dataset: `sft --dataset <hub-id> --filter ""`.

---

## 4. Classical Latin from your own files

Put plain-text files in this layout, one work (or book) per file, paragraphs separated by blank lines:

```
corpora/latin/train/caesar/bellum-gallicum.txt
corpora/latin/train/cicero/in-catilinam.txt
...
```

Then:

```bash
lfm3-cpt -p rtx3090 -c examples/latin-local.toml tokenizer-check
lfm3-cpt -p rtx3090 -c examples/latin-local.toml prepare
lfm3-cpt -p rtx3090 -c examples/latin-local.toml train --merge
lfm3-cpt -p rtx3090 -c examples/latin-local.toml rank --input examples/latin-gaps.jsonl
```

`prepare` holds out 5% of the *files* (whole works) for evaluation and prints which ones (they are also recorded
in `data/latin/metadata.json`). To choose them yourself, uncomment `eval_files` in the config.

---

## 5. Targumic / Samaritan Aramaic

### 5.1 Lay out the corpus

```
corpora/aramaic/targum/genesis.txt  exodus.txt  leviticus.txt  numbers.txt   # target, training
corpora/aramaic/heldout/deuteronomy.txt                                       # target, evaluation only
corpora/aramaic/related/*.txt          # Biblical / Jewish Babylonian Aramaic, Syriac in Hebrew letters
corpora/hebrew/bible/*.txt             # Biblical Hebrew
corpora/aramaic/pairs.tsv              # your minimal pairs (step 5.3)
```

All files must be UTF-8 text, with paragraphs (e.g. chapters or verses) separated by blank lines. To use
another split, set `-s doc_split=line` (one document per line) or `-s doc_split=file`.
`examples/aramaic-local.toml` reads exactly this layout. Edit the paths there if yours differ.

### 5.2 Decide on vowel points, then check the tokenizer

The example config strips cantillation and niqqud (consonantal text). Compare both conventions on your files:

```bash
lfm3-cpt -c examples/aramaic-local.toml tokenizer-check
lfm3-cpt -c examples/aramaic-local.toml -s strip_marks=hebrew_cantillation tokenizer-check
```

The second command keeps the vowel points and drops only the accents. Keep the convention with more characters
per token unless you need the vocalization. The example config sets `train_embeddings = true`, because Hebrew
script is unlikely to be well covered by the vocabulary.

### 5.3 Write minimal pairs

```bash
cp examples/aramaic-pairs.template.tsv corpora/aramaic/pairs.tsv
```

Then add lines in the form `grammatical<TAB>ungrammatical<TAB>phenomenon`. Even 50–100 good pairs make a useful
grammar test. Write them in the same orthography as the corpus (normalization is applied to both).

### 5.4 Prepare and train

```bash
lfm3-cpt -p rtx3090 -c examples/aramaic-local.toml prepare
lfm3-cpt -p rtx3090 -c examples/aramaic-local.toml train --merge
lfm3-cpt report runs/lfm3-1.7b-cpt-arc
```

Watch `eval_wiki_ppl` (the held-out books) in `runs/lfm3-1.7b-cpt-arc/metrics.jsonl`:

```bash
grep eval_wiki_ppl runs/lfm3-1.7b-cpt-arc/metrics.jsonl
```

When it starts rising while the training loss keeps falling, the model is memorizing. The example config keeps
the last 10 checkpoints. Merge the one from before the rise (the step number is in `metrics.jsonl`):

```bash
lfm3-cpt merge runs/lfm3-1.7b-cpt-arc/checkpoints/checkpoint-200 -o runs/lfm3-1.7b-cpt-arc/merged-step200
```

Or train fewer epochs:

```bash
lfm3-cpt -p rtx3090 -c examples/aramaic-local.toml train -o runs/arc-2ep --epochs 2 --merge
```

Measure the effect of the related languages by training on the target alone:

```bash
lfm3-cpt -p rtx3090 -c examples/aramaic-local.toml \
    -s 'train_files=corpora/aramaic/targum/*.txt@4' -s data_dir=data/arc-only train -o runs/arc-only
lfm3-cpt report runs/lfm3-1.7b-cpt-arc runs/arc-only
```

### 5.5 Rank restorations of a lacuna

Write the text with `___` for the gap, and list the candidates. One gap per line in a JSONL file:

```json
{"id": "Gen-12-3", "text": "... ___ ...", "candidates": ["reading A", "reading B", "reading C"]}
```

```bash
lfm3-cpt -p rtx3090 -c examples/aramaic-local.toml rank --input lacunae.jsonl --output lacunae-ranked.jsonl
```

This uses the run's merged CPT model by default. `--by mean` sorts by log-probability per token, instead of
the total, when the candidates differ a lot in length.

### 5.6 Optional stage 2: Hebrew → Aramaic translation

Make verse-aligned JSONL files in the format of `examples/sft-translation.example.jsonl`: one verse per line,
Hebrew in `input`, Aramaic in `output`. Use separate books for training and evaluation:

```
corpora/aramaic/sft/he-arc.jsonl            # e.g. Genesis-Numbers
corpora/aramaic/sft/he-arc.heldout.jsonl    # e.g. Deuteronomy
```

```bash
lfm3-cpt -p rtx3090 -c examples/aramaic-local.toml prepare --sft
lfm3-cpt -p rtx3090 -c examples/aramaic-local.toml sft --merge
less runs/lfm3-1.7b-cpt-arc/sft/generations.jsonl     # reference vs model, verse by verse
lfm3-cpt chat runs/lfm3-1.7b-cpt-arc/sft/merged
```

Here chrF against the reference is the meaningful metric. The "answers in Aramaic" rate only checks the
script, so it cannot tell Hebrew from Aramaic.

---

## 6. Control experiments

```bash
# SFT without CPT: what does CPT add?
lfm3-cpt -p rtx3090 sft --from base -o runs/sft-only --merge
lfm3-cpt report runs/lfm3-1.7b-cpt-tr runs/sft-only

# evaluate any model on the current data (base, adapter or merged)
lfm3-cpt -p rtx3090 evaluate LiquidAI/LFM3-1.7B --json-out base.json
lfm3-cpt -p rtx3090 evaluate runs/lfm3-1.7b-cpt-tr/adapter --json-out cpt.json
```

---

## 7. Everyday operations

```bash
lfm3-cpt -p rtx3090 -c examples/latin-local.toml show-config          # every setting, resolved
lfm3-cpt -p rtx3090 show-config --toml > my-run.toml                  # start your own config
lfm3-cpt -c my-run.toml train --resume                                # continue after a crash or Ctrl-C
lfm3-cpt merge runs/lfm3-1.7b-cpt-tr/adapter                          # merge later (bf16, on CPU)
lfm3-cpt merge runs/lfm3-1.7b-cpt-tr/adapter --push-to-hub johnlockejrr/LFM3-1.7B-CPT-tr
lfm3-cpt prepare --force                                              # rebuild the data cache
```

Settings can come from four places. Later ones win: `--profile` < `-c file.toml` < `-s key=value` < command
options.

---

## Troubleshooting

| symptom | fix |
|---|---|
| CUDA / HIP out of memory | `--batch-size 2 --grad-accum 4` (same effective batch), or `-s gradient_checkpointing=true` |
| Strix Halo: less memory than expected in `info` | the GPU share of unified memory is set in BIOS / kernel parameters |
| `no files match '…'` | the paths in the TOML are relative to the directory you run from: run from the repo root |
| `the grammar files contain no pairs` | the pairs file has only comments: add pairs, or `-s grammar_files=` to disable |
| data seems stale | the cache is rebuilt when data settings or file contents change; `prepare --force` forces it |
| `no CPT result in …` (sft) | run `train` first, or `sft --from base` / `sft --from <model path>` |
