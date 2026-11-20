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

## 3. Modern Hebrew

### 3.1 Quick look: Wikipedia only

The same recipe as Turkish, in one config. A fast first impression of how LFM3 takes to Hebrew:

```bash
lfm3-cpt -p rtx3090 -c examples/hebrew-wiki.toml prepare
lfm3-cpt -p rtx3090 -c examples/hebrew-wiki.toml train --merge
```

### 3.2 A real corpus: Wikipedia + web + English replay

Build the corpus once with `export`. Each source becomes `<name>.train.jsonl`, plus `<name>.eval.jsonl` with the
first documents held out. The budgets are in characters: `M` = million.

```bash
# Hebrew Wikipedia: encyclopedic
lfm3-cpt export wikimedia/wikipedia --subset 20231101.he --title-field title \
    -o corpora/he/wiki --max-chars 60M --eval-docs 100

# Hebrew web pages (FineWeb-2): news, forums, shops, blogs
lfm3-cpt export HuggingFaceFW/fineweb-2 --subset heb_Hebr \
    -o corpora/he/fineweb2 --max-chars 120M --eval-docs 100

# English replay (FineWeb-Edu), about 10-15% of the mix: keeps the model from forgetting English
lfm3-cpt export HuggingFaceFW/fineweb-edu --subset sample-10BT \
    -o corpora/en/fineweb-edu --max-chars 25M --eval-docs 100

# Grammar test: 2,330 Hebrew subject-verb agreement pairs from MultiBLiMP
lfm3-cpt multiblimp heb -o corpora/he/multiblimp-heb.jsonl
```

Each export streams only what it needs, so the multi-terabyte sources are not downloaded in full. Then check how
well the tokenizer covers Hebrew:

```bash
lfm3-cpt tokenizer-check corpora/he/wiki.train.jsonl
```

Then prepare and train:

```bash
lfm3-cpt -p rtx3090 -c examples/hebrew-mix.toml prepare
lfm3-cpt -p rtx3090 -c examples/hebrew-mix.toml train --merge
```

If the tokenizer verdict was "fair" or "poor", train the embeddings too, with a lower learning rate:

```bash
lfm3-cpt -p rtx3090 -c examples/hebrew-mix.toml -s train_embeddings=true train --lr 1e-4 --merge
```

`prepare` prints `train_tokens`. Divide by about 3,300 tokens/s to estimate the training time on the 3090: around
60M tokens is about 5 hours. For a first pass, halve the budgets above. With `-s gradient_checkpointing=false` it
is faster, if memory allows.

In the results, the held-out row is all three Hebrew sources together. The "out-of-distribution" row is the
English eval text: if it rises a lot, the replay share is too small.

**Optional: Knesset protocols** (384M tokens of spoken, parliamentary Hebrew). The dataset uses a loading script,
which current `datasets` no longer runs, so download its files and read them directly:

```bash
hf download HaifaCLGroup/KnessetCorpus --repo-type dataset \
    --include "protocols_sentences/*" --local-dir raw/knesset
(cd raw/knesset/protocols_sentences && unzip -q '*.zip')
lfm3-cpt export --data-files 'raw/knesset/protocols_sentences/**/*.json*' --inspect
```

`--inspect` shows the structure of the records. A protocol holds a list of sentences: pass that list's name
as `--field` and the sentence text's name as `--item-field`. For example, if the inspection shows
`protocol_sentences: list[…]` with `sentence_text: str` inside:

```bash
lfm3-cpt export --data-files 'raw/knesset/protocols_sentences/**/*.json*' \
    --field protocol_sentences --item-field sentence_text \
    -o corpora/he/knesset --max-chars 60M --eval-docs 50
```

The new files match `corpora/he/*.train.jsonl`, so the next `prepare` includes them automatically.

**Optional: HeDC4** (a cleaned Hebrew Common Crawl, 47.6 GB of CSV). Check the column name first, then export
a slice:

```bash
lfm3-cpt export HeNLP/HeDC4 --inspect
lfm3-cpt export HeNLP/HeDC4 --field text -o corpora/he/hedc4 --max-chars 60M --eval-docs 100
```

**Weighting the sources.** Every `*.train.jsonl` counts once. To weight one, list the files explicitly with a
repeat factor, e.g. Wikipedia twice:

```bash
lfm3-cpt -p rtx3090 -c examples/hebrew-mix.toml \
    -s 'train_files=corpora/he/wiki.train.jsonl@2,corpora/he/fineweb2.train.jsonl,corpora/en/fineweb-edu.train.jsonl' \
    prepare
```

### 3.3 Stage 2 and testing

```bash
lfm3-cpt -p rtx3090 -c examples/hebrew-mix.toml sft --merge
lfm3-cpt -c examples/hebrew-mix.toml chat --temperature 0
```

The SFT data in the config is machine-translated Alpaca. It teaches the model to answer, but expect some
translationese. Questions whose answers you can check:

| type this | meaning | a good answer contains |
|---|---|---|
| `מהי בירת צרפת?` | What is the capital of France? | **פריז** (Paris) |
| `כמה זה שתים עשרה ועוד שבע?` | What is twelve plus seven? | **19** / תשע עשרה |
| `באיזו טמפרטורה רותחים מים?` | At what temperature does water boil? | **100** |
| `תרגם לאנגלית: "מחר בבוקר נצא מוקדם לדרך."` | Translate into English | "Tomorrow morning we'll set out early" |
| `תן שלוש המלצות לתזונה בריאה.` | Give three tips for healthy eating. | three items, then it stops |

And the fact test without generation, on the original and your model:

```bash
for m in LiquidAI/LFM3-1.7B runs/lfm3-1.7b-cpt-he/merged; do
  lfm3-cpt rank $m --text "___ היא בירת צרפת." --candidate פריז --candidate ליון --candidate מרסיי
done
```

### 3.4 Keep English: mix the original model's own answers into stage 2

An all-Hebrew SFT set teaches the model to answer *everything* in Hebrew, and its short answers teach it to skip
the working on maths problems. The remedy is to let the original LFM3 answer English prompts (*distillation*), and
mix those answers into stage 2:

```bash
# English answers to human-written prompts (No Robots: 10,000 instructions, CC BY-NC 4.0)
CUDA_VISIBLE_DEVICES=1 lfm3-cpt distill HuggingFaceH4/no_robots \
    -o corpora/sft/en-no-robots.jsonl --max 3000

# Step-by-step maths (GSM8K, MIT): only answers whose final number matches the reference are kept
CUDA_VISIBLE_DEVICES=1 lfm3-cpt distill openai/gsm8k --subset main --verify-number \
    -o corpora/sft/en-gsm8k.jsonl --max 1500
```

Each run prints how many answers were kept, and why the others were dropped (unfinished, wrong language, wrong
number). `--resume` continues an interrupted run. Then repeat stage 2 with the mix, into a new folder so you can
compare it with the run before:

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/hebrew-mix.toml \
    -s 'sft_mix=corpora/sft/en-no-robots.jsonl@0.2,corpora/sft/en-gsm8k.jsonl@0.1' \
    sft --from runs/lfm3-1.7b-cpt-he/merged -o runs/he-sft-mix --merge
lfm3-cpt report runs/lfm3-1.7b-cpt-he runs/he-sft-mix
```

`@0.2` means 20% of the final training set. Some examples of each mixed source are held out for evaluation, so
the results table gains a row **answers in the question's language**. For the run without the mix, that is
where English questions answered in Hebrew show up. Then ask both models the English questions from § 3.3 again:

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt chat runs/he-sft-mix/sft/merged --temperature 0
```

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
| you want to see every library warning | `LFM3_CPT_SHOW_WARNINGS=1 lfm3-cpt …` (a few deprecations raised inside TRL/PEFT are hidden by default) |
| first download shows no progress | the bar appears once the file list is known; behind a proxy, check `HF_ENDPOINT` / `HTTPS_PROXY` |
