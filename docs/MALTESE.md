# Maltese with LFM3: a tested recipe

> **Status: tested** on an RTX 3090 (November 2026). Every command below was run as shown, and every number is
> measured.

LFM3-1.7B barely knows Maltese: perplexity 134 on held-out text, and on choosing the right word for a gap it is
at chance. One night of continued pre-training on 202M characters of curated Maltese changed that:

| | LFM3 (original) | **Maltese model** |
|---|---|---|
| perplexity, held-out Maltese (8 domains) | 133.7 | **5.56** |
| minimal pairs (word order, word choice) | 62.1% | **98.2%** |
| gap filling: right word among 8 plausible (chance 12.8%) | 13.2% | **54.8%** (60.8% with `--pmi 0.5`) |
| … right word in the top 3 | 39.5% | **85.5%** (87.0%) |
| English, unseen Wikipedia | 15.8 | **14.3** |

8 h 21 min, 90.7M tokens, one RTX 3090, peak 9 GB of GPU memory. English got *better*, not worse.

On the published Maltese benchmark (MELABench, § 9), the chat model built on it reaches 44.7 on Belebele reading
comprehension (original 25.7, chance 25; Gemma 2 2B 47.6) and 38.8 chrF on Flores-200 English → Maltese
(original 15.1; Salamandra 2B 34.1, EuroLLM 1.7B 51.1).

- [1. The data](#1-the-data)
- [2. Tokenizer and baseline](#2-tokenizer-and-baseline)
- [3. Export](#3-export)
- [4. Train](#4-train)
- [5. Results](#5-results)
- [6. Lessons learned](#6-lessons-learned)
- [7. Next steps](#7-next-steps)
- [8. Stage 2: chat](#8-stage-2-chat)
- [9. MELABench](#9-melabench)

---

## 1. The data

**[Korpus Malti](https://huggingface.co/datasets/MLRS/korpus_malti)** (MLRS, University of Malta) is the source:
curated Maltese, split by domain. **Licence: CC-BY-NC-SA 4.0**: fine for research; a model trained on it is not
for commercial use. Each record is a list of sentences, which `export` joins into a document.

Use the domain configurations, **not the default one**: the default (`shuffled`) is single sentences in random
order, from mixed sources, with no document context. `belles_lettres` is empty in the current version.

What each domain holds (full exports, measured):

| domain | documents | characters | what it is |
|---|---|---|---|
| parliament | 6,198 | 438.3M | committee notices and debates: very long documents, much boilerplate |
| press_mt | 77,557 | 147.7M | Maltese news |
| umlib_oar | 11,602 | 106.7M | University of Malta open-access repository: academic, partly OCR |
| blogs | 25,463 | 82.5M | informal |
| nonfiction | 2,140 | 20.6M | books, including the Bible |
| wiki | 2,980 | 9.8M | Maltese Wikipedia |
| theses | 18 | 1.5M | |
| speeches | 60 | 0.3M | |

To measure a domain's size, export it with a budget it can't reach (`--max-chars 1000M`) to a scratch folder.

**Is it really Maltese?** A quick count of Maltese markers (ċ ġ ħ ż, articles, `il-`) against English function
words found at most 0.1% mainly-English documents in any domain. Korpus Malti is clean in that respect.

**FineWeb-2** (`mlt_Latn`, 425,681 web documents, 587 MB) adds variety. **MultiBLiMP has no Maltese**: minimal
pairs are made from our own held-out text instead (`lfm3-cpt pairs`).

## 2. Tokenizer and baseline

```bash
lfm3-cpt export MLRS/korpus_malti --subset press_mt -o corpora/mt/press --max-chars 3M --eval-docs 200 --seed 7
lfm3-cpt tokenizer-check corpora/mt/press.train.jsonl
```

2.09 characters per token, verdict "fair": better prepared than for Hebrew (1.46). The Maltese letters are real
tokens (`rebħet → ['reb', 'ħ', 'et']`, `l-ewwel → ['l', '-', 'ew', 'wel']`), not byte pairs. The embeddings are
trained anyway, because the model knows so little Maltese.

**Apostrophes and hyphens are spelling in Maltese** (`ta'`, `il-kelma`, `għall-ewwel`): nothing is removed; only
typographic apostrophes are unified (`ta’` → `ta'`, `strip_marks = ["typographic_apostrophes"]`). The word rules
of `pairs`, `cloze` and `restore` keep inner apostrophes and hyphens, and form classes use the article (`il-`,
`l-`, `għall-`).

## 3. Export

A folder of its own (`corpora/maltese/`), so that earlier probes can't slip into the globs of the config. About
200M characters of Maltese, chosen for quality, with each domain's own held-out documents:

```bash
K="lfm3-cpt export MLRS/korpus_malti"
O=corpora/maltese
$K --subset press_mt   -o $O/press      --max-chars 60M   --eval-docs 200 --seed 7
$K --subset blogs      -o $O/blogs      --max-chars 30M   --eval-docs 100 --seed 7
$K --subset parliament -o $O/parliament --max-chars 30M   --eval-docs 10  --seed 7
$K --subset umlib_oar  -o $O/umlib      --max-chars 25M   --eval-docs 50  --seed 7
$K --subset nonfiction -o $O/nonfiction --max-chars 1000M --eval-docs 30  --seed 7
$K --subset wiki       -o $O/wiki       --max-chars 1000M --eval-docs 100 --seed 7
$K --subset theses     -o $O/theses     --max-chars 1000M --eval-docs 0   --seed 7
$K --subset speeches   -o $O/speeches   --max-chars 1000M --eval-docs 0   --seed 7
lfm3-cpt export HuggingFaceFW/fineweb-2 --subset mlt_Latn -o $O/fineweb2 --max-chars 25M --eval-docs 100 --seed 7

lfm3-cpt -c examples/maltese.toml pairs corpora/maltese/*.eval.jsonl -o corpora/maltese/pairs.jsonl
lfm3-cpt -p rtx3090 -c examples/maltese.toml prepare
```

"The source ran out before the budget was reached" means you got the whole domain, as asked. Parliament gets only
10 held-out documents because each is about 70k characters. English replay: `corpora/en/fineweb-edu.train.jsonl`
(25M characters) from the Hebrew recipe.

Result: 63,222 documents, 227M characters (202M Maltese, 25M English), **90.7M tokens**; 590 held-out documents
from every domain; 1,000 minimal pairs.

## 4. Train

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/maltese.toml evaluate LiquidAI/LFM3-1.7B   # the baseline
tmux new -s mt
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/maltese.toml train --merge
```

`examples/maltese.toml`: LoRA r=64 plus trained embeddings (327M trainable parameters), lr 1e-4, one pass,
3,266 tokens/s. The held-out perplexity was still at its best on the last step (5.56), and training fixed 371
minimal pairs while breaking 10.

## 5. Results

**English.** The config's out-of-distribution row (FineWeb-Edu held out, 18.65 → 15.90) is flattering: it comes
from the same source as the replay. The honest check is unseen English Wikipedia:

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/maltese.toml \
    -s ood_files=corpora/en-check/wiki.eval.jsonl -s data_dir=data/mt-encheck \
    evaluate runs/lfm3-1.7b-cpt-mt/merged --no-grammar
```

15.8 for the original model, **14.3** after Maltese training: no forgetting, a small gain.

**Gap filling.** The minimal pairs saturate at 98%; choosing among plausible words tells more. The proposer of the
hard wrong options is the Maltese model itself (the original knows too little Maltese to propose good ones), so
its own score is a conservative estimate:

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/maltese.toml cloze build \
    --vocab-file 'corpora/maltese/*.train.jsonl' \
    --proposer runs/lfm3-1.7b-cpt-mt/merged -o corpora/maltese/cloze.jsonl
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/maltese.toml cloze score corpora/maltese/cloze.jsonl \
    -m LiquidAI/LFM3-1.7B -m runs/lfm3-1.7b-cpt-mt/merged --pmi 0.5 --show 10
```

| 400 items, 7.9 candidates (chance 12.8%) | LFM3 | LFM3, α=0.5 | **Maltese** | **Maltese, α=0.5** |
|---|---|---|---|---|
| top-1 | 13.2% | 18.2% | 54.8% | **60.8%** |
| top-3 | 39.5% | 42.8% | 85.5% | **87.0%** |
| mean rank of the answer | 4.23 | 3.99 | 1.96 | **1.83** |
| beats the proposed options | 38.4% | 44.2% | 71.8% | **76.4%** |

The frequency correction (`--pmi 0.5`) helps by 6 points, as it did for Tannaitic Hebrew. Some misses are the test
data's fault: OCR garbage from scanned documents (`B FN wAġ Għaxar Snin Sezzjoni ___ ia`), a typo in the held-out
text where the model chose the correct spelling (`ż-Żghar` → `ż-Żgħar`), informal spellings (`pero'`). The rest
are mostly function words (`anqas` / `aktar`) and verbs that need more context.

**Its text:**

```bash
lfm3-cpt generate runs/lfm3-1.7b-cpt-mt/merged --temperature 0.7 --repetition-penalty 1.15 \
    --max-new-tokens 120 --prompt "Il-Gvern ħabbar li" --prompt "Malta hija gżira" --prompt "Fl-1964, Malta"
```

Fluent news prose; the article assimilates before "sun letters" (`id-diskors`, `t-teknoloġija`, `is-settur`), verbs
agree (`Malta bdiet`, `bdew jipproduċu`); occasional gender slips (`din il-Kumitat` for `dan`). Facts are invented,
as from any base model (a Maltese beer launched in 1964 by "OEM Productions Ltd."). A native speaker's judgement
is the next check worth having.

## 6. Lessons learned

1. **Survey the domains before choosing.** Sizes, a sample of each and a language count took minutes and
   decided the budget: news and books in full, parliament capped (438M characters, much of it boilerplate), the
   shuffled default left out.
2. **Same tools, different spelling rules.** The Hebrew cleaning removed quotation marks; Maltese needs its
   apostrophes and hyphens. Word rules that keep them made pairs and gap filling work unchanged.
3. **Build your own tests when there are none.** No MultiBLiMP for Maltese: pairs and gap-filling items made from
   held-out text worked as well as for Hebrew, and gap filling kept discriminating when the pairs saturated.
4. **Clean only what is dirty.** `lfm3-cpt clean` removed 3% of Korpus Malti (boilerplate, OCR garbage,
   mojibake) after two dry runs to tune it; a second night of training on the result changed nothing measurable
   (gap filling 60.8% → 58.5% at α=0.5, 54.8% → 56.0% raw: both within noise; English identical). Korpus Malti
   is already curated: the dry run's 97% kept was the answer. Keep the filter for web crawls and OCR.
5. **A low-resource language is not a lost cause for a small model.** From chance to 61% on gap filling and
   perplexity 134 → 5.6 in one night, with English intact.

## 7. Next steps

1. **Cleaner data** *(done: no measurable gain, § 6)*: `lfm3-cpt clean` removes OCR garbage (umlib_oar), boilerplate (parliament) and documents
   that lose most of their text, in the training files only; `examples/maltese-clean.toml` trains on the result and
   is scored on the same held-out text, pairs and gap-filling items:

   ```bash
   lfm3-cpt clean 'corpora/maltese/*.train.jsonl' --dry-run
   lfm3-cpt clean 'corpora/maltese/*.train.jsonl' -o corpora/maltese-clean
   lfm3-cpt -p rtx3090 -c examples/maltese-clean.toml prepare
   CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/maltese-clean.toml train --merge
   CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/maltese.toml cloze score corpora/maltese/cloze.jsonl \
       -m runs/lfm3-1.7b-cpt-mt/merged -m runs/lfm3-1.7b-cpt-mt-clean/merged --pmi 0.5
   ```
2. **Chat (stage 2)** *(done, § 8)*.
3. **MELABench** *(done for 6 of 11 tasks, § 9)*.
4. **Tokens for `ġ ċ ż`** before the next CPT run (§ 8).

## 8. Stage 2: chat

Human-written Maltese instructions don't exist openly (Aya has Maltese only in its templated, machine-translated
collection). The mix, as in the final Hebrew recipe (HEBREW.md § 4), from three sources converted with
`lfm3-cpt sft-source`:

```bash
# Alpaca with Maltese instructions; only the part of each answer after "Response in Maltese:" is kept
lfm3-cpt sft-source taco saillab/alpaca_maltese_taco -o corpora/sft/mt-alpaca.jsonl --max 10000
# real English-Maltese sentence pairs, different sentences for each direction
lfm3-cpt sft-source parallel MLRS/OPUS-MT-EN-Fixed --src en --tgt mt -n 1500 -o corpora/sft/opus
lfm3-cpt -p rtx3090 -c examples/maltese.toml prepare --sft
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/maltese.toml sft \
    --from runs/lfm3-1.7b-cpt-mt/merged -o runs/mt-sft --merge
```

- 10,000 TaCo answers kept of 11,912 records (1,912 had only a placeholder); 96.5% of the kept answers are
  Maltese (the rest: code, short answers, 31 in English).
- 13,243 training conversations, about 21% with English answers (`[sft]` in `examples/maltese.toml`: No Robots
  12%, GSM8K 5%, Maltese→English 4%, English→Maltese 5%). 828 steps, 28 minutes, peak 10.5 GB.

Results (100 held-out answers, before → after SFT):

| | before | after |
|---|---|---|
| answers that finish on their own | 60% | **88%** |
| closeness to the reference answer (chrF) | 24.9 | **46.6** |
| answer perplexity | 3.48 | **2.66** |
| right language (judged by source, below) | 85 / 100 | **99 / 100** |
| Maltese perplexity / English perplexity (forgetting) | 5.56 / 15.9 | 6.01 / 17.25 |

Right language, by source:

| source | expected | before | after |
|---|---|---|---|
| English questions (GSM8K, No Robots) | English | 24 / 29 | **29 / 29** |
| Maltese → English translation | English | 0 / 10 | **10 / 10** |
| English → Maltese translation | Maltese | 15 / 15 | 15 / 15 |
| Maltese Alpaca questions | Maltese | 46 / 46 | 45 / 46 |

Before SFT the model answered everything in Maltese, translation requests included; after it, it translates in
both directions and answers English in English. The one real miss: asked to translate `Bongu` into French, it
echoed it. (Three other Alpaca "misses" were right: English book titles, a JSON object, an answer too short for the
language detector.)

**A metric lesson.** The first table judged an answer's language against its *question*, and reported "answers
in Maltese" falling from 100% to 76%: every correct Maltese → English translation counted as a failure, and the
untrained model's habit of answering translation requests in Maltese counted as success. The report now judges
against the language of the *reference* answer ("answers in the reference's language").

**The cost:** about +8% perplexity in Maltese and English, much more than Hebrew's SFT (+0.25). If that matters,
a gentler run (`-s sft_learning_rate=5e-5`) should roughly halve it.

`lfm3-cpt languages runs/mt-sft/sft [more runs …]` prints the by-source table above for any SFT runs.

**A data artifact the model copied.** In chat it wrote `jista 'jinkludi` and `ta 'kontroll`: the machine
translation had detached the apostrophe of Maltese elision in 49% of the TaCo answers (about 13,500 times: `ta '`,
`jista '`, `tista '`, `ma '`, …). `sft-source taco` now joins them back (`ta' kontroll`) for words ending in -a,
leaving `u 'Kif…'` (an opening quote) and the article `'il` / `'l` (`mexa 'il quddiem`) alone.

**A tokenizer gap the model trips over.** `suġĠerimenti` is not in the data. LFM3's tokenizer has no token for
`ġ`: every `ġ` is two raw bytes (`suġġerimenti` → `su Ä ¡ Ä ¡ er imenti`), and capital `Ġ` (`Ä ł`) differs only in
the second byte, which the model sometimes gets wrong, systematically in `ġġ`. `chat`, `generate` and the SFT
evaluation now lower a capital that follows a lowercase letter inside a word (never right in Maltese) as the text
is written. The real fix, for the next CPT run: give the tokenizer tokens for `ġ ċ ż Ġ Ċ Ż`, which would also make
Maltese shorter in tokens.

**Second run** (repaired data, `-s sft_learning_rate=5e-5`, `-o runs/mt-sft2`):

| | run 1 (1e-4, old data) | **run 2 (5e-5, repaired)** |
|---|---|---|
| right language, by source (`lfm3-cpt languages`) | 97 / 100 | 97 / 100 |
| Maltese → English translated | 9 / 10 | 9 / 10 |
| answers finished / chrF | 88% / 46.6 | 89% / 45.7 |
| forgetting: Maltese / English perplexity | +0.45 / +1.35 | **+0.39 / +1.06** |

The lower learning rate cut forgetting by about a fifth, not half: most of the cost comes from the translated
data, not the rate. Run 2 is the chat model: the same behaviour, less forgetting, correct apostrophes. In chat,
answers to an English question in a fresh conversation come back in English; late in a long Maltese
conversation the model drifts towards Maltese (and translation requests deep in a conversation can fail): `/reset`
between topics.

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/maltese.toml chat runs/mt-sft2/sft/merged --temperature 0
```

Expect translationese (the Maltese training answers are machine-translated) and invented details.

## 9. MELABench

[MELABench](https://github.com/MLRS/MELABench) is the University of Malta's benchmark for Maltese, with published
scores for many models. It runs on the authors' fork of the LM Evaluation Harness, with fixed prompts (English
instructions, Maltese content, no chat template), zero-shot. Six of its 11 tasks load directly from the Hub; the
others need data from the authors or from GitHub.

### Setup

A separate environment: several tasks load their data with dataset scripts, which `datasets` 4 refuses, and
lfm3-cpt needs `datasets` 5.

```bash
cd /opt/halogen
git clone https://github.com/MLRS/MELABench
git clone https://github.com/KurtMica/lm-evaluation-harness melabench-harness
uv venv melabench-env --python 3.12
source melabench-env/bin/activate
uv pip install -e ./melabench-harness "datasets<4" "transformers>=5.20" accelerate sacrebleu torch
# the fork (v0.4.8) refers to a class transformers 5 removed
sed -i 's/    AUTO_MODEL_CLASS = transformers.AutoModelForVision2Seq/    AUTO_MODEL_CLASS = getattr(transformers, "AutoModelForVision2Seq", None) or transformers.AutoModelForImageTextToText/' \
    melabench-harness/lm_eval/models/hf_vlms.py
```

Two runs per model: the scoring tasks with automatic batch size (their longest prompts need batch 2 on 24 GB; a
fixed 8 runs out of memory), the generation tasks with batch 8. **No `--trust_remote_code`** (see below); the task
files pass it to the dataset loading themselves.

```bash
cd /opt/halogen/MELABench/prompting
R=/opt/halogen/lfm3-cpt/runs
for pair in lfm3:LiquidAI/LFM3-1.7B cpt-mt:$R/lfm3-1.7b-cpt-mt/merged sft-mt:$R/mt-sft2/sft/merged; do
  name=${pair%%:*}; model=${pair#*:}
  CUDA_VISIBLE_DEVICES=1 lm-eval --model hf --model_args pretrained=$model,dtype=bfloat16 \
      --tasks belebele_mlt,sib200_mlt,maltese_news_categories \
      --include_path tasks/1.0_english --num_fewshot 0 \
      --batch_size auto:4 --max_batch_size 64 --device cuda --output_path results-native/$name
  CUDA_VISIBLE_DEVICES=1 lm-eval --model hf --model_args pretrained=$model,dtype=bfloat16 \
      --tasks flores200_eng-mlt,opus100_eng-mlt,maltese_news_headlines \
      --include_path tasks/1.0_english --num_fewshot 0 \
      --batch_size 8 --device cuda --output_path results-native/$name
done
```

### Results

All on a 0–100 scale; published scores are the best models under 3B parameters in the MELABench paper, same
prompts and settings.

| task | metric | LFM3 | + Maltese CPT | **+ chat SFT** | published, under 3B |
|---|---|---|---|---|---|
| Belebele (reading comprehension, 4 options) | accuracy | 25.7 | 41.3 | **44.7** | Gemma 2 2B 47.6, EuroLLM 1.7B 24.1 |
| SIB-200 (topic, 7 classes) | macro F1 | 13.6 | **18.7** | 17.9 | mT0 1.2B 45.8, Gemma 2 2B 39.4, EuroLLM 14.0 |
| News Categories (multi-label) | macro F1 | 6.0 | 6.2 | **8.1** | EuroLLM 26.3, Gemma 2 2B 15.8 |
| Flores-200 English → Maltese | chrF | 15.1 | 11.1 | **38.8** | EuroLLM 51.1, Salamandra 2B 34.1 |
| OPUS-100 English → Maltese | chrF | 13.3 | 1.5 | **23.7** ¹ | EuroLLM 37.0, Salamandra 2B 23.4 |
| News Headlines (summarization) | Rouge-L | **6.0** | 0.0 | 0.4 | EuroLLM 9.4, mT0 1.2B 6.9 |

¹ The chat model trained on 1,500 sentences of the *training* split of this dataset (`MLRS/OPUS-MT-EN-Fixed`), so
it has a home advantage here; Flores-200 is the fair translation test.

What it shows:

- **Reading comprehension is the clearest gain:** from chance (25.7) to 44.7, close to Gemma 2 2B and far above
  EuroLLM 1.7B, a model trained with Maltese from the start. Most of it comes from continued pre-training.
- **Translation needs the chat stage.** The CPT model is a better Maltese model but a worse translator *under
  this prompt*: after "…Here is a translation to Maltese:" it usually writes a blank line, which ends the answer
  empty (4 of 5 Flores sentences in a check with `--log_samples`). It lost the original's habit of following
  the prompt format; SFT restores it, and the chat model reaches 38.8 chrF, above Salamandra 2B, below EuroLLM.
- **Classification with English labels stays weak.** SIB-200 and News Categories rank English topic names
  against a Maltese text; CPT and SFT barely move them.
- **Headlines failed for both trained models** (6.0 → 0.0 / 0.4). The task stops at the first line break;
  most likely the same blank-line start as in translation, not checked yet.

### Lessons

1. **`--trust_remote_code` broke generation silently.** In this harness the flag also applies to the model.
   LFM3's config points to LiquidAI's own modeling code, whose generation fails under `transformers` 5.24 while a
   single forward pass still works. Symptom: scoring tasks normal, every generation task near zero (chrF 0.1 for
   the chat model on Flores); without the flag, 38.8. lfm3-cpt also defaults to `trust_remote_code = true`; it
   works with the `transformers` version installed for training, but after an upgrade, check that `chat` still
   answers.
2. **Look at the raw outputs before trusting a zero.** Five items with `--log_samples` (two minutes) separated a
   harness problem (empty answers from a model that translates in chat) from a real weakness (the CPT model's
   blank line), after six hours of full runs.
3. **Size the batch per task type.** Scoring long prompts computes probabilities over the whole vocabulary at
   every position; automatic batch size handles it. Generation is one token per step; a fixed batch of 8 is far
   faster than the harness's automatic choice (which picked 1, sized on the longest article).
