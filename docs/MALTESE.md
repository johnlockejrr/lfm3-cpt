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

- [1. The data](#1-the-data)
- [2. Tokenizer and baseline](#2-tokenizer-and-baseline)
- [3. Export](#3-export)
- [4. Train](#4-train)
- [5. Results](#5-results)
- [6. Lessons learned](#6-lessons-learned)
- [7. Next steps](#7-next-steps)

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
4. **A low-resource language is not a lost cause for a small model.** From chance to 61% on gap filling and
   perplexity 134 → 5.6 in one night, with English intact.

## 7. Next steps

1. **Cleaner data:** `lfm3-cpt clean` removes OCR garbage (umlib_oar), boilerplate (parliament) and documents
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
2. **Chat (stage 2):** Maltese–English translation pairs plus English answers, as in HEBREW.md § 4.
3. **MELABench:** the University of Malta's benchmark of 11 Maltese tasks, to compare with published models.
