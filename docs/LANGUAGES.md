# Teaching LFM3 other languages

The project was built for Turkish, but the method (continued pre-training on raw text, then an optional SFT
pass) works for any language. This guide covers what changes when you move to another language or period,
and the settings for each case. For copy-paste command sequences, see [QUICKSTART.md](../QUICKSTART.md).

- [1. Will it work for my language?](#1-will-it-work-for-my-language)
- [2. Where the training text comes from](#2-where-the-training-text-comes-from)
- [3. Normalization](#3-normalization)
- [4. Checking the tokenizer](#4-checking-the-tokenizer)
- [5. Grammar tests with your own minimal pairs](#5-grammar-tests-with-your-own-minimal-pairs)
- [6. Ranking restorations: lacunae, conjectures, variants](#6-ranking-restorations-lacunae-conjectures-variants)
- [7. Stage 2 for other languages](#7-stage-2-for-other-languages)
- [8. Very small corpora](#8-very-small-corpora)
- [9. Settings reference](#9-settings-reference)

---

## 1. Will it work for my language?

Three things decide how CPT will go:

| question | why it matters | how to check |
|---|---|---|
| Does the model already know the language? | Strengthening a weak language is much easier than learning one from nothing | LFM3 officially supports 16 languages (incl. English, Arabic, Russian, Hindi); others may still be partly known from web data. Try `lfm3-cpt generate --prompt "…"` |
| Does the tokenizer cover the script? | Poor coverage splits words into single characters or bytes: slow training, short effective context | `lfm3-cpt tokenizer-check` ([§4](#4-checking-the-tokenizer)) |
| How much text is there? | Millions of tokens teach a language; thousands teach a style | `prepare` prints `train_tokens` |

| language | known to LFM3 | script coverage | text | what to expect |
|---|---|---|---|---|
| Turkish | weakly | Latin: good | large (Wikipedia, web) | the original recipe; clear gains |
| Modern Hebrew | not officially | check it | large | like Turkish, if the tokenizer covers Hebrew; train the embeddings if it does not |
| Modern Standard Arabic | yes | good | large | nothing to teach; use CPT for a *variety* instead (Classical, dialects) |
| Classical Arabic | partly | good | medium (e.g. OpenITI) | a register shift; local files ([§2.2](#22-your-own-files)) |
| Judeo-Arabic | probably not | Hebrew script: check it | small | treat as a small corpus ([§8](#8-very-small-corpora)) |
| Classical Latin | probably partly (web) | Latin: good | small-medium | strengthen; mind Neo-Latin Wikipedia ([§2.1](#21-the-hub-wikipedia--fineweb-2)) |
| Targumic / Samaritan Aramaic | no | Hebrew or Samaritan script: check it | very small | a good *language model* for scoring and restoration; see [§8](#8-very-small-corpora) |

---

## 2. Where the training text comes from

### 2.1 The Hub: Wikipedia + FineWeb-2

The default. Change the language with two settings:

```toml
target_language = "he"
wiki_config = "20231101.he"   # wikimedia/wikipedia subset: 20231101.<wiki language code>
ood_config = "heb_Hebr"       # HuggingFaceFW/fineweb-2 subset: <ISO 639-3>_<script>, out-of-distribution eval
```

`ood_config` must name a FineWeb-2 subset that exists; set `ood_pages = 0` to skip the OOD evaluation.
Wikipedia is a good source for living languages. For Latin it is mostly modern Neo-Latin and stubs, so it
teaches Neo-Latin style, not Cicero. Use your own files for historical periods.

### 2.2 Your own files

Setting `train_files` switches stage 1 to local files. Wikipedia and FineWeb-2 are then not used.

```toml
train_files = ["corpora/aramaic/targum/*.txt@4", "corpora/hebrew/bible/*.txt"]
eval_files  = ["corpora/aramaic/heldout/*.txt"]   # optional
ood_files   = ["corpora/aramaic/other-genre/*.txt"] # optional
doc_split = "paragraph"
```

- **Patterns** are globs (`**` recurses), relative to the directory you run from. A pattern that matches nothing
  is an error, not a silent empty corpus.
- **`@N`** repeats the matched files N times (upsampling). This is how you weight a small target corpus
  against larger related corpora ([§8](#8-very-small-corpora)).
- **Formats:** `.jsonl`/`.json`/`.ndjson` files have one JSON record per line; the document is the `text_field`
  (default `"text"`). Anything else is read as UTF-8 text and split by `doc_split`:
  - `file`: the whole file is one document
  - `paragraph`: blank-line separated
  - `line`: one document per non-empty line

  Documents are concatenated and cut into `max_length`-token rows for training, so long documents are not
  truncated; the split only decides where EOS tokens go.
- **`file_min_chars`** drops documents shorter than that (default 0).
- **Evaluation text.** With `eval_files`, those files are the held-out set. Otherwise, `eval_holdout = "files"`
  (default) holds out `eval_fraction` of the *files*, whole. `"documents"` holds out individual documents
  instead. Whole files are the honest choice for literature: a held-out paragraph from a book you trained on
  still shares its names, formulae and style, and flatters the perplexity. Holdout happens before upsampling,
  so a repeated document can never be in both sets. `prepare` prints what was held out.
- **The cache** (`data_dir`) records every file's size and modification time. Editing a file, adding one, or
  changing any data setting rebuilds it on the next `prepare`/`train`. Training-only settings do not.

### 2.3 Building a large corpus from Hub datasets

A good corpus mixes registers (encyclopedic, web, spoken, literary) and keeps some of the model's original
languages in the mix. `lfm3-cpt export` builds such a mix one source at a time, from any Hub dataset or from
raw downloaded files. Each source becomes local JSONL files that `train_files` reads.

```bash
lfm3-cpt export HuggingFaceFW/fineweb-2 --subset heb_Hebr --inspect                       # 1. look
lfm3-cpt export HuggingFaceFW/fineweb-2 --subset heb_Hebr -o corpora/he/fineweb2 --max-chars 120M   # 2. export
```

- **`--inspect`** prints the structure of the first records. Datasets name their text field differently, so look
  before exporting.
- **`--field`** is the text field, as a dotted path for nested records. If it holds a *list*, the items are joined
  with `--join` (default: newline). For a list of records, such as the sentences of a protocol,
  **`--item-field`** names the text inside each item. **`--title-field`** puts a title above the text.
- **Budgets:** `--max-chars` (e.g. `200M`) or `--max-docs` (e.g. `50K`) for the training file; whichever is reached
  first ends the export. Only that much is streamed, so a multi-terabyte dataset costs only its slice.
- **Evaluation:** the first `--eval-docs` documents go to `<prefix>.eval.jsonl` and the rest to
  `<prefix>.train.jsonl`, so the two never overlap. `--max-docs 0` writes only the eval file.
- **`--shuffle-buffer`** (default 10,000) shuffles while streaming, so a dataset sorted by date or topic doesn't
  give you only its first slice. `--skip N` takes a later, disjoint slice.
- **Raw files:** datasets that need a loading script (no longer supported by `datasets`) can be downloaded with
  `hf download` and read with `--data-files 'path/**/*.json*'` (and `--builder` if the extension is ambiguous).

Then point the config at the files: every source's train file in `train_files` (with `@N` weights if you want),
and every eval file in `eval_files`:

```toml
train_files = ["corpora/he/*.train.jsonl", "corpora/en/*.train.jsonl"]
eval_files  = ["corpora/he/*.eval.jsonl"]
ood_files   = ["corpora/en/*.eval.jsonl"]    # English held-out text: measures forgetting
```

**Replay.** Training only on the new language slowly erodes what the model knew in its original languages,
including facts it learned in English. Keeping 10–15% of the characters in English (FineWeb-Edu is a good
general source) is the standard remedy. Using the English eval file as `ood_files` makes the "out-of-distribution"
row of the results table a direct forgetting check.

**How big?** `prepare` reports the token count. Training time on the RTX 3090 is roughly tokens ÷ 3,300 seconds:
about 1 hour for 12M tokens, 5 hours for 60M, 3.5 days for 1B. `examples/hebrew-mix.toml` is a worked
example: about 200M characters from Wikipedia, FineWeb-2 and English replay, optionally with Knesset protocols
and HeDC4 ([QUICKSTART § 3](../QUICKSTART.md#32-a-real-corpus-wikipedia--web--english-replay)).

---

## 3. Normalization

Every text the model sees or is scored on gets the same normalization: corpus, eval text, minimal pairs, SFT
conversations, `rank` candidates and `generate` prompts.

```toml
unicode_form = "NFC"                                     # NFC (default) | NFD | NFKC | NFKD | none
strip_marks = ["hebrew_cantillation", "hebrew_points"]   # any of the sets below
```

| `strip_marks` | removes | keeps |
|---|---|---|
| `hebrew_cantillation` | accents (te'amim) U+0591–05AF | vowels, maqaf, sof pasuq |
| `hebrew_points` | niqqud, dagesh, shin/sin dots (U+05B0–05BD, 05BF, 05C1–05C2, 05C4–05C5, 05C7) | maqaf, paseq, sof pasuq |
| `arabic_diacritics` | harakat, tanwin, shadda, sukun, superscript alef, Quranic annotation signs | letters |
| `arabic_tatweel` | kashida (U+0640) | letters |
| `syriac_points` | Syriac vowels and diacritics (U+0730–074A) | letters |
| `samaritan_marks` | Samaritan vowel signs and marks (U+0816–082D) | letters, punctuation |
| `latin_macrons` | macrons and breves (ā, ă → a) | letters |

Why bother: in pointed text every word carries several combining marks. The tokenizer then splits it into many
more pieces, and the same word with and without points becomes two unrelated token sequences. Choose one
convention for the whole project. Consonantal text is the robust default when your sources mix pointed and
unpointed editions. Keep the points if the vocalization is what you study, and check the cost with
`tokenizer-check` ([§4](#4-checking-the-tokenizer)).

---

## 4. Checking the tokenizer

```bash
lfm3-cpt -c my.toml tokenizer-check                         # your train_files
lfm3-cpt tokenizer-check 'corpora/latin/**/*.txt'           # any files
lfm3-cpt tokenizer-check --text "בקדמין ברא יי ית שמיא"     # a quick sample
```

`prepare` also records the same numbers in `metadata.json`, and `train` prints them.

| number | meaning |
|---|---|
| characters per token | **script coverage.** About 3–5 is well covered. Around 1 means single characters; below 1 means the tokenizer falls back to bytes (a Hebrew letter is 2 bytes in UTF-8) |
| tokens per word | also reflects morphology (long agglutinative or affix-rich words); compare it with English rather than reading it alone |

The examples at the end show how a few of your words are split. `�` pieces are partial UTF-8 bytes.

| verdict | what to do |
|---|---|
| good (≥ 3 chars/token) | nothing |
| fair (2–3) | consider `--train-embeddings` |
| poor (< 2) | use `--train-embeddings` with a lower learning rate (`--lr 1e-4`); for very poor coverage, a byte-level model such as ByT5 avoids the problem entirely, at the cost of LFM3's general knowledge |

`--train-embeddings` trains the full input embedding matrix (tied to the output layer), about 130M extra
parameters. It does not add tokens. Vocabulary extension (new tokens for the script) is the next step beyond
it and is not implemented here.

---

## 5. Grammar tests with your own minimal pairs

TurBLiMP is Turkish-only, so the grammar test runs only when `target_language = "tr"` or when you supply your
own pairs:

```toml
grammar_files = ["corpora/aramaic/pairs.tsv"]
```

**TSV:** `grammatical<TAB>ungrammatical<TAB>phenomenon`. Lines starting with `#` are comments. An optional
header line (`sentence_good<TAB>sentence_bad<TAB>phenomenon`) may follow the comments. The phenomenon column
is optional; it defaults to the file name, so one file per phenomenon also works.

**JSONL:** `{"sentence_good": ..., "sentence_bad": ..., "phenomenon": ...}` (`good`/`bad` also accepted).

A pair passes when the model gives the grammatical sentence the higher total log-probability. Results are
broken down by phenomenon, and `results.json` counts the pairs training fixed and broke, with examples.

**Writing good pairs:**
- Change exactly one thing (one ending, one word), so the only difference between the sentences is the feature
  you test. Different lengths are fine, but keep them close.
- Use vocabulary the corpus actually contains, in the corpus's orthography. Normalization is applied to the
  pairs too.
- Cover each phenomenon with at least 10–20 pairs. Below that, one pair changes the score by 5–10 points.
- Avoid pairs where the "ungrammatical" form is attested as a variant reading: the model may have seen it.

`examples/latin-pairs.tsv` has eight Latin pairs (agreement, case) as a model, and
`examples/aramaic-pairs.template.tsv` is an empty template.

**Ready-made pairs for 101 languages.** MultiBLiMP 1.0 (Jumelet et al., 2025; CC-BY-4.0) derives subject-verb
agreement pairs (number, person, gender) from Universal Dependencies:

```bash
lfm3-cpt multiblimp heb -o corpora/he/multiblimp-heb.jsonl     # ISO 639-3 code; check the dataset for its list
```

It covers one phenomenon family only, so it is a narrower test than TurBLiMP, but it is the same test across
languages.

---

## 6. Ranking restorations: lacunae, conjectures, variants

`rank` fills a gap with each candidate and ranks the resulting texts by their probability under the model, using
the same scoring as the grammar test.

```bash
lfm3-cpt -c my.toml rank --text "Arma virumque ___, Troiae qui primus ab oris" --candidate cano --candidate canto
lfm3-cpt -c my.toml rank --candidates-file candidates.txt --text "..."
lfm3-cpt -c my.toml rank --input gaps.jsonl --output ranked.jsonl          # batch
```

`gaps.jsonl` has one gap per line: `{"id": "...", "text": "... ___ ...", "candidates": ["...", "..."]}`. The
output keeps every candidate with its total log-probability, its per-token mean, its token count and its
distance from the best.

- **Model:** by default the run's merged CPT model (else its adapter, else the base model). Pass a path to use
  another model, e.g. to compare the base and the CPT model on the same gaps.
- **`--by total`** (default) ranks by the probability of the whole text. That is the right score when the
  candidates compete for the same gap, but it favours shorter fillers, because every extra token lowers the
  probability. **`--by mean`** ranks per token. When candidates differ a lot in length, look at both.
- **Context** matters more than anything: give as much surrounding text as you have, on both sides of the gap.
  A causal model reads left to right, but the score of the whole text still reflects how well the words after
  the gap fit.
- Treat the ranking as evidence to weigh, not a verdict. A CPT model trained on a few million tokens knows the
  corpus's formulae well and rarer constructions poorly.

To check that the method works on your material, cut a few gaps out of the *held-out* files, where you know the
right answer, and see where the true reading ranks.

---

## 7. Stage 2 for other languages

`sft` works with any chat or instruction dataset ([README § Stage 2](../README.md#stage-2-optional-turkish-instruction-tuning-sft)).
For other languages:

- **A Hub dataset with many languages:** filter it, e.g. `sft --filter language_code=heb` for Aya, if the
  language is there. `--filter ""` disables the filter.
- **Your own file:** `-s sft_dataset=path/to/pairs.jsonl` (also `.json`, `.csv`, `.tsv`, `.parquet`, or a glob).
  The usual column layouts are detected (`instruction`/`input`/`output`, `prompt`/`response`, `messages`,
  ShareGPT `conversations`, Aya `inputs`/`targets`), or name the columns with `sft_user_column` and
  `sft_assistant_column`. A local file has no test split: give one with `sft_eval_file`, or 5% is held out.
- **Translation instead of chat:** for a language without conversation data, verse- or sentence-aligned pairs
  make a good stage 2 (`examples/sft-translation.example.jsonl`). The model learns to translate into the
  language CPT taught it. chrF against the reference is then a real MT metric. Read `generations.jsonl`
  alongside it.
- **Answer-language check:** `target_language` drives the "answers in …" metric. Turkish and Latin are detected
  by frequent words; Hebrew/Aramaic (`he`, `arc`), Arabic, Syriac and Samaritan by script. Script detection
  cannot tell Hebrew from Aramaic, so for Hebrew → Aramaic translation that rate only catches drift into
  English. Set `target_language = ""` to drop the check.

---

## 8. Very small corpora

Targumic Aramaic, Samaritan Aramaic or a single author's Latin may total a few million tokens or less. CPT still
helps, but the risks change:

- **Mix in related languages.** The model shares representations across related languages, so Biblical and
  Jewish Babylonian Aramaic, Syriac (in the same script) and Biblical Hebrew all help the target. Weight the
  target up with `@N` (e.g. target `@4`, close relatives `@2`, more distant ones `@1`). Measure the effect with
  a target-only run ([QUICKSTART § 5.4](../QUICKSTART.md#54-prepare-and-train)).
- **Expect several epochs, and watch for memorization.** Keep `eval_files` to whole held-out books, evaluate
  often (`eval_steps`), and stop when held-out perplexity turns up while training loss keeps falling.
  An adapter checkpoint is saved every `save_steps`; raise `save_total_limit` (default 2) to keep more, and
  merge the best one with `lfm3-cpt merge <run>/checkpoints/checkpoint-<step>`.
- **A lower learning rate** (1e-4) when training the embeddings, and LoRA's default rank 64 is plenty.
- **Normalize consistently** ([§3](#3-normalization)). With little data, the same word in two spellings is
  costly.
- **Use the result as a language model.** Scoring and restoration ([§6](#6-ranking-restorations-lacunae-conjectures-variants)),
  plausibility of readings, OCR/HTR post-correction (rank the recognizer's alternatives), and as the starting
  point for a translation SFT ([§7](#7-stage-2-for-other-languages)). Do not expect a fluent chat model in a
  language with no conversation data.

---

## 9. Settings reference

All settings can be set in a TOML file (`-c`), with `-s key=value`, and many with command options.
`lfm3-cpt show-config` prints every one.

| key | default | meaning |
|---|---|---|
| `target_language` | `"tr"` | `tr`, `la`, `he`, `arc`, `ar`, `syc`, `sam`, or `""`; picks TurBLiMP (tr only) and the SFT answer check |
| `unicode_form` | `"NFC"` | Unicode normalization form, or `none` |
| `strip_marks` | `[]` | mark sets to remove ([§3](#3-normalization)) |
| `train_files` | `[]` | glob patterns, `pattern@N` to repeat; setting it switches stage 1 to local files |
| `eval_files` | `[]` | held-out evaluation files (else a holdout from `train_files`) |
| `ood_files` | `[]` | out-of-distribution evaluation files (local mode only) |
| `file_format` | `"auto"` | `auto` (by extension), `text`, `jsonl` |
| `text_field` | `"text"` | JSONL field holding the document |
| `doc_split` | `"file"` | `file`, `paragraph`, `line` (text files) |
| `file_min_chars` | `0` | drop shorter documents |
| `eval_holdout` | `"files"` | `files` (whole files) or `documents` |
| `eval_fraction` | `0.05` | share held out when there are no `eval_files` |
| `grammar_files` | `[]` | minimal-pair TSV/JSONL files ([§5](#5-grammar-tests-with-your-own-minimal-pairs)) |
| `wiki_config` | `"20231101.tr"` | Wikipedia subset (Hub mode) |
| `ood_config` | `"tur_Latn"` | FineWeb-2 subset (Hub mode); `ood_pages = 0` disables it |
| `train_embeddings` | `false` | fully train the input embeddings |
| `sft_dataset` | `"CohereLabs/aya_dataset"` | Hub id or local file/glob |
| `sft_eval_file` | `""` | separate SFT evaluation file |
| `sft_filter_column` / `sft_filter_value` | `language_code` / `tur` | row filter; skipped for local files without that column |

With TOML, sections such as `[data]`, `[lora]`, `[training]` and `[sft]` are only for readability: every key is
global. See `examples/*.toml`.
