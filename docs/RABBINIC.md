# Tannaitic (Mishnaic) Hebrew with LFM3: a tested recipe

> **Status: tested** on an RTX 3090 (November 2026). Every command below was run as shown, and every number is
> measured. It builds on [HEBREW.md](HEBREW.md): its stage-1 Modern Hebrew model (`runs/lfm3-1.7b-cpt-he/merged`)
> is the starting point.

**What came out of it:** a model that reads and writes Tannaitic Hebrew, and two tools to measure and use it for
filling lacunae.

| | LFM3 (original) | Modern Hebrew model | **Tannaitic model** |
|---|---|---|---|
| perplexity on unseen tractates | 34.9 | 10.2 | **5.76** |
| minimal pairs (word order, word choice) | 73.3% | 83.5% | **94.8%** |
| gap filling: right word among 8 plausible (chance 12.6%) | | 42.5% | **59.8%** |
| … right word in the top 3 | | 72.5% | **89.0%** |
| … with the frequency correction (`--pmi 0.5`, § 6) | | | **63.2% / 90.2%** |
| Modern Hebrew perplexity (Wikipedia, unseen) | 21.0 | 5.30 | 5.61 |
| English perplexity (Wikipedia, unseen) | 15.8 | 13.0 | 13.1¹ |

¹ measured after one pass (§ 5).

Training took 77 minutes on one RTX 3090. The Tannaitic part of Sefaria is small (4.3M characters after cleaning,
about 3M tokens); the model learned how the Mishnah *speaks* far better than what it *says* (§ 8).

- [0. What you need](#0-what-you-need)
- [1. Choose the texts and editions](#1-choose-the-texts-and-editions)
- [2. Export and clean](#2-export-and-clean)
- [3. Tokenizer, replay, prepare](#3-tokenizer-replay-prepare)
- [4. Which starting model? Measure, then train both](#4-which-starting-model-measure-then-train-both)
- [5. The final model](#5-the-final-model)
- [6. Gap filling: the real test](#6-gap-filling-the-real-test)
- [7. Restoring a lacuna](#7-restoring-a-lacuna)
- [8. What it can and cannot do](#8-what-it-can-and-cannot-do)
- [9. Lessons learned](#9-lessons-learned)
- [10. Not done yet](#10-not-done-yet)

---

## 0. What you need

- The merged Modern Hebrew model from HEBREW.md: `runs/lfm3-1.7b-cpt-he/merged`, and its held-out sets
  `corpora/he/wiki.eval.jsonl` (Modern Hebrew) and `corpora/en-check/wiki.eval.jsonl` (English).
- Sefaria's library export, [Sefaria/hebrew_library](https://huggingface.co/datasets/Sefaria/hebrew_library):
  one 7.7 GB JSONL file with one segment (a mishnah, a paragraph) per line, in every edition Sefaria has.
  `lfm3-cpt sefaria` downloads it once into the Hugging Face cache (or give a copy with `--file`).
- `corpora/` must be in `.gitignore` (it is in the project's own): exported corpora must never be committed.

**Licences.** The dataset card says GPL-3.0, but every *edition* has its own licence on Sefaria (public domain,
CC-BY, CC-BY-NC, …). For a private experiment that doesn't matter; before publishing a model, check the editions
listed in `corpora/rabbinic/tannaitic.books.tsv`.

## 1. Choose the texts and editions

```bash
lfm3-cpt sefaria --preset tannaitic --prefer-version Kaufmann --prefer-version 'codex Vienna' --stats
```

The `tannaitic` preset selects the Mishnah (with Pirkei Avot), the Tosefta, Mekhilta (both), Sifra and Sifrei
(Bamidbar, Devarim, Zuta). It leaves out commentaries (`… on …`), *Mishnah Berurah* (20th-century halakhah, despite
its name), introductions and *Tosefta Kifshutah*. The table shows one row per work and the edition it will use.

What we chose, and why:

- **Mishnah: the Kaufmann manuscript** (Dan Be'eri's edition) for all 63 tractates: Mishnaic Hebrew as it was
  written, with its plene spellings, rather than a printed text. Sizes are compared *without vowel points*:
  otherwise a vocalized edition always looks largest, and an earlier version of the tool mixed Kaufmann with
  printed editions tractate by tractate.
- **Tosefta: Lieberman's edition of the Vienna codex** where it exists (Zeraim to Nezikin), Machon Mamre for the
  rest. Sefaria has most tractates twice (*Tosefta Berakhot* and *Tosefta Berakhot (Lieberman)*); only one of
  each pair is kept (33 books left out). `--keep-all-editions` would keep both, near-duplicates included.
- **Sifra** (Venice 1545) is stored by Sefaria as one "book" per chapter; the tool groups it by parashah.

Result: 145 works, about 5.0M characters before cleaning.

## 2. Export and clean

```bash
lfm3-cpt sefaria --preset tannaitic --prefer-version Kaufmann --prefer-version 'codex Vienna' \
    --strip-punctuation --drop-editorial -o corpora/rabbinic/tannaitic --pairs corpora/rabbinic/pairs.jsonl
```

Our export: **2,240 chapters, 4.28M characters** for training; **103 chapters, 0.25M** held out (Mishnah and Tosefta
Bekhorot, Eduyot, Ketubot, Parah, Zevachim); 1,000 minimal pairs from the held-out text. The cleaning counts:

| cleaning step | count |
|---|---|
| bracketed editorial passages removed (`(תהלי׳ ע״ד י״ג)`, `[…]`, `⟨…⟩`, nested ones too) | 16,317 |
| quoted passages unquoted (the verse stays, its quotation marks go) | 10,713 |
| `'` → geresh ׳ (`ר' יהודה` → `ר׳ יהודה`, `תהלי'` → `תהלי׳`) | 19,572 |
| `"` → gershayim ״ (`רז"ל` → `רז״ל`) | 9,076 |
| modern punctuation removed (`. , ; ! ?`, dashes; the colon closing a mishnah stays) | 146,578 |
| maqaf → space (`כל־אדם` → `כל אדם`) | 254 |
| numbering letters removed (`ח החשוד…`: mishnah 8) | several thousand |
| closing quotes after a final letter, stray brackets, control characters, markup codes, Latin segments | ~400 |

Rules worth knowing:

- **Native marks.** Medieval texts have no quotation marks, but abbreviations and truncations carry a geresh or
  gershayim. A quotation is an opening quote at the start of a word and a closing one at the end of a word: both
  go. A `'` after a final letter (ך ם ן ף ץ) also closes a quotation, since a truncation never ends in one. What
  remains after a letter is an abbreviation mark.
- **Nothing is glued.** Removed tags, brackets and punctuation leave a space (`<b>והקריבו</b>אל` → `והקריבו אל`).
- **Numbering.** Hebrew has no one-letter word without a geresh, so a lone letter left in the text is a number.
- **Duplicates.** A segment whose text (without vowels and punctuation) was already seen is dropped.
- **Held out by tractate, together with its parallels.** *Mishnah Berakhot* and *Tosefta Berakhot* always go to
  the same side, because the Tosefta often repeats the Mishnah word for word.
- Vowel points stay in the files and are removed at training time (`strip_marks` in the config).

**Audit before training.** This is where most of our time went, and where it paid off:

```bash
python3 - <<'E'
import json, random, re
from collections import Counter
docs = [json.loads(l)["text"] for l in open("corpora/rabbinic/tannaitic.train.jsonl")]
text = "\n".join(docs)
print(f"{len(docs):,} chapters, {len(text)/1e6:.2f}M characters")
for name, pat in [("ASCII quotes", r"[\"']"), ("Latin letters", r"[A-Za-z]"), ("digits", r"\d"),
                  ("brackets", r"[\[\](){}<>]"), ("maqaf", "־"), ("lone letters", r"(?<!\S)[א-ת](?=\s)")]:
    print(f"{name:13} {len(re.findall(pat, text)):9,}")
bare = Counter(re.sub(r"[֑-ׇ]", "", w) for w in re.findall(r"[א-ת֑-ׇ]+׳", text))
print("geresh, most common:", " ".join(f"{w}({c})" for w, c in bare.most_common(15)))
print("geresh, 4+ letters:", " ".join(w for w, _ in bare.most_common() if len(w) >= 5)[:600])
random.seed(1)
for d in random.sample(docs, 2):
    print("\n---\n" + d[:400])
E
```

All counts should be (near) zero. The long geresh words are mostly real truncations (`עזרי׳`, `ישעי׳`, `שנאמ׳`,
`והכת׳`); what is left are verse endings whose opening quote was missing in the edition (`תיראו׳`, `יבחר׳`). We
had about 350 of these in 4.3M characters and stopped there.

## 3. Tokenizer, replay, prepare

```bash
lfm3-cpt -c examples/rabbinic-tannaitic.toml tokenizer-check corpora/rabbinic/tannaitic.train.jsonl
```

1.46 characters per token, the same as Modern Hebrew (1.54), on which stage 1 trained well; "poor" is relative to
English. One finding: the tokenizer has **no token for the geresh ׳ or gershayim ״** (`וגו׳ → ['וג', 'ו', '�',
'�']`): each is spelled as its two UTF-8 bytes. That costs about 1% more tokens and the model learns the byte pair
like any other sequence, so we kept the native marks. (`--keep-quotes` would export ASCII `'` and `"` instead.)

Replay keeps Modern Hebrew and English from being pushed out: about half of the Tannaitic size (T = 4.28M) in
Modern Hebrew, a third in English:

```bash
lfm3-cpt export wikimedia/wikipedia --subset 20231101.he --title-field title \
    -o corpora/rabbinic/replay-he --max-chars 2.1M --eval-docs 0 --seed 7
lfm3-cpt export HuggingFaceFW/fineweb-edu --subset sample-10BT \
    -o corpora/rabbinic/replay-en --max-chars 1.4M --eval-docs 0 --seed 7
lfm3-cpt -p rtx3090 -c examples/rabbinic-tannaitic.toml prepare
```

`examples/rabbinic-tannaitic.toml` reads exactly these files, with the Tannaitic text three times (`@3`):
16.3M characters, **8.5M tokens** per pass.

## 4. Which starting model? Measure, then train both

Mishnaic Hebrew is not an early form of Modern Hebrew, and a model trained on Modern Hebrew brings modern usage
with it. On the other hand it already reads the script, much of the lexicon and the morphology. So we measured
both starting points before training, then trained both on the same data:

```bash
for m in LiquidAI/LFM3-1.7B runs/lfm3-1.7b-cpt-he/merged; do
  CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml evaluate $m
done

# A: from the Modern Hebrew model, embeddings frozen (they were trained in stage 1)
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/rabbinic-tannaitic.toml \
    -s num_train_epochs=1 -s output_dir=runs/lfm3-1.7b-cpt-rabbinic train --merge
# B: from the original LFM3, which has to train its Hebrew embeddings
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/rabbinic-tannaitic.toml \
    -s num_train_epochs=1 -s model_id=LiquidAI/LFM3-1.7B -s train_embeddings=true \
    -s output_dir=runs/lfm3-1.7b-cpt-rabbinic-base train --merge
```

| one pass (46 min each)¹ | before → after: A, from Modern Hebrew | before → after: B, from LFM3 |
|---|---|---|
| perplexity on unseen tractates | 10.2 → **6.13** | 34.9 → 6.56 |
| minimal pairs | 84.5% → **94.0%** | 73.3% → 92.1% |
| Modern Hebrew perplexity | 5.30 → 5.54 | 21.0 → 8.36 |

¹ on the data before the last cleaning fix (mishnah numbers); both runs on the same data.

**A wins on every measure**, and the texts agree. A's continuations stayed Mishnaic in form; one of B's two drifted
into Modern Hebrew web prose (`הערך האמיתי … היה 1/3, ולא 1/2 כפי שצוין`), the Hebrew the original model saw in
pretraining. Before any Mishnaic training, the Modern Hebrew stage had already cut the Mishnah's perplexity from 37
to 10: it works as a better reader of the script and the language, and the Tannaitic stage then sets the register.

## 5. The final model

The held-out perplexity of A was still at its best on the last step, so it hadn't finished learning. 1.7 passes
over the same mix (each Tannaitic text seen about 5 times; the replay keeps its share) is now the config's default:

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/rabbinic-tannaitic.toml train --merge
```

| from the Modern Hebrew model (10.15, 83.5% before) | 1 pass (46 min) | **1.7 passes** (77 min, 14.4M tokens) |
|---|---|---|
| perplexity on unseen tractates | 6.15 | **5.76** |
| minimal pairs | 94.4% | **94.8%** |
| gap filling, top-1 / top-3 (§ 6, raw score) | 55.0% / 85.2% | **59.8% / 89.0%** |
| Modern Hebrew perplexity | 5.54 | 5.61 |

English stayed where it was: 13.12 after one pass, against 12.98 for the Modern Hebrew model. To check it on
the final model:

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml \
    -s ood_files=corpora/en-check/wiki.eval.jsonl -s data_dir=data/rabbinic-encheck \
    evaluate runs/lfm3-1.7b-cpt-rabbinic-e17/merged --no-grammar
```

Reading its text (greedy decoding loops on any base model: sample, with a repetition penalty):

```bash
lfm3-cpt generate runs/lfm3-1.7b-cpt-rabbinic-e17/merged --temperature 0.7 --repetition-penalty 1.15 \
    --max-new-tokens 150 --prompt "כל הבכורות" --prompt "האומר לחבירו" --prompt "האשה נקנית"
```

The form is right: `רבי אליעזר אומ׳`, `אמר ר׳ שמעון: מודים`, `והלא דברים קל וחומר מה אם… אינו דין ש…`,
`הרי הוא אומר`, Kaufmann's spellings (`חיב`), no Modern Hebrew. The content is halakhic word salad: firstborns,
betrothal and slaughter in one breath.

## 6. Gap filling: the real test

The minimal pairs are easy (their wrong option is a random word) and stop telling models apart at ~95%. A lacuna
means choosing among *plausible* words. `cloze` builds such items once and scores any number of models on the same
file. We used model B as the *proposer* of hard wrong options: it knows Mishnaic Hebrew but is not one of the
models compared (a model is at a disadvantage on options chosen against itself).

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml cloze build \
    --vocab-file corpora/rabbinic/tannaitic.train.jsonl \
    --proposer runs/lfm3-1.7b-cpt-rabbinic-base/merged -o corpora/rabbinic/cloze.jsonl
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml cloze score corpora/rabbinic/cloze.jsonl \
    -m runs/lfm3-1.7b-cpt-he/merged -m runs/lfm3-1.7b-cpt-rabbinic-e17/merged --pmi 0.5
```

400 items from the held-out tractates, 8 candidates each: the right word, 4 of the same form class (first letter,
ending, length, frequency: `מבושלין`, `טהורין` against `עבודין`) and 3 the proposer predicts in that slot.

**Raw probability prefers common words** (`על`, `אבל`, `אדם` over `בבית`, `תמימים`, `כנסת`). `--pmi α` subtracts α
times the word's probability on its own; one scoring pass gives a column per α:

| Tannaitic model | α = 0 | 0.25 | **0.5** | 0.75 | 1 |
|---|---|---|---|---|---|
| top-1 | 60.5% | 63.5% | **63.2%** | 62.0% | 58.8% |
| top-3 | 89.0% | 89.8% | **90.2%** | 89.2% | 87.0% |
| words never seen in training (25) | 32% | 36% | **48%** | 48% | 48% |
| rare words, 1-3 times (38) | 50.0% | 55.3% | **55.3%** | 52.6% | 50.0% |
| beats the proposed options | 79.1% | 81.5% | **82.8%** | 83.3% | 83.5% |

**α = 0.5** is the setting to use. The Mishnaic training itself (Modern Hebrew model → Tannaitic model, raw score)
raised top-1 from 42.5% to 59.8% and rare words from 29% to 50%: it improved word choice, not only fluency.

What the remaining misses are:
- **The manuscript's own readings:** `ומלח קודמת לעצים` (Kaufmann) where the model chose `קודם`, `אין מקדשין` /
  `מקדש`: the reading of the printed tradition, also in the training data. Arguably not errors. Prefix variants
  (`הילך` / `והילך`) are rare: ignoring a ו/ה prefix adds only 0.4%.
- **Agreement and structure:** `אינן` (two firstborns) → `אינו`; `עשאה למעלן` after `למטן` → `למטן`.
- **Knowledge:** `אתו ואת בנו`, `כלי לבן` in Parah, `העיד` (the hallmark of Eduyot), `נפרעין` for collecting
  a ketubah.

Real variant readings can be scored the same way: write items by hand, one per line,
`{"text": "… ___ …", "answer": "…", "candidates": ["…", "…"]}`, and run `cloze score` on them.

## 7. Restoring a lacuna

`restore` is the tool. It collects fillers from the model's own continuations (also several words), from every
word of the corpus that fits, and from your conjectures; keeps those that fit what is visible; and ranks them with
the whole text, including the words after the gap:

```bash
# only the size of the gap is known (5-7 letters); the right word is נפרעין
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml restore runs/lfm3-1.7b-cpt-rabbinic-e17/merged \
    --text "בין ששעבד האב בחייו בין ששעבדו יורשין לאחר מיתת אביהן אין ___ מהן לענין כתובה אבל לפרנסה גובין מהן" \
    --letters 5-7 --pmi 0.5 --candidate נפרעין

# a few letters are legible: נ, two illegible, then עין
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml restore runs/lfm3-1.7b-cpt-rabbinic-e17/merged \
    --text "אין ___ מהן לענין כתובה אבל לפרנסה גובין מהן" --pattern "נ??עין"
```

| what is known | candidates that fit | rank of `נפרעין` |
|---|---|---|
| 5-7 letters, raw score | 43,068 | 54 |
| 5-7 letters, `--pmi 0.5` | 43,068 | **28** (top 0.07%) |
| pattern `נ??עין`, one clause of context | 11 | **2** (0.04 behind `נמנעין`) |

Above it, with only the letter count: `נלקחין`, `לקוחין`, `גובין`, words for *taking* and *collecting*. The model
has the meaning; what decides between near-synonyms is idiom, and that is what legible letters supply.

- `--letters 6` or `5-7` counts letters only (no spaces, vowel points, geresh); `--pattern` uses `?` for one
  letter, `*` for any run, a space between words; `--words 2` for longer gaps; `--regex` for anything else.
- `--pmi` only reorders the 300 most plausible fillers (`--rerank`): over a whole corpus it would otherwise lift
  words that are merely rare. The `raw #` column shows how far each one moved.
- Your `--candidate` words are always shown with their rank, also below the top list: that is how to test the
  tool on lacunae whose answer you know.
- Give all the context you have: the words after the gap count as much as those before.

## 8. What it can and cannot do

- **Can:** read and continue Tannaitic text in the right form, prefer the right word among plausible ones
  (63% / 90% top-3 among 8), shortlist fillers for a gap, and narrow them down sharply with legible letters.
- **Cannot:** follow a halakhic argument. Its continuations mix topics freely, and it will **invent sources**
  (tractate, chapter, the name of a Tanna) with full confidence. Treat its suggestions as candidates to check
  against parallels, never as readings.
- **Words it never saw** (48% top-1 at best) are its weakest point; more passes over the same text can't fix that,
  only more text can.
- The minimal pairs are coarse (a swapped word is *usually* wrong; Mishnaic word order is free in places). Compare
  models with them; don't read the number as a grammar score.

## 9. Lessons learned

1. **Look at the data before training on it.** Every cleaning rule above came from reading samples: editions mixed
   tractate by tractate (vowel points inflating sizes), the Tosefta twice, quotation marks turned into gereshes,
   editors' source references, words glued where markup was removed, maqafs, mishnah numbers. Each was cheap to
   fix before training and invisible in the metrics after it.
2. **The Modern Hebrew stage helps, despite the distance between the languages.** It cut the Mishnah's perplexity
   from 37 to 10 before any Mishnaic training, beat the from-scratch start on every measure, and its continuations
   showed *less* Modern Hebrew than the from-scratch model's.
3. **Measure the task, not a proxy.** Perplexity and minimal pairs looked excellent after one pass; the gap-filling
   benchmark showed what was still missing (55% top-1), that more passes helped (60%), and that the scoring rule
   mattered (63% with PMI, unseen words 32% → 48%).
4. **Small corpora want repetition, carefully.** The Tannaitic text seen ~5 times was still improving on unseen
   tractates; keep the replay's share when adding passes (raise the epochs, not the `@N`).
5. **Open search is much harder than multiple choice.** 8 candidates: right word first 63% of the time. 43,000
   corpus words: in the top 30. Legible letters: top 2.

## 10. Not done yet

- **Real lacunae.** A handful of gaps from manuscripts, with the answer known from a parallel witness, is the test
  that matters now.
- **Responsa:** measure first whether the Tannaitic stage transfers to later rabbinic Hebrew, before training on
  them. They are 111M characters, span the Geonim to the 20th century and are full of Aramaic:

  ```bash
  lfm3-cpt sefaria --preset responsa --stats
  lfm3-cpt sefaria --preset responsa --strip-punctuation --drop-editorial --eval-fraction 0.02 --max-chars 0 \
      -o corpora/rabbinic/responsa
  for m in runs/lfm3-1.7b-cpt-he/merged runs/lfm3-1.7b-cpt-rabbinic-e17/merged; do
    CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml \
        -s ood_files=corpora/rabbinic/responsa.eval.jsonl -s data_dir=data/rabbinic-resp evaluate $m --no-grammar
  done
  ```

  To train on them, export a whole-book sample (`--max-chars 20M`), add it to `train_files`, raise the replay to
  match, and keep the Tannaitic text repeated so it isn't drowned out.
- **Chat (stage 2)** isn't needed for gap filling. For a model to ask about texts, rerun the SFT of HEBREW.md § 4
  on this model; Sefaria's [Rabbinic-Hebrew-English-Pairs](https://huggingface.co/datasets/Sefaria/Rabbinic-Hebrew-English-Pairs)
  would add real translation pairs (check each source's licence).
