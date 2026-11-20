# Tannaitic (Mishnaic) Hebrew with LFM3

> **Status: plan, not yet run.** The commands and tooling are tested; the expected numbers are not measured.
> [HEBREW.md](HEBREW.md) is the tested Modern Hebrew recipe; its stage-1 model
> (`runs/lfm3-1.7b-cpt-he/merged`) is one of the two starting points compared here.

Mishnaic Hebrew is its own language stage, not an early form of Modern Hebrew, and a model trained on Modern
Hebrew brings Modern usage with it: vocabulary, idiom, and meanings that words acquired much later. On the
other hand it shares the script and much of the lexicon and morphology, and its embeddings have already been
trained for Hebrew letters, which a corpus of a few million tokens can't do on its own. Which effect wins is an
empirical question, so step 5 trains **from both starting points on the same data** and compares them on held-out
tractates: the original LFM3, and the Modern Hebrew model.

The corpus is Tannaitic text only (Mishnah, Tosefta, the halakhic midrashim), with a little English replay and,
for the Modern Hebrew start, a little Modern Hebrew replay. Responsa are measured first and only added if they
help (step 6).

- [0. What you need](#0-what-you-need)
- [1. Look at the library before exporting](#1-look-at-the-library-before-exporting)
- [2. Export and clean the Tannaitic corpus](#2-export-and-clean-the-tannaitic-corpus)
- [3. Replay slices](#3-replay-slices)
- [4. Before training: how much Mishnaic does each model already know?](#4-before-training-how-much-mishnaic-does-each-model-already-know)
- [5. Train from both starting points](#5-train-from-both-starting-points)
- [6. Check, and decide about the responsa](#6-check-and-decide-about-the-responsa)
- [7. Chat again (stage 2)](#7-chat-again-stage-2)
- [8. Limits](#8-limits)

---

## 0. What you need

- The merged stage-1 model from HEBREW.md: `runs/lfm3-1.7b-cpt-he/merged`.
- `corpora/he/wiki.eval.jsonl` (Modern Hebrew, held out) and `corpora/en-check/wiki.eval.jsonl` (English) from
  HEBREW.md § 1, to see what the new stage costs.
- Sefaria's library export, [Sefaria/hebrew_library](https://huggingface.co/datasets/Sefaria/hebrew_library):
  one 7.7 GB JSONL file, one segment (usually a mishnah or a paragraph) per line, every edition of every text.
  `lfm3-cpt sefaria` downloads it once into the Hugging Face cache (or give a copy with `--file`).

**Licences.** The dataset card says GPL-3.0, but every *edition* has its own licence on Sefaria (public domain,
CC-BY, CC-BY-NC, ...). That doesn't matter for a private experiment. Before publishing a model, check the
editions listed in the `.books.tsv` manifest.

## 1. Look at the library before exporting

```bash
lfm3-cpt sefaria --preset tannaitic --prefer-version Kaufmann --prefer-version 'codex Vienna' --stats
```

The first table lists Sefaria's categories by size. The second lists every work the selection catches and the
edition the export would use for it. The `tannaitic` preset selects titles starting with *Mishnah*, *Pirkei
Avot*, *Tosefta*, *Mekhilta*, *Sifra*, *Sifrei*, *Midrash Tannaim*. It leaves out commentaries (`… on …`),
*Mishnah Berurah* (20th-century halakhah, despite its name), introductions and Lieberman's *Tosefta Kifshutah*.
Sefaria stores Sifra as one "book" per chapter; these are shown (and held out) by parashah.

The Tannaitic part of Sefaria is about **5–6M characters** (without vowel points), roughly 3–4M tokens.

Things to decide:

- **Which Mishnah.** Sefaria has up to six editions per tractate: Dan Be'eri's edition of the **Kaufmann
  manuscript** (vocalized, the Palestinian tradition and its plene spellings), *Torat Emet*, and the Vilna print.
  Without a preference the export takes the largest edition, counted without vowel points so that pointed and
  plain editions compare fairly. For Mishnaic Hebrew as it was written, Kaufmann is the natural choice:
  `--prefer-version Kaufmann`. Check in the table that every tractate then shows Kaufmann, and that its size
  matches the other tractates' (a much smaller Kaufmann text would mean an incomplete edition on Sefaria).
- **Which Tosefta.** Sefaria has most tractates twice: Lieberman's edition of the Vienna codex (*Tosefta
  Berakhot (Lieberman)*, Zeraim to Nezikin) and the Machon Mamre/Vilna text (*Tosefta Berakhot*). Only one
  of each pair is kept (the table says how many were left out); `--prefer-version 'codex Vienna'` makes it
  Lieberman's wherever it exists. `--keep-all-editions` keeps both.
- **Anything unexpected.** Add `--exclude REGEX` (titles and categories) or `--title REGEX` to fix the selection.
  Rerun `--stats` until the list is right.

## 2. Export and clean the Tannaitic corpus

```bash
lfm3-cpt sefaria --preset tannaitic --prefer-version Kaufmann --prefer-version 'codex Vienna' \
    --strip-punctuation --drop-editorial -o corpora/rabbinic/tannaitic --pairs corpora/rabbinic/pairs.jsonl
```

What it does:

- **One edition per work,** as shown by `--stats`. Every choice is listed in `tannaitic.books.tsv`, including the
  books left out as other editions (`dropped`).
- **Cleaning,** with every change counted in the output table:
  - HTML tags, Sefaria footnotes, entities, invisible bidi/control characters, Hebrew presentation forms, legacy
    markup codes (`@11`) and note markers (`*`, `[3]`, `(4)`);
  - **native marks:** quoted passages lose their quotation marks first (an opening quote at the start of a word,
    a closing one at the end of a word; one-letter abbreviations like `ה'` inside the verse are allowed), and so
    does a `'` after a final letter (ך ם ן ף ץ), since a truncation never ends in one. Then `"` inside an
    abbreviation becomes gershayim (`רז"ל` → `רז״ל`) and `'` after a letter becomes geresh (`ר' יהודה` →
    `ר׳ יהודה`, `תהלי'` → `תהלי׳`), including typographic quotes (`’ “ ”`). Any quote mark left is removed;
  - **maqaf → space** (default; `--keep-maqaf` to keep it): `כל־אדם` → `כל אדם`, also a hyphen standing in
    for a maqaf between Hebrew words;
  - `--strip-punctuation`: modern editorial punctuation (`. , ; ! ?`, dashes) is removed too. The colon that
    closes a mishnah stays. Without this flag the model learns a modern edition's punctuation;
  - removed material (tags, brackets, punctuation) leaves a space, so neighbouring words never get glued;
  - **numbering** left in the text is removed: lone letters (Hebrew has no one-letter word without a geresh, so
    `ח החשוד…` is mishnah 8) and 11-29 at the start of a line;
  - `--drop-editorial`: removes everything in brackets (`[...]`, `(...)`, `<...>`, `{...}`, `⟨...⟩`), nested ones
    too, and any stray bracket: editors' additions, variant readings and source references such as
    `(תהלי׳ ע״ד י״ג)`. Medieval manuscripts have no brackets, so for this corpus it is on;
  - segments with more than 2% Latin letters are dropped (stray translations or notes).
- **Documents** are chapters, with segments in reading order. Vowel points and cantillation are kept in the files
  and removed at training time (`strip_marks` in the config), so you can still change your mind.
- **Held out: whole tractates (5%),** and the same tractate in every collection. *Mishnah Berakhot* and *Tosefta
  Berakhot* go to the same side, because the Tosefta often repeats the Mishnah word for word. To choose them
  yourself: `--eval-match 'Berakhot' --eval-match 'Bava Kamma'`.
- **Minimal pairs** from the held-out tractates only: the original sentence against the same sentence with two
  neighbouring words swapped (`word_order`), or one word replaced by another word of similar frequency
  (`word_substitution`). These pairs are coarse and test the register, not a rule of grammar. Their value is
  that they come from text the model never trains on.

**Audit the cleaning before training on it:**

```bash
python3 - <<'E'
import json, random, re
docs = [json.loads(l)["text"] for l in open("corpora/rabbinic/tannaitic.train.jsonl")]
text = "\n".join(docs)
print(f"{len(docs):,} chapters, {len(text)/1e6:.1f}M characters")
for name, pat in [("ASCII quotes", r"[\"']"), ("Latin letters", r"[A-Za-z]"), ("digits", r"\d"),
                  ("[brackets]", r"\[[^\]]*\]"), ("(parentheses)", r"\([^)]*\)"), ("geresh ׳", "׳"),
                  ("gershayim ״", "״")]:
    hits = re.findall(pat, text)
    examples = " | ".join(sorted(set(h for h in hits if len(h) > 1))[:6])
    print(f"{name:15} {len(hits):9,}  {examples}")
random.seed(1)
for d in random.sample(docs, 3):
    print("\n---\n" + d[:500])
E
```

The bracket and quote counts should be zero. Then look at the gereshes: truncations (`ר׳`, `אומ׳`, `תהלי׳`) are
right, full words with a geresh are leftover closing quotes:

```bash
python3 - <<'E'
import json, re
from collections import Counter
text = "\n".join(json.loads(l)["text"] for l in open("corpora/rabbinic/tannaitic.train.jsonl"))
bare = Counter(re.sub(r"[\u0591-\u05c7]", "", w) for w in re.findall(r"[\u05d0-\u05ea\u0591-\u05c7]+\u05f3", text))
print("most common:", " ".join(f"{w}({c})" for w, c in bare.most_common(25)))
print("4+ letters:", " ".join(w for w, _ in bare.most_common() if len(w) >= 5)[:1500])
E
```

Then check that the tokenizer handles this register as well as Modern Hebrew (stage 1 measured 1.54 characters
per token):

```bash
lfm3-cpt -c examples/rabbinic-tannaitic.toml tokenizer-check corpora/rabbinic/tannaitic.train.jsonl
```

## 3. Replay slices

The Tannaitic corpus is small: the export table prints its training size (call it *T* characters, about 5M).
Small corpora are repeated (`@3` in the config), and the replay is sized from *T*: about half of *T* in Modern
Hebrew and a third in English. With T ≈ 5M characters:

```bash
lfm3-cpt export wikimedia/wikipedia --subset 20231101.he --title-field title \
    -o corpora/rabbinic/replay-he --max-chars 2.5M --eval-docs 0 --seed 7
lfm3-cpt export HuggingFaceFW/fineweb-edu --subset sample-10BT \
    -o corpora/rabbinic/replay-en --max-chars 1.7M --eval-docs 0 --seed 7
```

Then `examples/rabbinic-tannaitic.toml` reads exactly these files. Prepare the data:

```bash
lfm3-cpt -p rtx3090 -c examples/rabbinic-tannaitic.toml prepare
```

## 4. Before training: how much Mishnaic does each model already know?

Score both starting points on the held-out tractates and the pairs (a few minutes each):

```bash
for m in LiquidAI/LFM3-1.7B runs/lfm3-1.7b-cpt-he/merged; do
  CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml evaluate $m
done
```

This is already informative: it shows how much (or how little) Modern Hebrew training helps with reading the
Mishnah before any Mishnaic training.

## 5. Train from both starting points

Same data, same settings, only the starting model differs (about 1 hour each for T ≈ 5M: 3 × 5M + 4M replay
characters ≈ 13M tokens at ~3,300 tokens/s):

```bash
# A: from the Modern Hebrew model (the config's default)
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/rabbinic-tannaitic.toml train --merge

# B: from the original LFM3; its Hebrew embeddings are untrained, so they are trained here
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/rabbinic-tannaitic.toml \
    -s model_id=LiquidAI/LFM3-1.7B -s train_embeddings=true \
    -s output_dir=runs/lfm3-1.7b-cpt-rabbinic-base train --merge
```

The settings, and why:

- Both continue from a base model, not a chat model. CPT erodes chat skills anyway, and stage 2 takes 15 minutes
  to redo (step 7).
- A keeps the embeddings frozen: they were trained in stage 1 for the same script, and a few million tokens would
  mostly overfit 262M embedding parameters. B has to train them; with this little text that is B's handicap.
- `learning_rate = 1e-4`, one pass over a mix in which the Tannaitic text appears three times.
- The Modern Hebrew replay is in both runs, so that the start is the only difference. For a stricter B (no Modern
  Hebrew at all), remove `replay-he` from `train_files` in a copy of the config.

**How to compare them** (`lfm3-cpt report runs/lfm3-1.7b-cpt-rabbinic runs/lfm3-1.7b-cpt-rabbinic-base`):

- held-out tractates' perplexity and pair accuracy: the main result;
- the same `generate` and `rank` checks as in step 6, looking for Modern Hebrew words and idioms in A's
  continuations. That is the concern a perplexity number can't show.

## 6. Check, and decide about the responsa

Run these checks on both models (A: `runs/lfm3-1.7b-cpt-rabbinic/merged`, B:
`runs/lfm3-1.7b-cpt-rabbinic-base/merged`); the commands show A. The results table answers three questions:

| row | good sign |
|---|---|
| held-out perplexity (unseen tractates) | clearly down from the starting model |
| minimal-pair accuracy (`word_order`, `word_substitution`) | up |
| out-of-distribution perplexity (Modern Hebrew Wikipedia) | for A: about the same (a rise of 10% or more means the replay is too small); for B it only shows how much Modern Hebrew it picked up |

English, as in HEBREW.md:

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml \
    -s ood_files=corpora/en-check/wiki.eval.jsonl -s data_dir=data/rabbinic-encheck \
    evaluate runs/lfm3-1.7b-cpt-rabbinic/merged --no-grammar
```

Look at the text, starting with the opening of a held-out tractate (see `tannaitic.books.tsv` for which ones):

```bash
lfm3-cpt generate runs/lfm3-1.7b-cpt-rabbinic/merged --temperature 0 \
    --prompt "מאימתי קורין את שמע בערבית" --prompt "האומר לחבירו"
```

And gap filling, the way you score lacunae:

```bash
lfm3-cpt rank runs/lfm3-1.7b-cpt-rabbinic/merged \
    --text "משעה שהכהנים נכנסים לאכול בתרומתן עד סוף האשמורה ___ דברי רבי אליעזר" \
    --candidate הראשונה --candidate השנייה --candidate האחרונה
```

**Gap filling (cloze).** The minimal pairs are easy: their wrong option is a random word, which usually breaks
the sentence. Restoring a lacuna means choosing among *plausible* words, which is harder (in a first test the
model preferred `שאינן מבושלין` to the correct `שאינן עבודין`). `cloze` builds such items once from the held-out
tractates and scores any number of models on the same file:

```bash
# once: 400 items; wrong options of the same form class and frequency, plus 3 the Modern Hebrew model predicts
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml cloze build \
    --vocab-file corpora/rabbinic/tannaitic.train.jsonl --proposer runs/lfm3-1.7b-cpt-he/merged \
    -o corpora/rabbinic/cloze.jsonl

# any time: compare models on the same items
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml cloze score corpora/rabbinic/cloze.jsonl \
    -m runs/lfm3-1.7b-cpt-he/merged -m runs/lfm3-1.7b-cpt-rabbinic/merged --output runs/cloze-scores.jsonl
```

- The masked word is never one of the 150 most frequent words, and has at least four words before it.
- Corpus options share the answer's first letter (often a prefix: ו ה ב ל מ ש כ) and ending (ין, ות, …), with a
  similar length and frequency; proposed options are what the proposer model itself expects in the slot.
- The table gives top-1 and top-3 accuracy against chance, top-1 ignoring a ו/ה prefix (variants manuscripts often
  disagree on), the mean rank of the right word, top-1 by how frequent
  the word is in training (rare words are where lacunae hurt), and how often the right word beats each kind of
  wrong option. The proposer is at a disadvantage on its own proposals, so compare the proposed-options row
  between the *other* models, or build a second file with a different proposer.
- Real variant readings can be scored the same way: write items by hand (`{"text": "… ___ …", "answer": …,
  "candidates": [...]}`, one per line) and run `cloze score` on them.

**Restoring a real lacuna.** `cloze` measures; `restore` is the tool. You give the text with the gap and what you
know about it, and it proposes fillers from the model's own continuations (also several words), from every corpus
word that fits, and from your own list, keeps those that fit, and ranks them with the whole text, including the
words after the gap:

```bash
# a gap of about six letters
lfm3-cpt -c examples/rabbinic-tannaitic.toml restore runs/lfm3-1.7b-cpt-rabbinic/merged \
    --text "אין ___ מהן לענין כתובה אבל לפרנסה גובין מהן" --letters 5-7

# legible traces: first letter נ, two illegible letters, then עין
lfm3-cpt -c examples/rabbinic-tannaitic.toml restore runs/lfm3-1.7b-cpt-rabbinic/merged \
    --text "אין ___ מהן לענין כתובה אבל לפרנסה גובין מהן" --pattern "נ??עין"

# a two-word gap of 8-10 letters, only the model's proposals
lfm3-cpt -c examples/rabbinic-tannaitic.toml restore runs/lfm3-1.7b-cpt-rabbinic/merged \
    --text "... ___ ..." --words 2 --letters 8-10 --no-vocab
```

- `--letters` counts letters only (no spaces, vowel points, geresh); `--pattern` uses `?` for one letter, `*` for
  any run and a space between words; `--regex` for anything else. Add `--candidate` for your own conjectures.
- The `source` column says where each filler came from, and how often the model proposed it in its samples
  (`model ×12`). A filler found only in the corpus is one the model would not have thought of, but still prefers.
- The score is the probability of the whole text, which favours shorter fillers: compare candidates of similar
  length (the letter constraint does that for you).
- On the gap-filling benchmark this model picks the right word among 8 plausible ones 55% of the time, and has
  it in its top 3 85% of the time: read the top of the list as suggestions to check, not as an answer.

**Responsa.** First as a measurement only: export 2% of the responsa as held-out text, without a training file,
and score both models on it:

```bash
lfm3-cpt sefaria --preset responsa --stats      # check the selection first
lfm3-cpt sefaria --preset responsa --strip-punctuation --drop-editorial --eval-fraction 0.02 --max-chars 0 \
    -o corpora/rabbinic/responsa
for m in runs/lfm3-1.7b-cpt-he/merged runs/lfm3-1.7b-cpt-rabbinic/merged runs/lfm3-1.7b-cpt-rabbinic-base/merged; do
  CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml \
      -s ood_files=corpora/rabbinic/responsa.eval.jsonl -s data_dir=data/rabbinic-resp evaluate $m --no-grammar
done
```

The responsa are much larger than the Tannaitic corpus. They span the Geonim to the 20th century, they are full
of Aramaic, and their Hebrew is medieval and later. If the Tannaitic stage already lowered their perplexity, it
transfers and nothing more is needed. If you do want them in training, add a budgeted, whole-book sample and
keep the Mishnah repeated so it isn't drowned out:

```bash
lfm3-cpt sefaria --preset responsa --strip-punctuation --drop-editorial --eval-fraction 0.02 --max-chars 20M \
    -o corpora/rabbinic/responsa
# then add "corpora/rabbinic/responsa.train.jsonl" to train_files, and raise both replay slices to match
```

## 7. Chat again (stage 2)

Rerun the final SFT from HEBREW.md § 4 with `-s model_id=` set to the better of the two merged models. It uses the same
data and takes about 15 minutes. Sefaria also publishes
[Rabbinic-Hebrew-English-Pairs](https://huggingface.co/datasets/Sefaria/Rabbinic-Hebrew-English-Pairs):
real Rabbinic Hebrew → English translations, which suit stage 2 better than distilled pairs. That is the next
step once stage 1 works; check each source's licence (many translations are CC-BY-NC).

## 8. Limits

- A 1.7B model learns the register well: continuing a mishnah, filling a gap, a sense of what "sounds
  Tannaitic". It will not follow a sugya reliably.
- **It will invent sources** (tractate, chapter, the name of a Tanna) with full confidence. For Rabbinic
  literature a wrong citation is worse than none: treat its output as a reading aid, not an authority.
- The minimal pairs are coarse. A swapped or replaced word is usually wrong, but some swaps are acceptable
  (Mishnaic word order is free in places). Compare models on them; don't read the absolute number as a grammar
  score.
- Cleaning is rule-based. A closing quote whose opening quote is missing, after a word not ending in a final
  letter, still becomes a geresh; the geresh check in step 2 shows how many are left.
