# Tannaitic (Mishnaic) Hebrew: a second CPT stage on the Modern Hebrew model

> **Status: plan, not yet run.** The commands and tooling are tested; the expected numbers are not measured.
> [HEBREW.md](HEBREW.md) is the tested recipe this builds on: its stage-1 model
> (`runs/lfm3-1.7b-cpt-he/merged`) is the starting point here.

Rabbinic Hebrew is not a new language for a model that already reads Modern Hebrew: the script, most of the
vocabulary and the trained embeddings carry over. What it has to learn is the register: participle-heavy verbs,
`-ין` plurals, `שֶׁ־` everywhere, different particles, dense abbreviations (`א״ר`, `רז״ל`). So this is a short,
focused stage: Tannaitic text only (Mishnah, Tosefta, the halakhic midrashim), with a little Modern Hebrew and
English replay. Responsa are measured first and only added if they help (step 6).

- [0. What you need](#0-what-you-need)
- [1. Look at the library before exporting](#1-look-at-the-library-before-exporting)
- [2. Export and clean the Tannaitic corpus](#2-export-and-clean-the-tannaitic-corpus)
- [3. Replay slices](#3-replay-slices)
- [4. Before training: how much Mishnaic does the model already know?](#4-before-training-how-much-mishnaic-does-the-model-already-know)
- [5. Train](#5-train)
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
lfm3-cpt sefaria --preset tannaitic --stats
```

The first table lists Sefaria's categories by size. The second lists every book the selection catches, how many
editions it has, and the largest one. The `tannaitic` preset selects titles starting with *Mishnah*, *Pirkei
Avot*, *Tosefta*, *Mekhilta*, *Sifra*, *Sifrei*, *Midrash Tannaim*. It leaves out commentaries (`… on …`),
*Mishnah Berurah* (20th-century halakhah, despite its name), introductions and Lieberman's *Tosefta Kifshutah*.

Check the list. Things to decide:

- **Two Tosefta editions.** Sefaria has the Vilna text (*Tosefta Berakhot*) and Lieberman's critical edition
  (*Tosefta Berakhot (Lieberman)*), which covers only Zeraim to Nezikin. Both are kept unless you exclude one.
  The export drops segments whose text (without vowels and punctuation) is identical, but different spellings
  survive as near-duplicates. To prefer Lieberman where it exists, exclude the Vilna books for those orders by
  name, e.g. `--exclude '^Tosefta (Berakhot|Peah|Demai)$'`.
- **Anything unexpected.** Add `--exclude REGEX` (titles and categories) or `--title REGEX` to fix the selection.
  Rerun `--stats` until the list is right.

## 2. Export and clean the Tannaitic corpus

```bash
lfm3-cpt sefaria --preset tannaitic --strip-punctuation \
    -o corpora/rabbinic/tannaitic --pairs corpora/rabbinic/pairs.jsonl
```

What it does:

- **One edition per book:** the professional one with the most text (or the first matching `--prefer-version`,
  e.g. `--prefer-version 'Torat Emet'`). Pointed and unpointed duplicates of the same text are not both kept.
- **Cleaning,** with every change counted in the output table:
  - HTML tags, Sefaria footnotes, entities, invisible bidi/control characters, Hebrew presentation forms, legacy
    markup codes (`@11`) and note markers (`*`, `[3]`, `(4)`);
  - **native marks:** `"` inside an abbreviation becomes gershayim (`רז"ל` → `רז״ל`) and `'` after a letter
    becomes geresh (`ר' יהודה` → `ר׳ יהודה`), including typographic quotes (`’ “ ”`) that some editions use.
    Quote marks left after that are quotation marks, which medieval texts don't have, and are removed;
  - `--strip-punctuation`: modern editorial punctuation (`. , ; ! ?`, dashes) is removed too. The colon that
    closes a mishnah stays, as does the maqaf. Without this flag the model learns a modern edition's punctuation;
  - `--drop-editorial` (off by default): removes `[...]`, `(...)` and `<...>`: editors' additions, variant
    readings, source references. Look at what your editions put in brackets before deciding (below);
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

If the brackets and parentheses hold editorial matter, rerun the export with `--drop-editorial`. Then check that
the tokenizer handles this register as well as Modern Hebrew (stage 1 measured 1.54 characters per token):

```bash
lfm3-cpt -c examples/rabbinic-tannaitic.toml tokenizer-check corpora/rabbinic/tannaitic.train.jsonl
```

## 3. Replay slices

The Tannaitic corpus is small: the export table prints its size (call it *T* characters). Small corpora are
repeated (`@3` in the config), and the replay is sized from *T*: about half of *T* in Modern Hebrew and a third in
English. For example, with T ≈ 8M characters:

```bash
lfm3-cpt export wikimedia/wikipedia --subset 20231101.he --title-field title \
    -o corpora/rabbinic/replay-he --max-chars 4M --eval-docs 0 --seed 7
lfm3-cpt export HuggingFaceFW/fineweb-edu --subset sample-10BT \
    -o corpora/rabbinic/replay-en --max-chars 3M --eval-docs 0 --seed 7
```

Then `examples/rabbinic-tannaitic.toml` reads exactly these files. Prepare the data:

```bash
lfm3-cpt -p rtx3090 -c examples/rabbinic-tannaitic.toml prepare
```

## 4. Before training: how much Mishnaic does the model already know?

`train` measures its starting model (the Modern Hebrew one) automatically. To see what stage 1 alone did for
Mishnaic Hebrew, score the original model too:

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml evaluate LiquidAI/LFM3-1.7B
```

Compare the two Tannaitic perplexities. If the Modern Hebrew model is already close to its Modern Hebrew
perplexity (5.8), the register is mostly known and a short stage is enough. If it is far, the corpus may be too
small, and that is the case for adding responsa (step 6).

## 5. Train

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/rabbinic-tannaitic.toml train --merge
```

The settings, and why:

- `model_id = runs/lfm3-1.7b-cpt-he/merged`: continue from the stage-1 model, not from the chat model. CPT
  would erode the chat skills anyway, and stage 2 only takes 15 minutes to redo (step 7).
- `train_embeddings = false`: the embeddings were trained in stage 1 for the same script. A few million tokens
  would mostly overfit 262M embedding parameters.
- `learning_rate = 1e-4`, one pass over a mix in which the Tannaitic text appears three times.

At about 3,300 tokens/s on the RTX 3090 and ~1.5 characters per token, T ≈ 8M characters (3 × 8M + 7M replay
≈ 21M tokens) is roughly 1.5–2 hours.

## 6. Check, and decide about the responsa

The results table answers three questions:

| row | good sign |
|---|---|
| held-out perplexity (unseen tractates) | clearly down from the starting model |
| minimal-pair accuracy (`word_order`, `word_substitution`) | up |
| out-of-distribution perplexity (Modern Hebrew Wikipedia) | about the same; a rise of 10% or more means the replay is too small |

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

**Responsa.** First as a measurement only: export 2% of the responsa as held-out text, without a training file,
and score both models on it:

```bash
lfm3-cpt sefaria --preset responsa --stats      # check the selection first
lfm3-cpt sefaria --preset responsa --strip-punctuation --eval-fraction 0.02 --max-chars 0 \
    -o corpora/rabbinic/responsa
for m in runs/lfm3-1.7b-cpt-he/merged runs/lfm3-1.7b-cpt-rabbinic/merged; do
  CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/rabbinic-tannaitic.toml \
      -s ood_files=corpora/rabbinic/responsa.eval.jsonl -s data_dir=data/rabbinic-resp evaluate $m --no-grammar
done
```

The responsa are much larger than the Tannaitic corpus. They span the Geonim to the 20th century, they are full
of Aramaic, and their Hebrew is medieval and later. If the Tannaitic stage already lowered their perplexity, it
transfers and nothing more is needed. If you do want them in training, add a budgeted, whole-book sample and
keep the Mishnah repeated so it isn't drowned out:

```bash
lfm3-cpt sefaria --preset responsa --strip-punctuation --eval-fraction 0.02 --max-chars 20M \
    -o corpora/rabbinic/responsa
# then add "corpora/rabbinic/responsa.train.jsonl" to train_files, and raise both replay slices to match
```

## 7. Chat again (stage 2)

Rerun the final SFT from HEBREW.md § 4 with `-s model_id=runs/lfm3-1.7b-cpt-rabbinic/merged`. It uses the same
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
- Cleaning is rule-based. A closing single quote after a word (`'כך'`) would become a geresh. Hebrew editions
  rarely quote that way, but the audit in step 2 is how you find out.
