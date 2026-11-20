# Teaching LFM3-1.7B Modern Hebrew: the recipe that worked

This is the path that took LFM3-1.7B from shaky Hebrew to a usable Hebrew assistant that still answers English
questions in English. Every command and number below comes from real runs on one RTX 3090 (24 GB). On the
Strix Halo, replace `-p rtx3090` with `-p strix-halo`.

**In short:**
1. **CPT** on about 100M tokens of Hebrew Wikipedia and web text, plus 12% English replay, with the embeddings
   trained. 9 hours.
2. **SFT** on Hebrew Alpaca, *mixed with* the original model's own English answers and a few translation pairs.
   15 minutes.

| | original LFM3 | after the recipe |
|---|---|---|
| Hebrew held-out perplexity | 23.8 | **5.8** |
| English Wikipedia perplexity (never trained on) | 15.8 | **13.0** |
| MultiBLiMP Hebrew agreement pairs | 69.6% | **81.3%** (80.6% after SFT) |
| `מהי בירת צרפת?` ("What is the capital of France?"), scored with `rank` | prefers **Lyon** | prefers **Paris** by 8.4 log-prob |
| answers that finish on their own | 36–50% | **94%** |
| Hebrew questions answered in Hebrew, English in English | partly | yes |

All commands run from the repository root. `CUDA_VISIBLE_DEVICES=1` picks the GPU on a two-GPU machine; drop it
if you have one.

---

## 0. Check the tokenizer first

```bash
lfm3-cpt export wikimedia/wikipedia --subset 20231101.he --title-field title \
    -o corpora/he/wiki --max-chars 60M --eval-docs 100
lfm3-cpt tokenizer-check corpora/he/wiki.train.jsonl
```

LFM3's vocabulary barely covers Hebrew: **1.54 characters per token** against 3.92 for English, and 3.26 tokens
per word (2.9× English). Words are split into one- or two-letter pieces. Nothing is lost (there are no unknown
tokens), but training costs about 3× more per word, and the model has to assemble words from letters. Verdict:
**train the embeddings too** (step 2).

---

## 1. Build the stage-1 corpus

```bash
# Hebrew web pages (FineWeb-2): news, forums, shops, blogs
lfm3-cpt export HuggingFaceFW/fineweb-2 --subset heb_Hebr \
    -o corpora/he/fineweb2 --max-chars 120M --eval-docs 100

# English replay (FineWeb-Edu): keeps the model's English
lfm3-cpt export HuggingFaceFW/fineweb-edu --subset sample-10BT \
    -o corpora/en/fineweb-edu --max-chars 25M --eval-docs 100

# grammar test: 2,330 Hebrew subject-verb agreement minimal pairs
lfm3-cpt multiblimp heb -o corpora/he/multiblimp-heb.jsonl

# an English text the model never trains on, to check forgetting honestly (eval only)
lfm3-cpt export wikimedia/wikipedia --subset 20231101.en --title-field title \
    -o corpora/en-check/wiki --max-docs 0 --eval-docs 100
```

Result: 58,273 documents, 205M characters, **101M tokens**, about 12% of it English. `examples/hebrew-mix.toml`
reads exactly these files.

---

## 2. Stage 1: continued pre-training (about 9 hours)

```bash
lfm3-cpt -p rtx3090 -c examples/hebrew-mix.toml prepare
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/hebrew-mix.toml \
    -s train_embeddings=true train --lr 1e-4 --merge
```

- 6,178 steps, 101M tokens at 3,261 tokens/s: **8 h 55 min**, peak memory 9.0 GB.
- 326M trainable parameters (LoRA r=64 plus the 262M-parameter embedding matrix).
- Training the embeddings is why the learning rate is 1e-4 instead of 2e-4.

| | before | after |
|---|---|---|
| Hebrew held-out perplexity | 23.79 | **5.81** |
| English replay perplexity | 20.19 | 15.61 |
| MultiBLiMP | 69.6% | **81.3%** (413 pairs fixed, 141 broken) |

All five agreement types gained 10–15 points (subject-verb and subject-participle; number, gender, person).

**Check English with text the model never saw.** The English replay text in the table above is in-domain, so it
can't show forgetting. Score both models on the English Wikipedia articles exported in step 1:

```bash
for m in LiquidAI/LFM3-1.7B runs/lfm3-1.7b-cpt-he/merged; do
  CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/hebrew-mix.toml -s ood_files=corpora/en-check/wiki.eval.jsonl \
    -s data_dir=data/he-encheck evaluate $m --no-grammar
done
```

English Wikipedia went from 15.82 to **12.98**: no forgetting. The replay worked, and the post-trained chat
model got better at plain prose.

---

## 3. Stage 2: the SFT data

### 3.1 Hebrew instructions

```bash
lfm3-cpt export ashercn97/hebrew_alpaca --inspect      # instruction / input / output: detected automatically
lfm3-cpt -p rtx3090 -c examples/hebrew-mix.toml prepare --sft
```

8,725 training conversations. This is a machine translation of Alpaca, so expect translationese ("unlikely
friends" becomes *חברים לא סבירים*) and some broken references (code identifiers translated into Hebrew).

### 3.2 The original model's own English answers (`distill`)

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt distill HuggingFaceH4/no_robots \
    -o corpora/sft/en-no-robots.jsonl --max 3000                 # 2,736 kept
CUDA_VISIBLE_DEVICES=1 lfm3-cpt distill openai/gsm8k --subset main --verify-number \
    -o corpora/sft/en-gsm8k.jsonl --max 1500                     # 1,298 kept (final number checked)
```

The original LFM3 answers human-written English prompts (No Robots) and maths word problems (GSM8K). Only answers
that finish, are in English, and (for GSM8K) end on the reference number are kept.

### 3.3 A first mixed SFT, needed to make the English → Hebrew pairs

The original model translates *into* English well but *into* Hebrew poorly, so English → Hebrew pairs need a model
that already speaks Hebrew. Train one first:

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/hebrew-mix.toml \
    -s 'sft_mix=corpora/sft/en-no-robots.jsonl@0.2,corpora/sft/en-gsm8k.jsonl@0.1' \
    sft --from runs/lfm3-1.7b-cpt-he/merged -o runs/he-sft-mix --merge
```

### 3.4 Translation pairs in both directions

```bash
python3 - <<'E'
import json, random, re
random.seed(0)

def sentences(path, limit=40000):
    out = []
    for line in open(path, encoding="utf-8"):
        for s in re.split(r"(?<=[.!?])\s+", json.loads(line)["text"]):
            if 6 <= len(s.split()) <= 25 and "\n" not in s:
                out.append(s.strip())
        if len(out) > limit:
            break
    random.shuffle(out)
    return out[:1500]

with open("corpora/sft/prompts-he2en.txt", "w", encoding="utf-8") as f:
    for s in sentences("corpora/he/wiki.train.jsonl"):
        f.write(f'תרגם לאנגלית: "{s}"\n')
with open("corpora/sft/prompts-en2he.txt", "w", encoding="utf-8") as f:
    for s in sentences("corpora/en/fineweb-edu.train.jsonl"):
        f.write(f'Translate into Hebrew: "{s}"\n')
E

CUDA_VISIBLE_DEVICES=1 lfm3-cpt distill corpora/sft/prompts-he2en.txt \
    -o corpora/sft/he2en.jsonl --language en                     # original model: 1,480 kept
CUDA_VISIBLE_DEVICES=1 lfm3-cpt distill corpora/sft/prompts-en2he.txt --model runs/he-sft-mix/sft/merged \
    -o corpora/sft/en2he.jsonl --language he                     # first mixed model: 1,491 kept
```

### 3.5 Remove "thinking aloud" answers

Even with thinking off, the original model sometimes reasons in its answer ("Wait, let me re-read…"). Drop those
answers before training:

```bash
python3 - <<'E'
import json, re
bad = re.compile(r"Wait,|Let me (think|check|re-?read|see)|Actually,|</think>|Hmm|I'm overthinking")
for name in ["en-no-robots", "en-gsm8k", "he2en", "en2he"]:
    rows = [json.loads(l) for l in open(f"corpora/sft/{name}.jsonl", encoding="utf-8")]
    keep = [r for r in rows if not bad.search(r["messages"][-1]["content"])]
    with open(f"corpora/sft/{name}.clean.jsonl", "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name:14} kept {len(keep)} of {len(rows)}")
E
```

---

## 4. Stage 2: the final SFT (about 15 minutes)

```bash
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -p rtx3090 -c examples/hebrew-mix.toml \
    -s 'sft_mix=corpora/sft/en-no-robots.clean.jsonl@0.12,corpora/sft/en-gsm8k.clean.jsonl@0.05,corpora/sft/he2en.clean.jsonl@0.03,corpora/sft/en2he.clean.jsonl@0.05' \
    sft --from runs/lfm3-1.7b-cpt-he/merged -o runs/he-sft-mix3 --merge
```

- 11,725 training conversations: 8,725 Hebrew Alpaca, 1,407 No Robots, 586 GSM8K, 352 Hebrew → English, 586
  English → Hebrew. English answers make up about **20%** of the total.
- 733 steps in **15 minutes**, peak memory 10.5 GB.
- Answers that finish on their own: 50% → **94%**.
- Forgetting check: Hebrew perplexity +0.25, English replay +1.23, MultiBLiMP −0.8. All small.

Make it the default `chat` model:

```bash
mv runs/lfm3-1.7b-cpt-he/sft runs/lfm3-1.7b-cpt-he/sft-previous
cp -r runs/he-sft-mix3/sft runs/lfm3-1.7b-cpt-he/sft
CUDA_VISIBLE_DEVICES=1 lfm3-cpt -c examples/hebrew-mix.toml chat --temperature 0
```

---

## 5. Test it

| type this | meaning | final model |
|---|---|---|
| `באיזו טמפרטורה רותחים מים?` | boiling point of water? | ✅ 100 °C (212 °F), in Hebrew |
| `לדני יש 5 תפוחים והוא קונה עוד 3. כמה תפוחים יש לו?` | 5 apples + 3? | ✅ *לדני יש 8 תפוחים.* |
| `תן שלוש המלצות לתזונה בריאה.` | three healthy-eating tips | ✅ Hebrew (one slip: *שתינו* for *שתו*) |
| `תרגם לאנגלית: "הספרייה סגורה בימי שבת."` | translate to English | ✅ "The library is closed on Saturdays." |
| `What is the capital of Australia?` | | ✅ Canberra, in English |
| `Translate into Hebrew: "The children are playing in the park."` | | ❌ *בחוף השנהב* ("on the Ivory Coast") |
| `Solve 3x + 4 = 2x + 12` | | ❌ x = 8/3 |

Facts can be checked without chat noise:

```bash
lfm3-cpt rank runs/he-sft-mix3/sft/merged --text "___ היא בירת צרפת." \
    --candidate פריז --candidate ליון --candidate מרסיי
```

To check that each answer is in the language it *should* be in, compared with the reference answer's language:

```bash
python3 - <<'E'
import json
from collections import defaultdict
from lfm3_cpt.chat_eval import answer_language
rows = [json.loads(l) for l in open("runs/he-sft-mix3/sft/generations.jsonl")]
stats = defaultdict(lambda: [0, 0])
for r in rows:
    expected = answer_language(r["reference"], "he")
    if expected in ("he", "en"):
        src = r["source"].split("/")[-1]
        stats[src][0] += r["after_language"] == expected
        stats[src][1] += 1
for src, (ok, n) in sorted(stats.items()):
    print(f"{src:30} {ok}/{n} in the expected language")
E
```

---

## 6. What we learned (and what went wrong on the way)

**Stage 1**
- **A poorly covered script still works.** At 1.5 characters per token, CPT cut Hebrew perplexity by 76%. Train
  the embeddings, at learning rate 1e-4.
- **English replay costs little and protects a lot.** With about 12% English, perplexity on *unseen* English
  Wikipedia improved.
- **Test forgetting on text you didn't train on.** The replay's own held-out text is in-domain, so it can't show
  forgetting.

**Stage 2**
- **Hebrew-only SFT makes the model answer *everything* in Hebrew.** "What is the capital of Australia?" got
  *קייב* ("Kyiv"), then *סידני* ("Sydney"), in Hebrew. The knowledge was still there (the original answered
  Canberra), but answering in Hebrew blocked it.
- **Mixing in the original model's own English answers fixes that.** It also brings back fuller, step-by-step
  answers.
- **Too much English breaks the other direction.** With about a third English answers, plus Hebrew → English
  pairs, *Hebrew* questions started getting English answers. Keep English answers around 20%, and Hebrew →
  English pairs small (3%).
- **Distilled answers can carry "thinking aloud".** Filter them out (step 3.5).
- **Gentle settings:** 1 epoch at learning rate 1e-4 (the default) never damaged the stage-1 results.

**Reading the results**
- **MultiBLiMP sentences are Universal Dependencies text,** with prefixes split off (*ב ימים*, *ה תמונה*). The
  comparison within each pair is fair, but no one writes like that.
- **The SFT "answers in Hebrew" and "question's language" rows assume an answer in the question's language.**
  Translation examples break that, so with translation data use the per-source check above.
- **Judge by chat, `rank` and `generations.jsonl`, not by chrF.** Translated references are noisy: one translated
  the `<table>` tag into `<טבלה>`.

**Known limits of the result**
- **Maths:** worse than the original model (x = 8/3 instead of 8). That's the 1.7B size and short-answer SFT data.
- **Multi-turn memory:** "My name is Danny… what's my name?" fails. There's no Hebrew multi-turn data in the mix.
- **Invented details** around correct facts. Typical for 1.7B, in any language.
- **Gaps in everyday vocabulary** ("park" → Ivory Coast). More English → Hebrew pairs with everyday words would
  help.

**Total time on the RTX 3090:** about 9 hours for stage 1, about 15–20 minutes for each SFT run, plus the
`distill` runs.
