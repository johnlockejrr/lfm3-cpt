"""Rabbinic Hebrew corpora from Sefaria's library export (Hugging Face: Sefaria/hebrew_library).

The export is one JSONL file (about 7.7 GB) with one *segment* per line, usually a paragraph or a mishnah:
`{"text": ..., "metadata": {"ref", "versionTitle", "lang", "docCategory", "dataQuality", "pagerank", "url"}}`.
The same text appears once per edition (`versionTitle`), with or without vowel points, and commentaries sit next
to the texts they comment on. This module therefore:

* selects works by title (regular expressions on the book part of `ref`) and/or by `docCategory`,
* keeps **one edition per book** (preferred by `--prefer-version`, then professional quality, then size),
* removes HTML markup and footnotes, and drops segments already seen in another edition,
* rebuilds documents: the segments of one chapter in reading order (Talmud folios `2a` < `2b` < `10a`),
* holds out **whole tractates** for evaluation, and the same tractate in every collection (Mishnah Berakhot and
  Tosefta Berakhot go to the same side, since the Tosefta often repeats the Mishnah),
* can derive simple minimal pairs from the held-out text, for `grammar_files`.

Titles and categories follow Sefaria's English table of contents ("Mishnah Berakhot", "Tosefta Shabbat
(Lieberman)", "Bartenura on Mishnah Berakhot"). Run with `--stats` first to see what a selection catches.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEFARIA_REPO = "Sefaria/hebrew_library"
SEFARIA_FILE = "data.jsonl"

# Presets: (title patterns, category substrings, exclude patterns). Titles are matched with re.search on the
# book title, categories as case-insensitive substrings of docCategory; a segment is selected if either matches.
PRESETS: dict[str, tuple[list[str], list[str], list[str]]] = {
    # Tannaitic works: Mishnah (incl. Pirkei Avot), Tosefta, the halakhic midrashim.
    "tannaitic": (
        [
            r"^Mishnah\b",
            r"^Pirkei Avot$",
            r"^Tosefta\b",
            r"^Mekhilta\b",
            r"^Sifra\b",
            r"^Sifrei\b",
            r"^Midrash Tannaim\b",
        ],
        [],
        [r"\bon\b", r"^Mishnah Berurah\b", r"Kifshutah|Ki-Fshutah", r"\bCommentary\b", r"\bIntroduction\b"],
    ),
    # Responsa (she'elot u-teshuvot), from the Geonim to the 20th century.
    "responsa": ([], ["Responsa"], [r"\bon\b", r"\bIntroduction\b", r"\bIndex\b"]),
}

# ---------------------------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Segment:
    text: str
    ref: str
    version: str
    category: str
    quality: str
    lang: str


def download_library() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(SEFARIA_REPO, SEFARIA_FILE, repo_type="dataset"))


def iter_segments(path: str | Path, progress: Callable[[int], None] | None = None) -> Iterator[Segment]:
    """Segments of the export; unreadable lines are skipped."""
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if progress and n % 50_000 == 0:
                progress(n)
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = rec.get("metadata") or {}
            text = rec.get("text")
            if isinstance(text, list):
                text = "\n".join(t for t in text if isinstance(t, str))
            ref = meta.get("ref") or rec.get("ref")
            if not isinstance(text, str) or not text.strip() or not isinstance(ref, str):
                continue
            category = meta.get("docCategory") or ""
            if isinstance(category, list):
                category = "/".join(str(c) for c in category)
            yield Segment(
                text=text,
                ref=ref.strip(),
                version=str(meta.get("versionTitle") or ""),
                category=str(category),
                quality=str(meta.get("dataQuality") or ""),
                lang=str(meta.get("lang") or "he"),
            )
        if progress:
            progress(n if "n" in locals() else 0)


# ---------------------------------------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------------------------------------

_ADDR_PART = r"\d+[ab]?"
_REF_RE = re.compile(rf"^(?P<book>.+?),?\s+(?P<addr>{_ADDR_PART}(?::{_ADDR_PART})*)$")


def _part_key(part: str) -> tuple[int, str]:
    return (int(part[:-1]), part[-1]) if part[-1] in "ab" else (int(part), "")


def split_ref(ref: str) -> tuple[str, tuple[str, ...], tuple[tuple[int, str], ...]]:
    """'Mishnah Berakhot 1:2' -> ('Mishnah Berakhot', ('1',), sort key). The chapter is every address level but
    the last; a ref without a numeric address is a book of its own."""
    m = _REF_RE.match(ref)
    if not m:
        return ref, (), ()
    parts = m["addr"].split(":")
    return m["book"], tuple(parts[:-1]), tuple(_part_key(p) for p in parts)


_COLLECTION_RE = re.compile(r"^(Mishnah|Tosefta|Jerusalem Talmud|Babylonian Talmud)\s+")
_EDITION_RE = re.compile(r"\s*\([^)]*\)\s*$")


def tractate_key(book: str) -> str:
    """'Tosefta Berakhot (Lieberman)' and 'Mishnah Berakhot' -> 'berakhot': parallel texts share a key."""
    return _COLLECTION_RE.sub("", _EDITION_RE.sub("", book)).strip().lower()


# ---------------------------------------------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------------------------------------------

_FOOTNOTE_RE = re.compile(
    r"<sup[^>]*>[^<]*</sup>\s*<i[^>]*class=\"footnote\"[^>]*>.*?</i>|<i[^>]*class=\"footnote\"[^>]*>.*?</i>"
    r"|<sup[^>]*class=\"footnote-marker\"[^>]*>.*?</sup>",
    re.S,
)
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACES_RE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u3000]+")
_POINTS_RE = re.compile(r"[\u0591-\u05c7]")  # cantillation + vowel points, for duplicate detection only
_INVISIBLE_RE = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff\u0000-\u0008\u000b-\u001f]")
_PRESENTATION_RE = re.compile(r"[\ufb1d-\ufb4f]+")  # Hebrew presentation forms (e.g. ﬠ, ﭏ) -> plain letters
_MARKUP_CODES_RE = re.compile(r"@\d+|\{[^{}]{0,40}\}|[*†‡§¶]+|\[\d+\]|\(\d+\)")  # legacy codes, note markers

_LETTER = "\u05d0-\u05ea"
_MARKS = "\u0591-\u05c7"
# "Latin" double quotes and look-alikes, and single quotes and look-alikes, as typed in many digital editions.
_DQ = '"\u201c\u201d\u201e\u201f\u2033\u02ba\u00ab\u00bb'
_SQ = "'\u2018\u2019\u201a\u201b\u2032\u02b9\u02bc\u0060\u00b4"
_GERSHAYIM_RE = re.compile(rf"([{_LETTER}][{_MARKS}]*)[{_DQ}](?=[{_LETTER}])")  # רז"ל -> רז״ל
_GERESH_RE = re.compile(rf"([{_LETTER}][{_MARKS}]*)[{_SQ}]")  # ר' -> ר׳ ; ג'ירפה -> ג׳ירפה
_QUOTE_RE = re.compile(rf"[{_DQ}{_SQ}]")  # what is left is quotation: not part of the medieval text
_MODERN_PUNCT_RE = re.compile(r"[.,;!?\u2013\u2014]|(?<=\s)-(?=\s)")
_EDITORIAL_RE = re.compile(r"\[[^\[\]\n]{0,300}\]|\([^()\n]{0,300}\)|<[^<>\n]{0,300}>")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass
class Cleaner:
    """Text cleaning for Rabbinic Hebrew. Counts what it changes (`stats`), so the result can be audited.

    * always: HTML tags and Sefaria footnotes, HTML entities, invisible and bidi control characters, Hebrew
      presentation forms, legacy markup codes (@11, {..}), note markers (*, [3], (4)), repeated spaces;
    * `native_marks` (default): abbreviation and numeral marks become the Hebrew geresh ׳ and gershayim ״
      (ר' -> ר׳, רז"ל -> רז״ל, also typographic quotes); quote marks left after that are quotation, which the
      manuscripts and early prints don't have, and are removed;
    * `strip_punctuation`: also remove modern editorial punctuation (. , ; ! ? dashes); the colon that closes
      a mishnah or a verse, and the maqaf, are kept;
    * `drop_editorial`: remove bracketed editorial matter ([...], (...), <...>): additions, variants, sources;
    * segments in which more than `max_latin` of the letters are Latin are dropped (translations, notes).
    """

    native_marks: bool = True
    strip_punctuation: bool = False
    drop_editorial: bool = False
    max_latin: float = 0.02
    stats: Counter = field(default_factory=Counter)

    def __call__(self, text: str) -> str:
        st = self.stats
        text, n = _FOOTNOTE_RE.subn("", text)
        st["footnotes removed"] += n
        text = _BR_RE.sub("\n", text)
        text = html.unescape(_TAG_RE.sub("", text))
        text, n = _INVISIBLE_RE.subn("", text)
        st["invisible/control characters removed"] += n
        text = _PRESENTATION_RE.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)
        text = unicodedata.normalize("NFC", text)
        text, n = _MARKUP_CODES_RE.subn("", text)
        st["markup codes / note markers removed"] += n
        if self.drop_editorial:
            text, n = _EDITORIAL_RE.subn("", text)
            st["bracketed editorial passages removed"] += n
        if self.native_marks:
            text, n = _GERSHAYIM_RE.subn("\\1\u05f4", text)
            st['" -> gershayim ״'] += n
            text, n = _GERESH_RE.subn("\\1\u05f3", text)
            st["' -> geresh ׳"] += n
            text, n = _QUOTE_RE.subn("", text)
            st["quotation marks removed"] += n
        if self.strip_punctuation:
            text, n = _MODERN_PUNCT_RE.subn("", text)
            st["modern punctuation removed"] += n
        text = re.sub(r"\s+([:\u05c3])", r"\1", text)  # no space before a closing colon / sof pasuq
        lines = (_SPACES_RE.sub(" ", line).strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)
        letters = sum(c.isalpha() for c in text)
        if letters and len(_LATIN_RE.findall(text)) / letters > self.max_latin:
            st["segments dropped (Latin script)"] += 1
            return ""
        return text


def clean_text(text: str) -> str:
    return Cleaner()(text)


def _fingerprint(text: str) -> str:
    bare = _POINTS_RE.sub("", unicodedata.normalize("NFD", text))
    return hashlib.sha1(re.sub(r"\W+", "", bare).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Selector:
    titles: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    professional_only: bool = False
    lang: str = "he"

    def __post_init__(self) -> None:
        self._titles = [re.compile(p) for p in self.titles]
        self._exclude = [re.compile(p) for p in self.exclude]
        self._categories = [c.lower() for c in self.categories]
        if not self._titles and not self._categories:
            raise ValueError("select something: a preset, --title or --category")

    @classmethod
    def from_preset(cls, name: str | None, titles=(), categories=(), exclude=(), **kw: Any) -> Selector:
        p_titles, p_cats, p_excl = PRESETS[name] if name else ([], [], [])
        return cls([*p_titles, *titles], [*p_cats, *categories], [*p_excl, *exclude], **kw)

    def matches(self, book: str, category: str) -> bool:
        cat = category.lower()
        if not (any(p.search(book) for p in self._titles) or any(c in cat for c in self._categories)):
            return False
        return not any(p.search(book) or p.search(category) for p in self._exclude)

    def accepts(self, seg: Segment, book: str) -> bool:
        if self.lang and seg.lang and seg.lang != self.lang:
            return False
        if self.professional_only and seg.quality != "professional":
            return False
        return self.matches(book, seg.category)


# ---------------------------------------------------------------------------------------------------------------
# Statistics (--stats)
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class LibraryStats:
    lines: int = 0
    categories: Counter = field(default_factory=Counter)  # category -> chars
    category_segments: Counter = field(default_factory=Counter)
    books: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))  # book -> version -> chars
    book_category: dict[str, str] = field(default_factory=dict)


def library_stats(segments: Iterable[Segment], selector: Selector | None = None) -> LibraryStats:
    """Characters per category for the whole library; per book and edition for the selected works."""
    st = LibraryStats()
    for seg in segments:
        st.lines += 1
        st.categories[seg.category] += len(seg.text)
        st.category_segments[seg.category] += 1
        if selector is not None:
            book = split_ref(seg.ref)[0]
            if selector.accepts(seg, book):
                st.books[book][seg.version] += len(seg.text)
                st.book_category.setdefault(book, seg.category)
    return st


# ---------------------------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class BookInfo:
    book: str
    category: str
    version: str
    quality: str
    versions: int
    docs: int = 0
    chars: int = 0
    duplicates: int = 0
    split: str = "train"


@dataclass
class SefariaExport:
    train: list[dict[str, str]]
    eval: list[dict[str, str]]
    books: list[BookInfo]
    segments_read: int = 0


def _choose_version(versions: dict[str, dict[str, Any]], prefer: list[re.Pattern]) -> str:
    def score(name: str) -> tuple:
        v = versions[name]
        preferred = next((i for i, p in enumerate(prefer) if p.search(name)), len(prefer))
        return (-preferred, v["quality"] == "professional", v["chars"], name)

    return max(versions, key=score)


def build_corpus(
    segments: Iterable[Segment],
    selector: Selector,
    prefer_versions: Iterable[str] = (),
    min_chars: int = 100,
    max_doc_chars: int = 20_000,
    eval_fraction: float = 0.05,
    eval_match: Iterable[str] = (),
    seed: int = 42,
    cleaner: Cleaner | None = None,
    max_train_chars: int | None = None,
) -> SefariaExport:
    prefer = [re.compile(p) for p in prefer_versions]
    cleaner = cleaner or Cleaner()
    # book -> version -> {"quality", "chars", "segments": [(sort key, chapter, text)]}
    store: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    categories: dict[str, str] = {}
    read = 0
    for seg in segments:
        read += 1
        book, chapter, key = split_ref(seg.ref)
        if not selector.accepts(seg, book):
            continue
        v = store[book].setdefault(seg.version, {"quality": seg.quality, "chars": 0, "segments": []})
        v["chars"] += len(seg.text)
        v["segments"].append((key, chapter, seg.text))  # cleaned below, for the chosen edition only
        categories.setdefault(book, seg.category)

    seen: set[str] = set()
    docs_by_book: dict[str, list[dict[str, str]]] = {}
    books: list[BookInfo] = []
    for book in sorted(store):
        versions = store[book]
        name = _choose_version(versions, prefer)
        chosen = versions[name]
        info = BookInfo(book, categories[book], name, chosen["quality"], len(versions))
        chapters: dict[tuple[str, ...], list[tuple]] = defaultdict(list)
        for key, chapter, raw in chosen["segments"]:
            text = cleaner(raw)
            if not text:
                continue
            fp = _fingerprint(text)
            if fp in seen:
                info.duplicates += 1
                continue
            seen.add(fp)
            chapters[chapter].append((key, text))
        docs: list[dict[str, str]] = []
        for chapter in sorted(chapters, key=lambda c: tuple(_part_key(p) for p in c)):
            parts = [t for _, t in sorted(chapters[chapter], key=lambda kt: kt[0])]
            label = f"{book} {':'.join(chapter)}".strip()
            for text in _chunk(parts, max_doc_chars):
                if len(text) >= min_chars:
                    docs.append({"text": text, "source": label})
        info.docs, info.chars = len(docs), sum(len(d["text"]) for d in docs)
        if docs:
            docs_by_book[book] = docs
            books.append(info)

    eval_keys = _eval_groups(books, eval_fraction, [re.compile(p) for p in eval_match], seed)
    for info in books:
        if tractate_key(info.book) in eval_keys:
            info.split = "eval"
    if max_train_chars is not None:
        # A budget keeps whole books, picked at random (deterministically), so large collections such as the
        # responsa can be sampled without cutting a responsum in half.
        used = 0
        order = sorted(
            (b for b in books if b.split == "train"),
            key=lambda b: hashlib.sha1(f"{seed}:{b.book}".encode()).hexdigest(),
        )
        for info in order:
            if used >= max_train_chars or (used and used + info.chars > 1.2 * max_train_chars):
                info.split = "unused"
            else:
                used += info.chars
    train = [d for b in books if b.split == "train" for d in docs_by_book[b.book]]
    evaluation = [d for b in books if b.split == "eval" for d in docs_by_book[b.book]]
    return SefariaExport(train, evaluation, books, read)


def _chunk(parts: list[str], max_chars: int) -> Iterator[str]:
    """Join segments into documents of at most ~max_chars, splitting only between segments."""
    buf: list[str] = []
    size = 0
    for p in parts:
        if buf and max_chars and size + len(p) > max_chars:
            yield "\n".join(buf)
            buf, size = [], 0
        buf.append(p)
        size += len(p) + 1
    if buf:
        yield "\n".join(buf)


def _eval_groups(books: list[BookInfo], fraction: float, match: list[re.Pattern], seed: int) -> set[str]:
    """Tractate keys held out for evaluation: those matching --eval-match, else whole groups picked at random
    (deterministically) until `fraction` of the characters is reached. At least one group stays in training."""
    sizes: Counter = Counter()
    for b in books:
        sizes[tractate_key(b.book)] += b.chars
    if len(sizes) < 2:
        return set()
    if match:
        keys = {tractate_key(b.book) for b in books if any(p.search(b.book) for p in match)}
        return keys if len(keys) < len(sizes) else set()
    if fraction <= 0:
        return set()
    total = sum(sizes.values())
    order = sorted(sizes, key=lambda k: hashlib.sha1(f"{seed}:{k}".encode()).hexdigest())
    chosen: set[str] = set()
    held = 0
    for k in order:
        if held >= fraction * total or len(chosen) == len(sizes) - 1:
            break
        if held + sizes[k] > 2 * fraction * total and chosen:
            continue  # don't let one huge work swallow the eval set
        chosen.add(k)
        held += sizes[k]
    return chosen


def write_jsonl(rows: Iterable[dict[str, str]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_manifest(books: list[BookInfo], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("book\tsplit\tdocs\tchars\tversion\tquality\tversions\tduplicates_dropped\tcategory\n")
        for b in books:
            fh.write(
                f"{b.book}\t{b.split}\t{b.docs}\t{b.chars}\t{b.version}\t{b.quality}\t{b.versions}\t"
                f"{b.duplicates}\t{b.category}\n"
            )


# ---------------------------------------------------------------------------------------------------------------
# Minimal pairs from held-out text
# ---------------------------------------------------------------------------------------------------------------

_WORD_RE = re.compile(r"^[\u05d0-\u05ea\u0591-\u05c7\u05f3\u05f4]+$")
_SPLIT_RE = re.compile(r"(?<=[.:;?!])\s+")


def _is_word(w: str) -> bool:
    return bool(_WORD_RE.match(w)) and sum("א" <= c <= "ת" for c in w) >= 2


def make_pairs(docs: Iterable[dict[str, str]], n: int = 1000, seed: int = 0, words=(6, 30)) -> list[dict[str, str]]:
    """Coarse minimal pairs from real sentences: the original against (a) two neighbouring words swapped,
    (b) one word replaced by another word of the corpus with a similar frequency. Both corruptions are usually,
    but not always, wrong; the score measures how well the model knows the register, not a grammar rule."""
    rng = random.Random(seed)
    sentences: list[list[str]] = []
    freq: Counter = Counter()
    for d in docs:
        for line in d["text"].splitlines():
            for sent in _SPLIT_RE.split(line):
                toks = sent.split()
                freq.update(t for t in toks if _is_word(t))
                if words[0] <= len(toks) <= words[1]:
                    sentences.append(toks)
                elif len(toks) > words[1]:
                    start = rng.randrange(0, len(toks) - words[1] + 1)
                    sentences.append(toks[start : start + words[1]])
    if not sentences:
        return []
    by_band: dict[int, list[str]] = defaultdict(list)
    for w, c in freq.items():
        by_band[min(c.bit_length(), 12)].append(w)
    for band in by_band.values():
        band.sort()
    rng.shuffle(sentences)
    pairs: list[dict[str, str]] = []
    want = {"word_order": n // 2, "word_substitution": n - n // 2}
    seen: set[tuple[str, str]] = set()
    for toks in sentences:
        if not any(want.values()):
            break
        good = " ".join(toks)
        kind = "word_order" if want["word_order"] >= want["word_substitution"] else "word_substitution"
        if want[kind] == 0:
            kind = "word_order" if kind == "word_substitution" else "word_substitution"
        bad_toks = _swap(toks, rng) if kind == "word_order" else _substitute(toks, freq, by_band, rng)
        if bad_toks is None:
            continue
        bad = " ".join(bad_toks)
        if bad == good or (good, bad) in seen:
            continue
        seen.add((good, bad))
        pairs.append({"sentence_good": good, "sentence_bad": bad, "phenomenon": kind})
        want[kind] -= 1
    return pairs


def _swap(toks: list[str], rng: random.Random) -> list[str] | None:
    spots = [i for i in range(len(toks) - 1) if _is_word(toks[i]) and _is_word(toks[i + 1]) and toks[i] != toks[i + 1]]
    if not spots:
        return None
    i = rng.choice(spots)
    out = list(toks)
    out[i], out[i + 1] = out[i + 1], out[i]
    return out


def _substitute(toks: list[str], freq: Counter, by_band: dict[int, list[str]], rng: random.Random) -> list[str] | None:
    spots = [i for i, t in enumerate(toks) if _is_word(t)]
    rng.shuffle(spots)
    for i in spots[:5]:
        band = by_band.get(min(freq[toks[i]].bit_length(), 12), [])
        choices = [w for w in rng.sample(band, min(len(band), 20)) if w != toks[i] and w not in toks]
        if choices:
            out = list(toks)
            out[i] = choices[0]
            return out
    return None
