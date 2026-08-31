"""
rag_xper.core.retrieval.bm25_retriever

Arabic/English BM25 Okapi retriever with normalization, stemming, ordinal expansion,
and persistent on-disk serialization.
"""
from __future__ import annotations

import math
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rag_xper.core.models import Chunk
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)

# Arabic text normalization tables
_NORM_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})

_TASHKEEL_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")

_ARABIC_STOPWORDS: Set[str] = {
    "من", "الى", "إلى", "عن", "على", "في", "مع", "هذا", "هذه", "ذلك",
    "التي", "الذي", "الذين", "اللذين", "ان", "أن", "كان", "كانت",
    "او", "أو", "ثم", "حيث", "كل", "لم", "لن", "لا", "ما", "هل",
    "ماذا", "كيف", "متى", "اين", "أين", "لماذا", "هو", "هي", "هم",
}

_NUM_TO_WORDS = {
    "1": ["اول", "الاول", "اولي", "الاولي", "واحد", "واحده", "حادي", "حاديه"],
    "2": ["ثاني", "الثاني", "ثانيه", "الثانيه", "اثنين", "اثنان", "اثنتين", "اثنتان"],
    "3": ["ثالث", "الثالث", "ثالثه", "الثالثه", "ثلاثه", "ثلاث"],
    "4": ["رابع", "الرابع", "رابعه", "الرابعه", "اربعه", "اربع"],
    "5": ["خامس", "الخامس", "خامسه", "الخامسه", "خمسه", "خمس"],
    "6": ["سادس", "السادس", "سادسه", "السادسه", "سته", "ست"],
    "7": ["سابع", "السابع", "سابعه", "السابعه", "سبعه", "سبع"],
    "8": ["ثامن", "الثامن", "ثامنه", "الثامنه", "ثمانيه", "ثمان"],
    "9": ["تاسع", "التاسع", "تاسعه", "التاسعه", "تسعه", "تسع"],
    "10": ["عاشر", "العاشر", "عاشره", "العاشره", "عشره", "عشر"],
    "20": ["عشرون", "عشرين", "العشرون", "العشرين"],
    "30": ["ثلاثون", "ثلاثين", "الثلاثون", "الثلاثين"],
    "40": ["اربعون", "اربعين", "الاربعون", "الاربعين"],
    "50": ["خمسون", "خمسين", "الخمسون", "الخمسين"],
    "60": ["ستون", "ستين", "الستون", "الستين"],
    "70": ["سبعون", "سبعين", "السبعون", "السبعين"],
    "80": ["ثمانون", "ثمانين", "الثمانون", "الثمانين"],
    "90": ["تسعون", "تسعين", "التسعون", "التسعين"],
    "100": ["مائه", "ميه", "مائة", "المائه", "المائة"],
}


def normalize_arabic(text: str) -> str:
    """Normalize Arabic orthography, digits, and remove diacritics."""
    t = _TASHKEEL_RE.sub("", text)
    t = t.translate(_NORM_MAP)
    return t.lower()


def stem_arabic_word(word: str) -> str:
    """Lightweight Arabic stemmer removing common prefixes and suffixes."""
    if len(word) <= 3:
        return word

    w = word
    # Prefix stripping
    if w.startswith("ال") and len(w) > 4:
        w = w[2:]
    elif (w.startswith("وال") or w.startswith("فال") or w.startswith("بال") or w.startswith("كال") or w.startswith("لل")) and len(w) > 5:
        w = w[3:] if not w.startswith("لل") else w[2:]
    elif (w.startswith("و") or w.startswith("ف") or w.startswith("ب") or w.startswith("ك") or w.startswith("ل")) and len(w) > 4:
        w = w[1:]

    # Suffix stripping
    for suffix in ["هم", "هن", "كم", "كن", "نا", "ها", "ات", "ون", "ين", "ان", "يه", "يا", "ه", "ي"]:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[:-len(suffix)]
            break

    return w


def tokenize(text: str, expand_numbers: bool = True) -> List[str]:
    """Tokenize and normalize text with Arabic stemming and number expansion."""
    norm_text = normalize_arabic(text)
    raw_tokens = re.findall(r"\b[\w\u0621-\u064A0-9]+\b", norm_text)

    tokens: List[str] = []
    for tok in raw_tokens:
        if tok in _ARABIC_STOPWORDS:
            continue
        tokens.append(tok)

        # Also add un-prefixed form if starts with 'ال'
        if tok.startswith("ال") and len(tok) > 4:
            tokens.append(tok[2:])

        stemmed = stem_arabic_word(tok)
        if stemmed != tok:
            tokens.append(stemmed)

        if expand_numbers and tok in _NUM_TO_WORDS:
            for w in _NUM_TO_WORDS[tok]:
                norm_w = normalize_arabic(w)
                tokens.append(norm_w)
                if norm_w.startswith("ال") and len(norm_w) > 4:
                    tokens.append(norm_w[2:])
                stem_w = stem_arabic_word(norm_w)
                if stem_w != norm_w:
                    tokens.append(stem_w)

    return list(dict.fromkeys(tokens))


class BM25Retriever:
    """BM25 Okapi search engine with disk persistence."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        persist_path: Optional[str] = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._persist_path = persist_path

        self._chunks: List[Chunk] = []
        self._corpus_tokens: List[List[str]] = []
        self._doc_lens: List[int] = []
        self._avgdl: float = 0.0
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}

        if self._persist_path and Path(self._persist_path).exists():
            self.load()

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add new chunks to the BM25 index and persist."""
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self._chunks.append(chunk)
            self._corpus_tokens.append(tokens)
            self._doc_lens.append(len(tokens))

        self._recompute_idf()
        if self._persist_path:
            self.save()

    def _recompute_idf(self) -> None:
        n_docs = len(self._corpus_tokens)
        if n_docs == 0:
            self._avgdl = 0.0
            return

        self._avgdl = sum(self._doc_lens) / n_docs
        self._df.clear()

        for tokens in self._corpus_tokens:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

        self._idf.clear()
        for term, freq in self._df.items():
            self._idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 6) -> List[Tuple[Chunk, float, int]]:
        """Search index and return List of (Chunk, score, rank)."""
        if not self._chunks:
            return []

        query_tokens = tokenize(query)
        scores = [0.0] * len(self._chunks)

        for q_term in query_tokens:
            idf_val = self._idf.get(q_term, 0.0)
            if idf_val <= 0:
                continue

            for doc_idx, doc_toks in enumerate(self._corpus_tokens):
                tf = doc_toks.count(q_term)
                if tf == 0:
                    continue
                doc_len = self._doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avgdl))
                scores[doc_idx] += idf_val * (numerator / denominator)

        scored = [(self._chunks[i], scores[i]) for i in range(len(self._chunks)) if scores[i] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [(chunk, score, rank) for rank, (chunk, score) in enumerate(scored[:top_k])]

    def remove_file_chunks(self, file_path: str) -> int:
        """Remove chunks matching source file path and persist."""
        norm_target = Path(file_path).name.lower()
        new_chunks = []
        new_tokens = []
        new_lens = []

        removed = 0
        for chunk, toks, dlen in zip(self._chunks, self._corpus_tokens, self._doc_lens):
            chunk_src = Path(chunk.metadata.get("source", "")).name.lower()
            if chunk_src == norm_target:
                removed += 1
            else:
                new_chunks.append(chunk)
                new_tokens.append(toks)
                new_lens.append(dlen)

        self._chunks = new_chunks
        self._corpus_tokens = new_tokens
        self._doc_lens = new_lens
        self._recompute_idf()

        if self._persist_path:
            self.save()
        return removed

    def save(self) -> None:
        """Serialize BM25 state to disk."""
        if not self._persist_path:
            return
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({
                    "chunks": self._chunks,
                    "corpus_tokens": self._corpus_tokens,
                    "doc_lens": self._doc_lens,
                    "avgdl": self._avgdl,
                    "df": self._df,
                    "idf": self._idf,
                }, f)
            logger.debug("Persisted BM25 index (%d docs) to '%s'", len(self._chunks), path.name)
        except Exception as exc:
            logger.warning("Failed to persist BM25 index: %s", exc)

    def load(self) -> None:
        """Deserialize BM25 state from disk."""
        if not self._persist_path or not Path(self._persist_path).exists():
            return
        try:
            with open(self._persist_path, "rb") as f:
                data = pickle.load(f)
            self._chunks = data.get("chunks", [])
            self._corpus_tokens = data.get("corpus_tokens", [])
            self._doc_lens = data.get("doc_lens", [])
            self._avgdl = data.get("avgdl", 0.0)
            self._df = data.get("df", {})
            self._idf = data.get("idf", {})
            logger.info("Loaded persisted BM25 index (%d docs) from '%s'", len(self._chunks), self._persist_path)
        except Exception as exc:
            logger.warning("Failed to load persisted BM25 index: %s", exc)
