"""
core/retrieval/bm25_retriever.py for RAG_XPER

Lightweight BM25 retriever optimized for Arabic and English text.
Performs Arabic normalization, stemming, and numeral-to-ordinal expansion.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

from core.models import Chunk, RetrievedChunk

_TASHKEEL_REGEX = re.compile(r"[\u064B-\u0652\u0670\u0640]")


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for robust keyword matching."""
    text = _TASHKEEL_REGEX.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[ىي]", "ي", text)
    text = re.sub(r"ة", "ه", text)
    return text.lower()


def clean_arabic_token(token: str) -> str:
    """Normalize and stem common Arabic prefixes and suffixes for legal matching."""
    t = normalize_arabic(token)

    if len(t) > 3 and t.startswith(("و", "ف", "ب", "ك", "ل")):
        t = t[1:]

    if len(t) > 4 and t.startswith("ال"):
        t = t[2:]

    t = re.sub(r"^(عشر|ثلاث|اربع|خمس|ست|سبع|ثمان|تسع)(ون|ين)$", r"\1", t)
    t = re.sub(r"^(حاد|ثان|ثالث|رابع|خامس|سادس|سابع|ثامن|تاسع|عاشر)(ي|يه|ه)?$", r"\1", t)
    return t


def tokenize(text: str) -> List[str]:
    """Tokenize normalized text into words/numbers with intelligent Arabic stemming."""
    spaced = re.sub(r"(\d+)", r" \1 ", text)
    raw_tokens = re.findall(r"[\w]+", spaced)

    tokens: List[str] = []
    for raw in raw_tokens:
        clean = clean_arabic_token(raw)
        if clean:
            tokens.append(clean)
            norm_orig = normalize_arabic(raw)
            if norm_orig != clean:
                tokens.append(norm_orig)
    return tokens


_UNITS = {
    1: ("الاولى", "الاول", "حادي", "حاد"),
    2: ("الثانية", "الثاني", "ثاني", "ثان"),
    3: ("الثالثة", "الثالث", "ثالث"),
    4: ("الرابعة", "الرابع", "رابع"),
    5: ("الخامسة", "الخامس", "خامس"),
    6: ("السادسة", "السادس", "سادس"),
    7: ("السابعة", "السابع", "سابع"),
    8: ("الثامنة", "الثامن", "ثامن"),
    9: ("التاسعة", "التاسع", "تاسع"),
}

_TENS = {
    10: "العاشرة",
    20: "العشرون",
    30: "الثلاثون",
    40: "الاربعون",
    50: "الخمسون",
    60: "الستون",
    70: "السبعون",
    80: "الثمانون",
    90: "التسعون",
}


def number_to_arabic_words(n: int) -> List[str]:
    """Convert integer number (1-200) to Arabic legal ordinal forms."""
    words = []
    if n in _UNITS:
        words.extend(_UNITS[n])
    elif n in _TENS:
        words.append(_TENS[n])
        words.append(_TENS[n].replace("ون", "ين"))
    elif 11 <= n <= 19:
        unit = n % 10
        unit_words = _UNITS.get(unit, ())
        for u in unit_words:
            words.append(f"{u} عشر")
            words.append(f"{u} عشره")
    elif 21 <= n <= 99:
        unit = n % 10
        ten = (n // 10) * 10
        ten_nom = _TENS.get(ten, "")
        ten_gen = ten_nom.replace("ون", "ين")
        unit_words = _UNITS.get(unit, ())
        for u in unit_words:
            words.append(f"{u} و{ten_nom}")
            words.append(f"{u} و{ten_gen}")
    elif n == 100:
        words.extend(["المائه", "المئه", "المائة", "المئة"])
    elif n > 100:
        rem = n - 100
        rem_words = number_to_arabic_words(rem)
        for rw in rem_words:
            words.append(f"{rw} بعد المائه")
            words.append(f"{rw} بعد المائة")
    return words


def expand_query_tokens(query: str) -> List[str]:
    """Extract query tokens plus Arabic word expansions for digits."""
    base_tokens = tokenize(query)
    extra_tokens: List[str] = []

    digits = [int(d) for d in re.findall(r"\b(\d+)\b", query) if 1 <= int(d) <= 200]
    for d in digits:
        for phrase in number_to_arabic_words(d):
            extra_tokens.extend(tokenize(phrase))

    return list(dict.fromkeys(base_tokens + extra_tokens))


class BM25Retriever:
    """In-memory BM25Okapi index for lexical chunk retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.total_docs: int = 0

    def add_chunks(self, new_chunks: List[Chunk]) -> None:
        if not new_chunks:
            return

        for chunk in new_chunks:
            self.chunks.append(chunk)
            tokens = tokenize(chunk.text)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))

        self.total_docs = len(self.chunks)
        self.avgdl = sum(self.doc_lengths) / self.total_docs if self.total_docs > 0 else 0.0

        self.doc_freqs = {}
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log(1.0 + (self.total_docs - freq + 0.5) / (freq + 0.5))

    def clear(self) -> None:
        self.chunks = []
        self.doc_tokens = []
        self.doc_lengths = []
        self.avgdl = 0.0
        self.doc_freqs = {}
        self.idf = {}
        self.total_docs = 0

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float, int]]:
        if self.total_docs == 0:
            return []

        query_tokens = expand_query_tokens(query)
        if not query_tokens:
            return []

        scores: List[float] = [0.0] * self.total_docs

        for q_token in query_tokens:
            if q_token not in self.idf:
                continue
            idf_val = self.idf[q_token]

            for doc_idx, doc_toks in enumerate(self.doc_tokens):
                tf = doc_toks.count(q_token)
                if tf == 0:
                    continue

                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl + 1e-10)))
                scores[doc_idx] += idf_val * (numerator / denominator)

        ranked_indices = sorted(
            [i for i in range(self.total_docs) if scores[i] > 0.0],
            key=lambda idx: scores[idx],
            reverse=True,
        )

        results: List[Tuple[Chunk, float, int]] = []
        for rank, idx in enumerate(ranked_indices[:top_k]):
            results.append((self.chunks[idx], scores[idx], rank))

        return results
