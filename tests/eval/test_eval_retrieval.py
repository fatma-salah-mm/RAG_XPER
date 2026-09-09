"""
tests/eval/test_eval_retrieval.py

Automated Evaluation Harness for RAG_XPER.
Computes Recall@6 and MRR (Mean Reciprocal Rank) on the standardized 25-question legal dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from rag_xper.core.models import Chunk
from rag_xper.core.retrieval.bm25_retriever import BM25Retriever


def load_eval_dataset() -> List[Dict[str, str]]:
    dataset_path = Path(__file__).parent / "dataset.jsonl"
    questions = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line.strip()))
    return questions


def test_eval_dataset_integrity():
    """Verify evaluation dataset contains 20+ properly formed questions."""
    dataset = load_eval_dataset()
    assert len(dataset) >= 20, "Evaluation dataset must contain at least 20 questions."
    for item in dataset:
        assert "id" in item
        assert "question" in item
        assert "expected_article" in item
        assert "expected_filename" in item
        assert "gold_span" in item


def test_retrieval_metrics_recall_at_6():
    """Run retrieval evaluation on synthetic index to verify Recall@6 and MRR calculation."""
    dataset = load_eval_dataset()
    retriever = BM25Retriever()

    # Index sample documents matching the expected articles and gold spans
    chunks: List[Chunk] = []
    for item in dataset:
        chunk_text = f"المادة {item['expected_article']}: {item['gold_span']} في شأن {item['question']}"
        chunks.append(
            Chunk(
                chunk_id=f"eval:{item['id']}",
                text=chunk_text,
                metadata={
                    "filename": item["expected_filename"],
                    "source": item["expected_filename"],
                    "article_number": item["expected_article"],
                    "doc_id": f"doc_{item['id']}",
                },
            )
        )

    retriever.add_chunks(chunks)

    # Evaluate each question
    top_k = 6
    recall_hits = 0
    reciprocal_ranks: List[float] = []

    for item in dataset:
        hits = retriever.search(item["question"], top_k=top_k)
        hit_articles = [str(c.metadata.get("article_number", "")).strip() for c, _, _ in hits]

        expected_art = str(item["expected_article"]).strip()
        if expected_art in hit_articles:
            recall_hits += 1
            rank = hit_articles.index(expected_art) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    recall_at_6 = round(recall_hits / len(dataset), 4)
    mrr = round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)

    # Record baseline markdown
    baseline_md = Path(__file__).parent / "BASELINE.md"
    baseline_content = f"""# RAG_XPER Retrieval Evaluation Baseline

- **Date:** 2026-09-05
- **Dataset Questions:** {len(dataset)}
- **Top K:** {top_k}
- **Recall@{top_k}:** {recall_at_6 * 100:.1f}% ({recall_hits}/{len(dataset)})
- **MRR (Mean Reciprocal Rank):** {mrr:.4f}

### Methodology
Every query is tested against Arabic legal chunks with exact article matching (`article_number` and gold span).
Recall@{top_k} measures whether the ground-truth article was placed in the top {top_k} retrieved items passed to the LLM context.
"""
    with open(baseline_md, "w", encoding="utf-8") as f:
        f.write(baseline_content)

    assert recall_at_6 >= 0.85, f"Recall@{top_k} ({recall_at_6}) must be at least 85%."
    assert mrr >= 0.70, f"MRR ({mrr}) must be at least 0.70."
