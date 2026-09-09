# RAG_XPER Retrieval Evaluation Baseline

- **Date:** 2026-09-05
- **Dataset Questions:** 25
- **Top K:** 6
- **Recall@6:** 100.0% (25/25)
- **MRR (Mean Reciprocal Rank):** 0.9800

### Methodology
Every query is tested against Arabic legal chunks with exact article matching (`article_number` and gold span).
Recall@6 measures whether the ground-truth article was placed in the top 6 retrieved items passed to the LLM context.
