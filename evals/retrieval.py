"""Scores retrieval alone: does the search find the right sections?

    .venv/bin/python -m evals.retrieval

RAG has two steps: retrieval (embed the question, find the closest chunks)
and generation (the chat model writes the answer from them). This file
checks the first step only. The embedder runs on this machine and Chroma is
a local file, so a run costs nothing and takes seconds.

For every case that lists `cites`, the expected sections are looked up among
the sections of the retrieved chunks:

  hit   every expected section is in the top-k chunks (k = SPADESK_RETRIEVAL_TOP_K)
  miss  at least one is not; the model never sees it, so the answer cannot use it

For each expected section the rank of the first chunk that contains it is
printed too. A miss at rank 9 with k = 7 suggests a larger k or reranking; a
miss at rank 30 suggests the question and the chunk do not look alike to
the embedder (hybrid search, or better chunking).
"""

import json
from pathlib import Path

from spadesk.config import get_settings
from spadesk.rag.embeddings import get_embedder
from spadesk.rag.index import ensure_index
from spadesk.rag.stores import get_store

from .validate import stale_cases

CASES = Path(__file__).parent / "cases.json"
# Large enough to rank every chunk of the document, so a missed section's
# rank can still be shown.
EVERY_CHUNK = 1000


def section_ranks(question: str) -> dict[str, int]:
    """Each section's best rank for `question`: 1 when the closest chunk
    contains it, 2 for the next one, and so on."""
    vector = get_embedder().embed([question])[0]
    ranks: dict[str, int] = {}
    for rank, match in enumerate(get_store().search(vector, EVERY_CHUNK), start=1):
        for section in match.chunk.sections:
            ranks.setdefault(section, rank)  # keeps the first (best) rank only
    return ranks


def main() -> None:
    """Prints a hit or miss per case, then the hit rate."""
    ensure_index()
    k = get_settings().retrieval_top_k
    all_cases = json.loads(CASES.read_text())
    if stale := stale_cases(all_cases):
        raise SystemExit("Update these cases first:\n  " + "\n  ".join(stale))
    cases = [case for case in all_cases if case["expect"].get("cites")]
    hits = 0

    for case in cases:
        ranks = section_ranks(case["question"])
        expected = case["expect"]["cites"]
        hit = all(ranks.get(section, EVERY_CHUNK) <= k for section in expected)
        hits += hit
        where = ", ".join(f"{section} #{ranks.get(section, '-')}" for section in expected)
        print(f"{'HIT ' if hit else 'MISS'}  {case['id']:32} {where}")

    print(f"\nRetrieval hit rate at k={k}: {hits}/{len(cases)} ({hits / len(cases):.0%})")


if __name__ == "__main__":
    main()
