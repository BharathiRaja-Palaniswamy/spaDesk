"""Scores retrieval alone: does the search hand the model what it needs?

    .venv/bin/python -m evals.retrieval

RAG has two steps: retrieval (embed the question, find the closest chunks)
and generation (the chat model writes the answer from them). This file
checks the first step only. The embedder runs on this machine and Chroma is
a local file, so a run costs nothing and takes seconds, which makes it the
tool for trying retrieval changes (top-k, reranking) before a paid eval run.

Each case is scored two ways, from what it already lists in cases.json:

  section  (from `cites`)     a top-k chunk is labelled with each expected section
  fact     (from `evidence`,  a top-k chunk's text contains each expected text
            else `mentions`)

`evidence` is the exact line of the document that holds the answer, e.g.
"10:00 AM – Restorative Yoga". It is listed for cases whose `mentions` are
too common to pin down one chunk: "10:00 AM" alone also appears in the
opening hours, so it would be found in the wrong chunk and count as a hit.

The fact score is the one to trust. A section spans several chunks, so the
section score passes as soon as ANY chunk of the right section comes back,
even when the chunk holding the answer does not: in the baseline, the
Sunday timetable chunk ranked 12th while another timetable chunk ranked 1st.
The section score is kept so the two can be compared.

Chunks are ranked as the app ranks them, reranker included (SPADESK_RERANKER);
run with SPADESK_RERANKER=none to see the first search alone. Nothing is
traced to LangSmith: this is a local measurement.

For each expected section or text, the rank of the first chunk that has it
is printed (k = SPADESK_RETRIEVAL_TOP_K). A fact at rank 9 with k = 7 is a
near miss that reranking can recover; one at rank 30 means the question and
the chunk do not look alike to the embedder.
"""

import json
from pathlib import Path

from langsmith import tracing_context

from spadesk.config import get_settings
from spadesk.rag.embeddings import get_embedder
from spadesk.rag.index import ensure_index
from spadesk.rag.rerankers import get_reranker
from spadesk.rag.retriever import rerank
from spadesk.rag.stores import get_store
from spadesk.rag.stores.base import Match

from .validate import stale_cases

CASES = Path(__file__).parent / "cases.json"
# Large enough to rank every chunk of the document, so a missed section's
# rank can still be shown.
EVERY_CHUNK = 1000
# Shown when an expected section or text is in no chunk at all.
NOWHERE = EVERY_CHUNK + 1


def ranked_chunks(question: str) -> list[Match]:
    """Every chunk of the document, in the order the app would rank them.

    The same steps as retrieve() in spadesk/rag/retriever.py: the first
    search, then, with a reranker on, its top SPADESK_RERANK_CANDIDATES
    reordered by the reranker. The rest follow in the first search's order,
    so a chunk the reranker never saw still has a rank to show.
    """
    vector = get_embedder().embed([question])[0]
    matches = get_store().search(vector, EVERY_CHUNK)
    if get_reranker() is None:
        return matches
    candidates = get_settings().rerank_candidates
    return rerank(question, matches[:candidates]) + matches[candidates:]


def section_rank(matches: list[Match], section: str) -> int:
    """The rank of the first chunk labelled with `section` (1 is the closest)."""
    return next((rank for rank, m in enumerate(matches, start=1) if section in m.chunk.sections), NOWHERE)


def fact_rank(matches: list[Match], text: str) -> int:
    """The rank of the first chunk whose text contains `text`, ignoring case."""
    return next((rank for rank, m in enumerate(matches, start=1) if text.lower() in m.chunk.text.lower()),
                NOWHERE)


def score(expected: list[str], ranks: list[int], k: int) -> tuple[bool, str]:
    """Whether every expected item is in the top k, and each item with its rank."""
    shown = ", ".join(f"{item} #{rank if rank != NOWHERE else '-'}" for item, rank in zip(expected, ranks))
    return all(rank <= k for rank in ranks), shown


def main() -> None:
    """Prints both scores per case, then both hit rates."""
    cases = json.loads(CASES.read_text())
    if stale := stale_cases(cases):
        raise SystemExit("Update these cases first:\n  " + "\n  ".join(stale))
    ensure_index()
    settings = get_settings()
    k = settings.retrieval_top_k
    totals = {"section": [0, 0], "fact": [0, 0]}  # [hits, cases scored]

    print(f"Reranker: {settings.reranker} (candidates: {settings.rerank_candidates})\n")
    print(f"{'':32} {'section':8} {'fact':8}")
    for case in cases:
        cites = case["expect"].get("cites", [])
        facts = case.get("evidence") or case["expect"].get("mentions", [])
        if not (cites or facts):
            continue  # nothing to retrieve: judge-only or not-covered cases
        matches = ranked_chunks(case["question"])
        columns, details = [], []
        for kind, expected, rank_of in (("section", cites, section_rank), ("fact", facts, fact_rank)):
            if not expected:
                columns.append("-")
                continue
            hit, shown = score(expected, [rank_of(matches, item) for item in expected], k)
            totals[kind][0] += hit
            totals[kind][1] += 1
            columns.append("HIT" if hit else "MISS")
            details.append(f"{kind}: {shown}")
        print(f"{case['id']:32} {columns[0]:8} {columns[1]:8} {' | '.join(details)}")

    print()
    for kind, (hits, scored) in totals.items():
        print(f"{kind.capitalize():8} hit rate at k={k}: {hits}/{scored} ({hits / scored:.0%})")


if __name__ == "__main__":
    with tracing_context(enabled=False):  # the reranker step is traced in the app; not here
        main()
