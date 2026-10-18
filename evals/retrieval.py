"""Scores retrieval alone: does the search hand the model what it needs?

    .venv/bin/python -m evals.retrieval                      # the hand-built pipeline
    .venv/bin/python -m evals.retrieval --engine langchain   # the LangChain one

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

import argparse
import json
from pathlib import Path

from langsmith import tracing_context

from spadesk.config import get_settings
from spadesk.lc.rerankers import get_compressor
from spadesk.lc.vectorstores import get_vectorstore
from spadesk.rag.embeddings import get_embedder
from spadesk.rag.index import ensure_index
from spadesk.rag.rerankers import get_reranker
from spadesk.rag.retriever import rerank
from spadesk.rag.stores import get_store

from .validate import stale_cases

CASES = Path(__file__).parent / "cases.json"
# Large enough to rank every chunk of the document, so a missed section's
# rank can still be shown.
EVERY_CHUNK = 1000
# Shown when an expected section or text is in no chunk at all.
NOWHERE = EVERY_CHUNK + 1


# One chunk as the scores see it: its text and its headings.
Ranked = list[tuple[str, list[str]]]


def ranked_handbuilt(question: str) -> Ranked:
    """Every chunk, in the order the hand-built pipeline ranks them.

    The same steps as retrieve() in spadesk/rag/retriever.py: the first
    search, then, with a reranker on, its top SPADESK_RERANK_CANDIDATES
    reordered by the reranker. The rest follow in the first search's order,
    so a chunk the reranker never saw still has a rank to show.
    """
    ensure_index()
    matches = get_store().search(get_embedder().embed([question])[0], EVERY_CHUNK)
    if get_reranker() is not None:
        candidates = get_settings().rerank_candidates
        matches = rerank(question, matches[:candidates]) + matches[candidates:]
    return [(m.chunk.text, list(m.chunk.sections)) for m in matches]


def ranked_langchain(question: str) -> Ranked:
    """Every chunk, in the order the LangChain pipeline ranks them: the same
    steps as SpaRetriever in spadesk/lc/retriever.py, without the cut to top-k."""
    documents = [doc for doc, _ in get_vectorstore().similarity_search_with_relevance_scores(question, EVERY_CHUNK)]
    if (compressor := get_compressor()) is not None:
        candidates = get_settings().rerank_candidates
        documents = list(compressor.compress_documents(documents[:candidates], question)) + documents[candidates:]
    return [(doc.page_content, doc.metadata["sections"].split("\n")) for doc in documents]


ENGINES = {"handbuilt": ranked_handbuilt, "langchain": ranked_langchain}


def section_rank(ranked: Ranked, section: str) -> int:
    """The rank of the first chunk labelled with `section` (1 is the closest)."""
    return next((rank for rank, (_, sections) in enumerate(ranked, start=1) if section in sections), NOWHERE)


def fact_rank(ranked: Ranked, text: str) -> int:
    """The rank of the first chunk whose text contains `text`, ignoring case."""
    return next((rank for rank, (chunk, _) in enumerate(ranked, start=1) if text.lower() in chunk.lower()),
                NOWHERE)


def score(expected: list[str], ranks: list[int], k: int) -> tuple[bool, str]:
    """Whether every expected item is in the top k, and each item with its rank."""
    shown = ", ".join(f"{item} #{rank if rank != NOWHERE else '-'}" for item, rank in zip(expected, ranks))
    return all(rank <= k for rank in ranks), shown


def main() -> None:
    """Prints both scores per case, then both hit rates."""
    parser = argparse.ArgumentParser(description="Score retrieval alone, without calling the model.")
    parser.add_argument("--engine", choices=ENGINES, default="handbuilt",
                        help="which pipeline to score: the hand-built one (spadesk/rag/) or LangChain (spadesk/lc/)")
    engine = parser.parse_args().engine
    ranked_chunks = ENGINES[engine]
    cases = json.loads(CASES.read_text())
    if stale := stale_cases(cases):
        raise SystemExit("Update these cases first:\n  " + "\n  ".join(stale))
    settings = get_settings()
    k = settings.retrieval_top_k
    totals = {"section": [0, 0], "fact": [0, 0]}  # [hits, cases scored]

    print(f"Engine: {engine} | reranker: {settings.reranker} (candidates: {settings.rerank_candidates})\n")
    print(f"{'':32} {'section':8} {'fact':8}")
    for case in cases:
        cites = case["expect"].get("cites", [])
        facts = case.get("evidence") or case["expect"].get("mentions", [])
        if not (cites or facts):
            continue  # nothing to retrieve: judge-only or not-covered cases
        ranked = ranked_chunks(case["question"])
        columns, details = [], []
        for kind, expected, rank_of in (("section", cites, section_rank), ("fact", facts, fact_rank)):
            if not expected:
                columns.append("-")
                continue
            hit, shown = score(expected, [rank_of(ranked, item) for item in expected], k)
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
