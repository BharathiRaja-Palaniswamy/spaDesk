# spaDesk RAG: where it started and how it improved

A one-page record of the evaluation: what was measured, the first result,
what it revealed, and each change since. Each run's full output is kept in
`docs/eval-runs/`. How to run the evals: `evals/` (see README).

## The setup

- **Corpus:** one business document for Pebble Spa (`data/business_doc.md`, about 4,000 tokens),
  plus generated bookings data for the owner's figures.
- **Pipeline:** fixed 150-token chunks with 30-token overlap (33 chunks), all-MiniLM-L6-v2
  embeddings on the machine, Chroma with cosine distance, top 7 chunks into the prompt, Claude for the answer.
- **The test set:** 60 questions (51 customer, 9 owner), each asked two ways:
  - **Full:** the model is given the whole document. The upper bound.
  - **RAG:** the model is given only the 7 retrieved chunks. The real product.

  Same model, same prompt, so any difference between the two is caused by retrieval.
- **How answers are graded:**

  | Check | What it proves | How |
  |---|---|---|
  | covered | the bot said whether the document had the answer | code |
  | cites | the answer came from the right section | code |
  | mentions | the exact fact is in the answer (a price, a time, "15%") | code |
  | mentions_report | the owner's figures match the bookings data, worked out at run time | code |
  | uses_report | the owner's figures came from the report tool, not invented | code |
  | judge | yes/no, safety and guardrail behaviour, graded by a second model call | model |

  A drift check stops a run if the document or data changed under a case.

## 1. Baseline (18 Oct 2026)

| | Full | RAG |
|---|---|---|
| Pass rate | **59/60 (98%)** | **54/60 (90%)** |
| Average answer time | 4.8 s | 4.9 s |
| Slowest answer | 12.7 s | 12.8 s |

**In plain words.** With the whole document, the bot is right almost every time. With RAG, a search first
picks the 7 most relevant chunks and the bot sees only those. It lost 5 questions because the search
handed over chunks on the right topic but not the chunk holding the exact fact.

**What went well**
- **No made-up answers:** when the right chunk was missing, the bot said it could not find the
  answer instead of guessing.
- **Safety and guardrails held:** no medical claims, nothing about other customers, no business
  figures shown to customers. All 16 judged cases passed.
- **Owner figures correct:** all 9 owner cases passed in both modes, every figure from the report tool.

**What failed with RAG, and why.** The chunk that holds the fact was not in the top 7:

| Question | The fact | Its chunk's rank |
|---|---|---|
| What time is Restorative Yoga on Sunday? | 10:00 AM | 12 |
| What are the yoga studio hours on Sunday? | 8:00 AM – 4:00 PM | 21 |
| Is there a membership that includes yoga? | Wellness Plus | 13 |
| What if I cancel a facial late? | 50% fee | 11 |
| Is my first yoga class free? How do I book? | by phone or at the front desk | 9 |
| Can I request Priya for a prenatal massage? | the FAQ on requesting a therapist | 15 |

**Root cause: fixed-size chunks cut across sections.** For example, the chunk holding Sunday's
timetable also holds Studio Notes and part of Packages, so its meaning is blurred and the search
ranks it low.

**What we learned about our own measurement**
- The free retrieval check (`evals/retrieval.py`) said 92% of searches found the right section. It
  counted any chunk from the right section, even when that chunk did not hold the answer. It was
  too lenient, so it will be made fact-level.
- One case expected "not covered" for live availability. With the full document the bot reasonably
  answered from the staff schedule instead. That expectation is debatable, and it is better graded
  by the judge alone.
- Answer time is the same in both modes: the model writing the answer dominates, not the search.
  On a document this small, RAG's benefit is the smaller prompt (cost), and it matters more as the
  document grows.

## 2. Improvements

*Each step: the change, why, and the result compared with the baseline.*

| Step | Change | Why | RAG pass rate | Retrieval hit rate | Notes |
|---|---|---|---|---|---|
| Baseline | top 7 chunks, dense search only | | 54/60 (90%) | 92% (section-level, too lenient) | |
| | | | | | |
