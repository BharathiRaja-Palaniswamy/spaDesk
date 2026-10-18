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

**Cost of the run (from LangSmith):** 120 traces, 1,029,393 tokens, **$4.90** in all, about 4 cents
per answer including the judge's grading calls. Median 8,868 tokens per traced question; latency
p50 4.37 s, p99 12.62 s; 0% errors.

**Full vs RAG cost (LangSmith, filtered by the `full` and `rag` tags):**

| | Full | RAG | RAG saves |
|---|---|---|---|
| Tokens, 60 questions | 684,528 | 344,865 | 50% |
| Cost | $3.12 | $1.79 | 43% |
| Median tokens per question | 9,047 | 4,065 | 55% |
| Latency p50 | 4.01 s | 5.42 s | none: RAG is slower at the median |
| Latency p99 | 12.55 s | 11.13 s | 11% faster on the slowest answers |

RAG reads about half the tokens per question. The saving is not the full 4x of the document alone
(about 4,000 tokens against about 1,000 for 7 chunks) because the rest of each request is the same in
both modes: the instructions, the reply format, the owner's report-tool round-trips, the answer
itself and the judge's grading call. RAG is not faster: the search adds a step, and the time to
write the answer dominates.

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

### 2a. A stricter retrieval score (measurement, not a fix)

The free retrieval check now asks whether the chunk holding the answer was retrieved, not just any
chunk from the right section. Short answers such as "10:00 AM" appear in several chunks, so 17 cases
also list `evidence`: the exact line of the document that holds the answer ("10:00 AM – Restorative
Yoga"), which appears in only one chunk.

| Score | Hit rate at k=7 | Agrees with the paid run? |
|---|---|---|
| Section-level (before) | 36/39 (92%) | no: counted 3 of the RAG failures as hits |
| **Fact-level (now)** | **24/29 (83%)** | **yes: its 5 misses are exactly the 5 RAG fact failures** |

It now predicts retrieval failures in seconds, for free, so retrieval changes can be tried on it
before paying for a full run.

### 2b. Why answers were missed: how the document is cut

The document is fine; the cut points are the problem. Chunks are cut every 150 tokens wherever that
lands, so a chunk can start mid-section and lose its heading to the chunk before. The chunk holding the
yoga studio's Sunday hours begins mid-line (" 8:30 PM"), has three lines of hours, and is otherwise the
Facials list. Nothing in it says "yoga" or "studio", so to the search it is a chunk about facials, and it
ranked 21st for "What are the yoga studio hours on Sunday?". Every RAG miss had the same shape: the
answer line sat in a chunk that was mostly about something else.

**The section metadata did not help, by design.** Each chunk is stored with the headings it overlaps
(`sections`), but the search compares vectors only. The metadata is used after retrieval: it decides
which sections an answer may cite, fills the "From:" line under an answer, and appears in traces and in
the eval. Filtering the search by metadata was discussed and deferred.

### 2c. Experiment: reranking and contextual chunk headers (free, retrieval score only)

Two ideas, tried in a scratch script before any app change:

- **Reranking:** fetch the 20 closest chunks, have a small cross-encoder model (ms-marco-MiniLM-L-6-v2,
  runs on the machine, no API key) read the question and each chunk together and score how well the
  chunk answers it, then keep the best 7. The model still receives 7 chunks, so prompt size and cost do
  not change. The first search compares a question and a chunk that were turned into vectors
  separately; the reranker reads them side by side, which is more accurate and slower, so it only sees
  the top 20.
- **Contextual chunk headers:** keep the same cut points, but write each chunk's headings into the text
  that is embedded, so the vector itself carries the topic. Version A uses the `##`/`###` headings;
  version B adds the bold labels the chunk belongs to (entry titles such as "Yoga Studio Hours" or a
  service name). A chunk that spans two sections gets both. Unlike metadata, this changes the search.

| Setup | Fact hit rate (k=7) | Still missed |
|---|---|---|
| Baseline | 24/29 (83%) | 5 |
| Reranking only | 26/29 (90%) | studio hours, yoga class list, "u guys open sat?" |
| Headers A only | 23/29 (79%) | 6 |
| Headers B only | 24/29 (83%) | 5 |
| **Headers A + reranking** | **27/29 (93%)** | studio hours, "u guys open sat?" |
| **Headers B + reranking** | **27/29 (93%)** | "u guys open sat?", membership with yoga |

Other settings tried with reranking: fetching 15 or 33 instead of 20 gave the same score (15 is
faster); keeping 5 chunks instead of 7 lost 1 to 2 cases; a larger reranker (bge-reranker-base) gave the
same score at about 5 times the time (389 ms against 74 ms per question).

**What we learned**
- **Headers alone do not help the first search.** The embedder averages the whole chunk; a short line
  of headings barely moves it.
- **Headers do help the reranker.** It reads the question and chunk together, so a line naming the
  chunk's topics changes its judgment. Version B recovered the studio hours (rank 21) for the first time.
- **Reranking alone fixed 4 and broke 2.** It pushed parts of the yoga class list out, and it misreads
  "u guys open sat?" (it was trained on well-formed queries). Headers fixed the class list.
- **Casual slang is a real limit of these small models.** "u guys open sat?" fails in every setup.
- **Keeping fewer chunks hurts here.** Cutting from 7 to 5 to shrink the prompt (fighting context rot)
  cost accuracy, so 7 stays.

## 3. Decisions

| Decision | Choice | Why |
|---|---|---|
| Chunking | fixed 150 tokens, 30 overlap; not section-based | kept by choice; headers give chunks their topic without moving the cut points |
| Embedder | all-MiniLM-L6-v2, on the machine | free, and 150-token chunks fit its 256-token limit |
| Chunks sent to the model | 7 | 5 lost accuracy in the experiment |
| Metadata filtering | not now | deferred |
| Reranking | yes: MiniLM-L-6 cross-encoder, 20 candidates, keep 7, on the machine | +3 cases with headers, 74 ms per question, no API key, no cost per question |
| Contextual chunk headers | yes, version to choose (A or B) | needed for reranking to work well; both scored 27/29 |
| Eval answers that change with the data | owner figures computed from the bookings at run time; document facts kept as written, guarded by a drift check | figures change when bookings are regenerated; facts change rarely and the drift check names any stale case |
| RAG on/off switch | hide on the customer site before submission; keep full-document mode for evals | customers should not choose; full mode is the baseline that shows what RAG costs |
| LangChain | after submission | a rewrite now would change behaviour on the deadline day; the eval is the baseline to compare it against |

## 4. Results by step

| Step | Change | Why | RAG pass rate | Fact hit rate (k=7) | Notes |
|---|---|---|---|---|---|
| Baseline | top 7 chunks, dense search only | | 54/60 (90%) | 24/29 (83%) | |
| Reranking | first search fetches 20, MiniLM-L-6 cross-encoder keeps the best 7 (`spadesk/rag/rerankers/`) | near misses at ranks 8 to 13 | **56/60 (93%)**; full 58/60 | 26/29 (90%) | see 2d |

### 2d. Reranking: the paid run (18 Oct, `--label rerank`, `docs/eval-runs/2026-10-18-rerank.txt`)

| | Baseline RAG | Reranking RAG |
|---|---|---|
| Pass rate | 54/60 (90%) | **56/60 (93%)** |
| Average answer time | 4.9 s | 5.0 s |
| Slowest answer | 12.8 s | 12.5 s |

**Fixed by reranking** (failed with RAG in the baseline, pass now): Restorative Yoga on Sunday, the late
cancellation fee, the membership that includes yoga, requesting Priya. First-class-free now answers
correctly too ("book by phone or at the front desk" reached the model), but still fails one check, see below.

**Broken by reranking, as the experiment predicted:** the yoga class list (Kids Yoga missing) and
"u guys open sat?". **Still missed:** the yoga studio's Sunday hours, whose chunk ranks 21st, outside
the 20 candidates the reranker sees.

**Failures that are the eval's fault, not the bot's** (to fix in the cases):
- *first-class-free:* the answer says "call us"; the check looks for the word "phone". Correct answer, too-strict check.
- *out-of-scope-botox (full mode):* the model answered "we don't offer that" and marked it covered. As with
  live availability, `covered` is ambiguous here; the behaviour should be judged instead.
- *groom-service (full mode):* the judge failed "does not invent a service" because it cannot see the
  document, so it could not tell that the Bridal Glow Prep Package is real. A criterion that needs facts
  cannot be judged from the answer alone; it has to be reworded.

**Fixed after this run, in `cases.json`:** first-class-free checks "front desk" and judges "by phone or
at the front desk"; out-of-scope-botox is judged ("does not offer Botox"), not checked with `covered`;
groom-service's criteria no longer need facts the judge cannot see. Not yet rerun.

**Run-to-run variation:** full mode did not change (no retrieval), yet scored 58 against 59: the botox
and groom answers differed from the baseline run's, and live availability passed this time. One run of
each is a measurement with noise of about one case.

**Cost:** to read from LangSmith (filter on the tag `run:rerank`). Reranking adds no tokens: the model
still receives 7 chunks.
