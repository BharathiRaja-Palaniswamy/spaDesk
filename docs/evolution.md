# How spaDesk evolved

The story of the project, stage by stage: what was built, why, what we found,
and what that led to. Measurements are in [eval-report.md](eval-report.md)
(with each run's output in `eval-runs/`); how the RAG steps work is in
[rag-notes.md](rag-notes.md).

## Version 1: an assistant that reads the whole document (7–10 Oct 2026)

**Built.** A customer site and an owner's manage page for a fictional spa,
Pebble Spa: services, prices, hours and a booking request form for customers;
a reports dashboard (revenue, bookings, channels, feedback) for the owner. An
assistant on the manage page answered questions by sending the whole business
document to Claude with every question.

**Decisions that shaped what came next.**
- **Structured replies.** The model fills a fixed shape (answer, list items,
  covered, sections, suggestions) instead of free text, so the page can show a
  list as a list, name its sources, and style an answer the document does not
  cover differently.
- **Facts only in the document.** Every service, price and policy lives in one
  Markdown file (`data/business_doc.md`); bookings live in one JSON file. The
  code names headings, never facts.
- **The model is told the current time,** because it has no clock and
  questions such as "are you open now?" depend on it.

## Stage 1: retrieval-augmented generation (13 Oct)

**Built.** The document is cut into fixed 150-token chunks with a 30-token
overlap, each chunk turned into a vector by a small local model
(all-MiniLM-L6-v2), stored in Chroma, and the closest chunks found for each
question. A RAG switch in the chat sends either the whole document or only the
retrieved chunks, so the same question can be asked both ways.

**Why.** To learn RAG by building each step by hand, and because a real
business's documents will not always fit in a prompt.

**Findings.**
- Every provider (embedding model, vector store) sits behind a small interface
  and is chosen by a setting, so it can be swapped without touching the rest.
- The index is rebuilt only when a fingerprint of the document, chunk settings
  and model changes, checked on every question: an edited document is picked
  up without a restart.

## Stage 2: observability and a tool for the owner (15 Oct)

**Built.** LangSmith tracing of every question (the chunks retrieved and their
scores, the exact prompt, the reply, tokens and time). A `get_report` tool:
the owner's assistant asks the server for report figures for the dates it has
worked out, and answers from them, with short insights on what they mean.

**Findings.**
- **RAG vs tools.** The code always retrieves for RAG; with a tool, the model
  decides when to ask. Owner figures come only from the tool, so the model
  never estimates a number.
- **Guardrails belong in what the model can reach.** The customer chat
  (stage 3) is given no tools and no bookings, so it cannot reveal revenue or
  another customer's visits, whatever it is asked. Telling a model "don't
  share revenue" is only a soft guard.

## Stage 3: a customer chat and a deeper document (16 Oct)

**Built.** A redesigned customer site: a hero, the whole document laid out in
bands below it, and a chat panel that slides in from the right, answering from
the business document with no tools. The prompt and model call were moved into
code both assistants share. The document grew to about 4,000 tokens (33
chunks): a yoga studio with a timetable, the team, service suitability for
pregnancy, heat, skin and health conditions, facilities and more FAQs; the
bookings were regenerated to match (2,949).

**Why.** A richer document makes retrieval harder and more realistic:
answers now need facts from more than one section.

**Finding.** "Can I have the Sea Mineral Body Wrap while pregnant?" needs two
sections (Pregnancy says no heat-based treatments; Body Treatments says the
wrap is warmed). Plain retrieval at 7 chunks found both, so linking facts
across sections did not need graph RAG here.

## Stage 4: evaluation (18 Oct)

**Built.** 60 questions (51 customer, 9 owner), each asked with the whole
document and with RAG. Code checks (covered, cited sections, exact facts in the
answer, owner figures read from the bookings at run time, whether the report
tool was used), a second model call as judge for yes/no, safety and guardrail
answers, a drift check that stops a run when a case no longer matches the
document, and every run traced to its own LangSmith project.

**Baseline.** Whole document 59/60; RAG 54/60. Cost $4.90 for the run. RAG
used half the tokens and was 43% cheaper, at about the same speed.

**Findings.**
- **The misses were in the cut, not the model.** Each RAG failure had the
  answer line in a chunk mostly about something else. The yoga studio's Sunday
  hours start a chunk that is otherwise the Facials list; the heading fell into
  the chunk before, so the chunk ranked 21st.
- **When retrieval missed, the bot said it could not find the answer.** No
  made-up answers: the refusal path worked.
- **Our first retrieval score was too lenient.** "Was any chunk from the right
  section retrieved?" said 92%. "Was the chunk holding the answer retrieved?"
  said 83% and named exactly the cases the paid run failed. Each case now
  lists the exact line that holds its answer (`evidence`).
- **Answers that change with the data are worked out, not written down.**
  Owner figures are computed from the bookings when the eval runs; document
  facts are kept as written and guarded by the drift check.
- **Metadata does not help search.** The sections stored with each chunk are
  used after retrieval (citations, the sources line, traces), never to rank.

## Stage 5: reranking (18 Oct)

**Built.** The search fetches 20 chunks; a small cross-encoder on the machine
reads the question and each chunk together, scores how well each answers it,
and the best 7 go to the model. Switchable, and traced as its own step.

**Tried first, for free, on the retrieval score:** reranking alone, section
headings written into the embedded text, and both. Headings alone did nothing
for the first search (the embedder averages the whole chunk); with reranking
they reached 27/29, because the reranker reads them.

**Result.** RAG 54 → 56/60; answer chunk in the top 7: 24 → 26/29. Cost the
same ($4.95): the model still receives 7 chunks.

**Findings.**
- It fixed four near misses (ranks 8–13) and broke two: a list spread over
  several chunks, and the slang question "u guys open sat?".
- A larger reranker scored the same at five times the time; keeping 5 chunks
  instead of 7 lost accuracy.
- Three failures were our checks, not the bot: "call us" failing a check for
  the word "phone", an ambiguous `covered` for "we don't offer Botox", and a
  judge asked to confirm a package was real without seeing the document. The
  cases were fixed.
- Running unchanged code twice varied by about one case: one run is a
  measurement with noise.

## Stage 6: LangChain (18 Oct, in progress)

**Built.** The same pipeline with LangChain parts, next to the hand-built one
and chosen by `SPADESK_RAG_ENGINE`: a `TokenTextSplitter`, an `Embeddings` class
for the same model, any LangChain `VectorStore` (Chroma, or Pinecone), a
`BaseRetriever` with a reranking compressor, a `ChatPromptTemplate` and a chat
model replying in the same structured shape. Every provider is a setting.
Tests prove both pipelines make the same 33 chunks, retrieve the same chunks
in the same order, and build the same prompt.

**Findings.**
- `langchain-community` is being retired, so only `langchain-core` and the
  standalone integration packages are used.
- LangChain did not remove the decisions: index freshness, the
  fetch-20-keep-7 reranking and the citation checks had to be written for it
  too. Its `add_start_index` gives -1 with a token splitter; its compression
  retriever cannot fetch more candidates than it keeps.
- A LangChain compressor is not traced on its own; the search and the
  reranking were made steps of their own so a trace shows both lists.

**Relevance.** "Can I bring my kid with me?" got a correct list of kids'
services that never answered whether a child can come along. A prompt rule
("answer the question that was asked; if only related information exists, say
so first") and a judged case now target this. The rule is kept general: an
example taken from the eval's own question would have inflated the score.

**Result.** With the customer chat on LangChain and the new rule, the whole
document scored 61/61 and RAG 57/61 (93%). The kid question passed in both
modes. Three RAG failures were the known retrieval misses; the fourth was a
judge criterion too strict for an honest "I couldn't find that" about Botox.
The run changed two things at once, so the gains most likely come from the
rule rather than the framework, which by design behaves the same.

**The owner chat on LangGraph.** Its report tool needs a loop: Claude asks for
a report, the app runs it, Claude answers, and Claude decides how many times
(none for a profile question, one for "last month"). A chain runs once, so
the owner chat became a graph (`spadesk/lc/owner/`): `prepare` (search and
prompt), `agent` (one Claude call), `tools` (run the report), and a choice
after `agent` between `tools` and the end, at most 4 rounds. It uses the same
prompt, tool and reply shape as the hand-built loop; the same setting picks
it. Not yet measured with a paid eval run.

## Findings that run through every stage

1. **Measure before and after every change**, on the same questions; a free
   retrieval score first, a paid run to confirm.
2. **Check the checks.** Several "failures" were the eval's own mistakes.
3. **Faithful is not the same as relevant.** An answer can be true and still
   not answer the question.
4. **Design the refusal first.** A miss that says "I could not find that" is
   worth more than a confident guess.
5. **Keep each provider behind an interface.** Swapping Chroma for Pinecone, or
   the hand-built pipeline for LangChain, is a setting, not a rewrite.

## Roadmap

- Section headings in the embedded chunks (27/29 in the experiment).
- One shared flow for both assistants; when tool rounds run out, one last call without tools instead of an error.
- Agentic RAG (search as a tool), hybrid keyword and vector search.
- Conversation memory for follow-up questions; escalation to the front desk when an answer is not covered.
- The document in S3, refreshed by its ETag.
- LangSmith experiments and a comparison dashboard.
- A login for the manage page, then hosting.
