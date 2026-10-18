# RAG in spaDesk: how it works and what we learned

RAG (retrieval-augmented generation) means giving the model only the parts of
a document that match the question, rather than the whole document. In
spaDesk it is an option: the switch in the assistant's chat header turns it
on and off, so both ways can be compared on the same question.

## The pipeline

Done once, and again whenever the document or a setting changes:

```
business_doc.md -> chunks -> one vector per chunk -> vector store
```

Done for every question:

```
question -> its vector -> closest 20 chunks -> reranker keeps the best 7 -> prompt -> model -> answer
```

| Step | File | What it does |
|---|---|---|
| 1. Chunking | `spadesk/rag/chunking.py` | Cuts the document every 150 tokens, each chunk repeating the last 30 tokens of the one before |
| 2. Embeddings | `spadesk/rag/embeddings/` | Turns text into a vector of 384 numbers that stands for its meaning |
| 3. Vector store | `spadesk/rag/stores/` | Saves chunks with their vectors in Chroma (`storage/chroma`) |
| 3. Index | `spadesk/rag/index.py` | Builds the store from the document; rebuilds when something changed |
| 4. Retrieval | `spadesk/rag/retriever.py` | Embeds the question and returns the closest chunks with scores |
| 4b. Reranking | `spadesk/rag/rerankers/` | A small local model reads the question and each of the 20 closest chunks together, scores how well each answers it, and the best 7 are kept (`SPADESK_RERANKER`, `SPADESK_RERANK_CANDIDATES`; "none" switches it off) |
| 5. Context | `spadesk/rag/context.py` | Lays the chunks out between `<excerpt>` tags for the prompt |
| 5. Prompt | `spadesk/assistant/prompt.py` | Puts the excerpts, or the whole document, into the system prompt, depending on `rag` in the request. Both chat routes use it: `assistant/customer.py` (customer site) and `reports/routes.py` (owner) |
| 5. Switch | `static/home/js/chat/rag-switch.js`, `static/dashboard/js/chat/rag-switch.js` | The "RAG on / RAG off" switch on each page and the label under each answer |

To try a step alone:

```sh
.venv/bin/python -c "
from spadesk.rag.retriever import retrieve
for m in retrieve('What is the cancellation policy?'):
    print(round(m.score, 2), m.chunk.id, m.chunk.sections)
"
```

## Terms

- **Chunk**: one piece of the document. Ours are a fixed number of tokens.
- **Token**: a piece of a word. 100 English words are about 130 tokens.
- **Overlap**: the tokens a chunk shares with the one before it, so text
  around a cut is whole in at least one of the two.
- **Embedding / vector**: a list of numbers standing for the meaning of a
  text. Texts with a similar meaning get similar vectors.
- **Dimension**: how many numbers are in each vector (384 for our model). It
  is fixed by the embedding model.
- **Cosine similarity**: the score for how alike two vectors are. Near 1 is
  the same meaning, near 0 unrelated.
- **Top-k**: how many chunks retrieval returns (`SPADESK_RETRIEVAL_TOP_K`).
- **Index**: the stored chunks and vectors, ready to search.

## What we learned

**1. RAG does not make answers better on a small document.** The document is
about 1,700 tokens and fits in the prompt many times over. Sent whole, the
model sees everything. RAG can only match that, or do worse when retrieval
misses. RAG earns its place when the knowledge is too large to send.

**2. Lookup questions work; aggregate questions do not.** "What is the
cancellation policy?" has an answer in one place, and retrieval finds it. "Which
service is cheapest?" needs every price. With 3 chunks the model answered
$35 (wrong); the chunk with Lip Wax at $18 ranked 7th. It became right at
top-k 7, which is nearly half the document. The proper fix for such
questions is a tool that returns the full price list, not a bigger top-k.

**3. Retrieval always returns something.** Asked about bridal makeup, which
the document does not mention, it still returned three chunks. Only the low
scores hint that nothing matched.

**4. Tell the model it is seeing excerpts.** With that one instruction added,
the model said "I was only given parts of the profile, so I can't confirm
there isn't a cheaper service". Without it, a wrong answer would have sounded
certain.

**5. Fixed-size chunks cut through meaning.** One cut landed inside
"Price: $130 / $180". The overlap kept the price whole in the next chunk,
but that chunk had lost the service's name. A chunk also spans several
headings, so citations are broader: one chunk covers five policies. Cutting
by the document's structure (one service per chunk) avoids this when a
document has clear structure.

**6. Embedding models have a limit per text.** Ours reads 256 tokens and
silently ignores the rest. The index refuses to build if the chunk size is
larger than the model can read.

**7. Change the embedding model, rebuild the index.** Vectors from different
models cannot be compared. The index stores a signature (a hash of the
document, the chunk settings and the model's name) and is rebuilt when it
no longer matches.

**8. Wording differences are not RAG differences.** The model phrases each
answer a little differently on every call. Compare the facts, not the words.

**9. Keep providers behind a small interface.** The embedder and the vector
store are each one interface, one file per provider, and one setting. The
rest of the app never imports Chroma directly, so swapping is a change to
`.env`.

**10. `.env` is read at start-up.** A changed setting needs a server restart.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `SPADESK_CHUNK_TOKENS` | 150 | Tokens per chunk |
| `SPADESK_CHUNK_OVERLAP` | 30 | Tokens shared with the chunk before |
| `SPADESK_EMBEDDING_PROVIDER` | chroma | Which model makes the vectors |
| `SPADESK_VECTOR_STORE` | chroma | Where chunks and vectors are kept |
| `SPADESK_VECTOR_STORE_PATH` | storage/chroma | The folder a local store saves to |
| `SPADESK_RETRIEVAL_TOP_K` | 3 | Chunks retrieved per question |

## Not done yet

- An eval: a fixed set of questions with expected answers, scored for both
  modes, so a change to chunk size or top-k can be measured.
- A tool for aggregate questions such as "cheapest" or "list every service".
- Hosted embedding providers and other vector stores.
