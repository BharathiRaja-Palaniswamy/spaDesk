# Running spaDesk

All commands are run from the project folder (`spadesk/`).

## 1. Install (once)

You need Python 3.10 or newer.

```sh
python3 -m venv .venv                          # a private Python environment for this project
.venv/bin/pip install -r requirements.txt      # install the libraries into it
```

## 2. Add your API key (once)

```sh
cp .env.example .env
```

Open `.env` and put your Anthropic API key after `ANTHROPIC_API_KEY=`.
You can create a key at https://console.anthropic.com. `.env` is listed in
`.gitignore`, so the key is never committed.

The key is only used by the assistant chat on the dashboard. Without one,
both pages still load and every chart works; the assistant replies "Sorry,
something went wrong".

Never commit `.env`. To share the project, commit `.env.example` (the same
settings with the key left blank); anyone running it makes their own `.env`.

## 3. Start the app

```sh
.venv/bin/uvicorn spadesk.app:app --reload
```

Open http://127.0.0.1:8000 for the customer site, or
http://127.0.0.1:8000/manage for the owner's manage page.

- `--reload` restarts the server whenever you save a Python file. Leave it out
  if you don't want that.
- Stop the server with `Ctrl+C`.
- To use another port: add `--port 8080`.

## 4. Try it

- On the customer site, click **Request a booking**, fill in the form and send it. The request is
  appended to `storage/booking_requests.jsonl`.
- Open the manage page, go to the **Reports** tab, switch between the date presets, pick a category or
  channel, and use **Show table** on a chart.
- On the manage page, click **Ask assistant** and ask about the business, for
  example "What is our cancellation policy?" (needs the API key).

## 5. Run the tests

```sh
.venv/bin/python -m pytest
```

The tests do not need an API key and do not call Claude.

## 6. Run the evals

The evals ask the real model the questions in `evals/cases.json` and grade
the answers, with the whole document and with RAG. They need the API key and
cost a few dollars per full run.

```sh
.venv/bin/python -m evals.retrieval   # retrieval only: free, a few seconds
.venv/bin/python -m evals.run --label rerank   # every case, both modes: about 15 minutes
```

`--label` names the run in LangSmith (tag `run:<label>`), so two runs can be
compared there. Set `SPADESK_RERANKER=none` to measure without reranking.

The results so far, and what changed between runs, are in
[docs/eval-report.md](docs/eval-report.md).

## Changing things

| To change | Edit | Then |
|---|---|---|
| Prices, hours, policies, descriptions | `data/business_doc.md` | restart the server |
| Add, rename or remove a service | `data/business_doc.md` | regenerate the bookings (below), then restart |
| The mock bookings (volume, mix, period) | the constants at the top of `scripts/generate_bookings.py` | run `.venv/bin/python scripts/generate_bookings.py`, then restart |
| What the dashboard assistant sends to the model | the `chat` route in `spadesk/reports/routes.py` | saved automatically with `--reload` |
| Featured services, highlighted policies | `spadesk/config.py` | restart |
| Which model the assistant uses | `.env` | restart |
| Customer site text, layout, colours | `spadesk/static/` | refresh the browser |
| Dashboard tiles, charts, colours | `spadesk/static/dashboard/` | refresh the browser |

## Settings in `.env`

`.env` is read once, when the server starts. `--reload` only watches Python
files, so after changing `.env`, stop the server and start it again.

| Name | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Your API key. Required for the dashboard assistant. |
| `SPADESK_MODEL` | `claude-opus-5-5` | Which Claude model the assistant uses. |
| `LANGSMITH_TRACING` | off | Set to `true` to record each assistant question in LangSmith. |
| `LANGSMITH_API_KEY` | none | Your LangSmith key. Needed only when tracing is on. |
| `LANGSMITH_PROJECT` | `default` | The LangSmith project the traces are filed under, e.g. `spadesk`. |
| `SPADESK_TIMEZONE` | `America/Los_Angeles` | The spa's time zone, so the assistant knows the current time there. |
| `SPADESK_DATA` | `data/business_doc.md` | The business document. |
| `SPADESK_REPORT_DATA` | `data/spadesk_bookings.json` | The bookings file the dashboard reads. |
| `SPADESK_CHUNK_TOKENS` | `150` | RAG: how many tokens each chunk of the document holds. |
| `SPADESK_CHUNK_OVERLAP` | `30` | RAG: how many tokens a chunk shares with the one before it. |
| `SPADESK_EMBEDDING_PROVIDER` | `pinecone` | RAG: which provider turns text into vectors: `pinecone` (llama-text-embed-v2 on Pinecone's servers, needs `PINECONE_API_KEY`) or `minilm` (on this machine, downloaded once). |
| `SPADESK_RERANKER` | `pinecone` | RAG: which reranker reorders the candidates: `pinecone` (bge-reranker-v2-m3 on Pinecone's servers), `cross-encoder` (on this machine) or `none`. |
| `SPADESK_VECTOR_STORE` | `pinecone` | RAG: which vector store keeps the chunks and their vectors: `pinecone` (needs `PINECONE_API_KEY`) or `memory` (offline). |
| `SPADESK_VECTOR_STORE_PATH` | `storage/vectors` | RAG: names a local index (tests give each its own). |
| `SPADESK_RETRIEVAL_TOP_K` | `3` | RAG: how many chunks are fetched for a question. |
| `SPADESK_CALENDAR` | `memory` | Booking: where bookings are kept: `memory` (lost on restart) or `google` (one Google calendar per therapist; see [docs/google-calendar.md](docs/google-calendar.md)). |
| `GOOGLE_SERVICE_ACCOUNT` | none | Booking with `google`: the service account's key file, as a path or its JSON text. |
| `SPADESK_CALENDAR_OWNER` | none | Booking with `google`: a Google account the calendars are shared with, to see the bookings. Optional. |
| `SPADESK_SLOT_MINUTES` | `30` | Booking: appointments start every this many minutes from opening. |

## If something goes wrong

- **The assistant says "Sorry, something went wrong"**: the API key is missing
  or wrong, or the API could not be reached. The terminal running the server
  prints the error for the `/api/report/chat` request.
- **The page still behaves the old way after editing JavaScript**: the browser
  cached the old file. Hard refresh with Cmd+Shift+R, or keep DevTools open
  with "Disable cache" ticked.
- **`Address already in use`**: another server is on port 8000. Stop it, or
  start this one with `--port 8001`.
- **The customer site looks out of date after editing the document**: restart
  the server; the document is read once at startup.
- **The Reports tab shows an error instead of numbers**: the bookings file is
  missing, or a booking points at a service that is no longer in the business
  document (it was renamed or removed). The server terminal names the booking.
  Regenerate the bookings with `.venv/bin/python scripts/generate_bookings.py`.
- **`command not found: uvicorn`**: use `.venv/bin/uvicorn`, or run step 1 again.
