# spaDesk

spaDesk is a small web app for a spa, built around a fictional business called
Pebble Spa. It has two sides:

- **A customer site** where visitors browse services, prices, opening hours
  and policies, and send a booking request.
- **A manage page** for the owner, with two tabs: the business profile and a
  reports dashboard covering a year of bookings (revenue, traffic, channels
  and customer feedback, filtered by date range, category and channel). An
  assistant chat window is available on both.

It is a small, readable codebase: Python (FastAPI) on the server, plain HTML,
CSS and JavaScript in the browser, no build step.

## What it does

### Customer site (`/`)

- **Landing page**: services, prices, opening hours, policies and contact
  details, all read from the business document. The page itself is static; a
  script fills it in from `/api/spa`.
- **Booking request**: a form that saves a request with status `requested`.
  Nothing checks availability or confirms an appointment, and the page says so.

### Manage page (`/manage`)

The page has two tabs, switched without reloading; the address keeps the
active one (`/manage#reports`).

**Business profile tab** (read-only): the whole business document laid out
section by section: about, mission, contact and opening hours, services with
duration and price per category, packages, offers, add-ons, FAQs and policies.

**Reports tab**:

- **Filters**: date presets, a custom date range, category and channel. Every
  number and chart follows the filters.
- **Headline tiles**: revenue, bookings, completion rate, average ticket,
  average rating, and revenue lost to cancellations and no-shows, each with its
  change against the previous period.
- **Charts**: revenue trend and traffic (bookings) trend as vertical bars;
  revenue by category, top services, revenue by channel and new against returning
  customers as horizontal bars; and "How customers feel", a face for each
  star rating with the feedback tags behind each sentiment. Every chart has
  a table view.
- **Assistant**: a button that opens a chat window, available on both tabs. A
  question is sent to `POST /api/report/chat`, which gives Claude the business
  document and the current time at the spa. It answers questions about the
  business profile (services, prices, hours, policies). The reply is
  structured output: the answer, a list of entries when the answer is a
  list, whether the profile covered the question, and which sections it came
  from, and a few follow-up questions. So the chat lays lists out as bullets,
  shows a "From: Policies" line, marks answers the profile could not cover,
  and offers the follow-ups as buttons. When first opened, the assistant
  introduces itself and suggests some questions to start from. It does not
  remember earlier messages.
- **Report questions (tool use)**: the assistant can also answer about the
  reports ("How did we do last month?"). The model is offered one tool,
  `get_report`. When a question needs figures it asks the server to run it
  for the dates it has worked out, the server runs the same code that fills
  the Reports tab, and the model answers from those figures. The line under
  the answer names the period it looked up. See `spadesk/reports/tools.py`.
- **RAG switch**: a switch in the chat header chooses how the assistant is
  given the business document. Off, the whole document is sent. On, only the
  few chunks that best match the question are sent (retrieval-augmented
  generation). The line under each answer says which was used, so the same
  question can be asked both ways and compared. See
  [docs/rag-notes.md](docs/rag-notes.md).
- **Tracing** (optional): with LangSmith switched on in `.env`, every
  question is recorded as a trace: the chunks retrieved, the exact prompt
  sent to the model, its reply, the tokens used and the time each step took.
  See `spadesk/tracing.py`.

## Running it

See [RUNNING.md](RUNNING.md) for install, start, test and troubleshooting steps.
The short version:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY
.venv/bin/uvicorn spadesk.app:app --reload
```

Then open http://127.0.0.1:8000 (customer site) or
http://127.0.0.1:8000/manage (manage page). Only the
assistant needs the API key; everything else works without one.

`.env` holds your API key and is never committed (it is listed in
`.gitignore`). `.env.example` is the committed template: it lists the same
settings with the key left blank.

## Where the data comes from

| File | What it is | Read by |
|---|---|---|
| `data/business_doc.md` | What the business is: name, services, prices, hours, policies, FAQs, offers | Customer site, business profile tab, assistant, and the reports (for service names, categories and prices) |
| `data/spadesk_bookings.json` | What happened: one year of mock bookings, the customers who made them, and the channels they came through | Reports |
| `storage/booking_requests.jsonl` | Booking requests sent from the site (created on first use) | Nothing yet |

Each fact is stored once. The bookings file holds no business details: a
booking names its service by id, and the service's name, category and price
are looked up in the business document when the app runs
(`spadesk/business/catalog.py`). So the Reports tab and the Business profile
tab cannot disagree.

`scripts/generate_bookings.py` writes the bookings JSON. It reads the business
document for the services, opening hours and offers a realistic booking
depends on, and invents the bookings around them. It is run by hand. Run it
again after renaming, adding or removing a service in the document; until
then the app refuses to start the reports and says which booking no longer
matches.

No service, price or label is written in application code.

## How it works

### The customer site

1. The browser loads `static/home/index.html`, an empty shell, and `static/home/js/main.js`.
2. `main.js` requests `/api/spa`.
3. The server parses the business document once (`business/knowledge.py`) and
   returns the name, introduction, hours, services and contact details.
4. The scripts fill the page and wire up the booking form.

### A dashboard view

1. The page asks `/api/report/options` what can be filtered and builds the
   filter row.
2. On every filter change it requests `/api/report` with the chosen filters.
3. The server filters the bookings, looks up each booking's service in the
   business document, runs each calculation in `spadesk/reports/metrics/`,
   and returns the results as one JSON document.
4. Each file in `static/dashboard/js/sections/` draws one part of it.

The browser never reads the data file; filtering and maths happen on the server.

### A dashboard assistant question

1. `js/chat/widget.js` shows the question and calls `askAssistant`.
2. `js/chat/assistant.js` posts it to `/api/report/chat`.
3. The route in `reports/routes.py` builds the system prompt with
   `assistant/prompt.py` (rules, current time, the business document), sends
   it to Claude with the question through `assistant/model.py` (using the API
   key from `.env`), and asks for the reply in a fixed shape, the `ChatReply`
   class.
   With the RAG switch on, step 3 sends only the retrieved chunks instead:
   `rag/retriever.py` embeds the question and finds the closest chunks in the
   vector store, and `rag/context.py` lays them out for the prompt.
   If the model replies with a request for the `get_report` tool instead of
   an answer, the route runs it (`reports/tools.py`), sends the figures back
   and asks again, until the model answers.
4. The window shows the answer and uses the other fields to add the source
   line or the "not covered" style.

The model is called from the server so the API key never reaches the browser.

## Layout

```
data/                          business document and bookings JSON
scripts/
  generate_bookings.py         writes the bookings JSON
spadesk/
  config.py                    settings; which parts of the document the site features
  tracing.py                   what each step of an assistant question records in LangSmith
  app.py                       the FastAPI app and the customer site's routes
  business/
    knowledge.py               reads the business document
    store.py                   the one shared parsed copy (`get_knowledge`)
    catalog.py                 the bookable services, as a catalogue with ids
    profile.py                 lays it out as the manage page's business profile
  booking/
    service.py                 mock booking requests
  rag/
    chunking.py                cuts the document into fixed-size chunks with overlap
    embeddings/                turns text into vectors; the provider is a setting
    stores/                    keeps chunks and vectors (Chroma); the store is a setting
    index.py                   builds the index, and rebuilds it when something changed
    retriever.py               finds the chunks that match a question
    context.py                 lays the retrieved chunks out for the model
  reports/
    dataset.py                 loads the bookings JSON, joins it to the services, filters it
    metrics/                   one file per calculation
    report.py                  assembles the full report
    tools.py                   the `get_report` tool the assistant can ask for
    routes.py                  /manage, /api/report and the owner's assistant route
  assistant/
    prompt.py                  the system prompt both assistants share (profile rules, time, profile)
    model.py                   the traced model call
    replies.py                 the question body and one reply list entry
    customer.py                the customer assistant: /api/chat, profile only, no tools
  static/
    home/                      the customer site: index.html, css/ and js/, one file per area
    dashboard/
      index.html               the dashboard shell
      css/                     one stylesheet per area
      js/
        main.js                entry point
        tabs.js                switches between the two tabs
        profile.js             draws the business profile tab
        api.js, filters.js, presets.js, tiles.js, card.js, table.js, format.js, dom.js
        charts/                column and bar list charts, and their helpers
        sections/              one file per dashboard card
        chat/                  the assistant window and the call to the server
tests/                         one test file per area
```

Every file opens with a note on what it does. Python functions have
docstrings and JavaScript functions have JSDoc blocks describing their
parameters and return values.

## Reading the code

To follow one dashboard view:

1. `spadesk/static/dashboard/js/main.js`: loads options, fetches the report, hands it to each section.
2. `spadesk/static/dashboard/js/api.js`: the two requests the page makes.
3. `spadesk/reports/routes.py`: the routes that answer them.
4. `spadesk/reports/report.py`: which calculations make up a report.
5. `spadesk/reports/dataset.py`: loading and filtering the JSON.
6. Any file in `spadesk/reports/metrics/`: one calculation.
7. Any file in `spadesk/static/dashboard/js/sections/`: one card drawn from the report.

To follow the customer site: `spadesk/static/home/js/main.js`, then the `/api/spa`
route in `spadesk/app.py`, then `spadesk/business/knowledge.py`.

## Limits

- Booking is a mock: requests are stored, never confirmed.
- The bookings data is generated, not real, and ends on a fixed date, so the
  dashboard's "last 30 days" means the last 30 days of the data.
- Nothing has a login. Put the dashboard behind authentication before using it
  with real data.
- Each assistant question stands alone (no memory).
- With RAG on, questions that need the whole document ("which service is
  cheapest?") can be answered wrongly, because only a few chunks are sent.

## Where later work plugs in

| Next step | Where |
|---|---|
| Another tool for the assistant | a description and a function in `reports/tools.py`, and one more case in the `chat` route's tool loop |
| Follow-up questions (memory) | the same route, keeping earlier messages per session |
| Another embedding provider or vector store | a file in `rag/embeddings/` or `rag/stores/`, and one line in its `__init__.py` |
| A new dashboard chart | a metric in `reports/metrics/`, a line in `report.py`, a file in `js/sections/` |
| Real calendar booking | `BookingDesk.submit` in `booking/service.py` |
