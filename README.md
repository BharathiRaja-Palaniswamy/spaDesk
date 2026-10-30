# spaDesk

spaDesk is a small web app for a spa, built around a fictional business called
Pebble Spa. It has two sides:

- **A customer site** where visitors browse services, prices, opening hours
  and policies, and book an appointment in a free time.
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
- **Booking**: a form that asks for the customer's name and phone, a service
  and a date, shows the free times by therapist, and books the chosen one,
  with a reference to keep. The free times come from the business document
  (who offers what, on which days, opening hours, classes taught) and the
  therapists' Google calendars (see [docs/google-calendar.md](docs/google-calendar.md)). If Google
  cannot be reached, booking says it is unavailable; bookings are never kept anywhere else.
- **Pebby, the chat and the phone line**: the front AI agent (`spadesk/agent/`,
  a LangGraph graph with memory per conversation). It asks for the customer's
  name and phone first, answers from the guide, and books, changes or cancels
  (packages and add-ons too), but only after the customer says yes to exactly
  what will happen; the booking itself is code. Feedback goes to the
  Feedback agent, which replies and files a card the owner closes on /manage.
  When it cannot help (a person asked for, more than five services, a
  recurring booking, repeated failures) it hands the customer to the staff:
  the request is written to `storage/handoffs.jsonl`.

### Manage page (`/manage`)

For the owner only: it opens after signing in at `/login` with the owner's
password (`SPADESK_OWNER_PASSWORD`). Who may do what is one table in
`spadesk/access/permissions.py`: customers may search the guide, ask for the
staff and book under their own number; only the owner sees the reports, the
day's bookings and the phone-line test. Every owner route checks it (401
without the sign-in), and Pebby is only ever given the tools a customer may
use.

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
- **RAG**: both chats give the assistant only the few chunks of the business
  document that best match the question (retrieval-augmented generation),
  not the whole document. The evals still ask both ways, to compare. See
  [docs/rag-notes.md](docs/rag-notes.md).
- **Tracing** (optional): with LangSmith switched on in `.env`, every
  question is recorded as a trace: the chunks retrieved, the exact prompt
  sent to the model, its reply, the tokens used and the time each step took.
  See `spadesk/tracing.py`.
- **Evals**: 61 questions asked to the real model with the whole document
  and with RAG, graded by code checks and, for yes/no and safety answers, by
  a second model call. See `evals/` and
  [docs/eval-report.md](docs/eval-report.md).
- **LangChain pipeline** (the default): the RAG steps with LangChain in
  `spadesk/lc/`, every provider pluggable. The owner chat's tool loop is a
  LangGraph graph (`spadesk/lc/owner/`). The hand-built pipeline (`spadesk/rag/`,
  `assistant/model.py`) is kept as the reference; `SPADESK_RAG_ENGINE=handbuilt`
  switches to it.
- **Model gateway**: every model call of the LangChain pipeline, the agent and
  the evals goes through `spadesk/models/`. One setting, `SPADESK_MODELS`,
  picks a set of models: `claude` (Opus and Haiku) or `opensource` (two Qwen3
  models hosted by Nebius). Each job asks for a tier, high-end for the chats
  and Pebby, low-end for quick jobs, so switching sets keeps a
  model of the right size for each. The evals always answer with
  `opensource` (if open models do well, Claude should too); their judge stays
  on Claude. The gateway also logs the tokens and time of each call.

How the project got here, stage by stage, and what each stage taught us:
[docs/evolution.md](docs/evolution.md).

A one-page project brief, as a web page: [docs/index.html](docs/index.html) (this week: booking by chat
and phone, and the agents), published with GitHub Pages at https://bharathiraja-palaniswamy.github.io/spaDesk/.
Last week's brief (answering questions) is [docs/week-2.html](docs/week-2.html).

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
| `storage/handoffs.jsonl` | Customers the chat handed to the staff, with why (created on first use) | The staff |
| `data/spadesk_bookings.json` | What happened: one year of mock bookings, the customers who made them, and the channels they came through | Reports |

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
3. The route in `reports/routes.py` runs the owner's LangGraph graph
   (`lc/owner/`): it retrieves the chunks that best match the question
   (`lc/retriever.py`: search, then a reranker keeps the best), builds the
   prompt (rules, current time, those chunks), and asks the model gateway
   (`spadesk/models/`) for the reply in a fixed shape, the `ChatReply` class,
   with the `get_report` tool on offer.
   If the model asks for the tool instead of answering, the graph runs it
   (`reports/tools.py`), sends the figures back and asks again, until the
   model answers. (The hand-built engine does the same in a plain loop in
   `reports/routes.py`, calling Claude through `assistant/model.py`.)
4. The window shows the answer and uses the other fields to add the source
   line or the "not covered" style.

The model is called from the server so the API key never reaches the browser.

## Layout

```
data/                          business document and bookings JSON
scripts/
  generate_bookings.py         writes the bookings JSON
  setup_calendars.py           creates the therapists' Google calendars and shares them
  check_calendar.py            books, reads back and cancels a test appointment
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
    schedule.py                what can be booked, with whom, when: read from the business document
    calendars/                 where bookings are kept, one calendar per therapist: in memory or Google Calendar
    appointments.py            the booking rules: free times, book, cancel, the day's bookings
    actions.py                 those rules in the shape a model's tool request takes
    routes.py                  those rules as the booking form's routes
  rag/
    chunking.py                cuts the document into fixed-size chunks with overlap
    headers.py                 writes each chunk's place in the document above it, for search only
    embeddings/                turns text into vectors; the provider is a setting
    stores/                    keeps chunks and vectors (in memory); the store is a setting
    index.py                   builds the index, and rebuilds it when something changed
    retriever.py               finds the chunks that match a question
    rerankers/                 reorders them with a small local model and keeps the best; a setting
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
    customer.py                the question-and-answer assistant: /api/chat, no tools or memory (kept for the evals)
    engine.py                  which pipeline answers both chats: LangChain (default) or hand-built
  voice/                       the phone line: Pebby behind /v1/chat/completions, which ElevenLabs calls
  access/                      who may do what: roles and permissions, the owner's sign-in, /login
  agent/                       the customer chat: Pebby, the Feedback agent, and booking in code; /api/assistant,
                               /api/feedback (the owner's feedback cards)
    rules.py                   how it behaves: six principles, and their settings
    flow.py                    a turn, step by step
    pebby.py                   Pebby: instructions, tools, name and phone first
    details.py, yes.py         the customer's own words -> details; the questions for their yes
    booking.py, executor.py    what can be booked; booking it all or none
    feedback.py                the Feedback agent: reads feedback, answers it, files a card the owner closes
    staff.py                   handoffs: what is written down for the staff
    debug.py                   debug mode: every step of a turn, under the reply
  lc/                          the same pipeline with LangChain; each provider a setting
    owner/                     the owner chat as a LangGraph graph: state, nodes, graph
  models/                      the model gateway: jobs -> models, providers, tools, fixed-shape replies, fallback, usage
  static/
    home/                      the customer site: index.html, css/ and js/, one file per area
    login/                     the owner's sign-in page
    shared/                    scripts both pages use (the "Powered by" line)
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
evals/                         cases played on the real models and graded (python -m evals.run)
  golden.json                  every case, for Pebby and both chats, with its label (what must happen)
  golden.py                    reads the cases, by target and split
  pebby.py                     plays a Pebby conversation; routing, extraction and outcome checks
  chats.py                     asks a guide or owner question through the app's own route
  checks.py                    the answer checks every target shares
  judge.py                     a second model call that grades an answer against written criteria
  figures.py                   owner figures read from the bookings data at run time, so they never go stale
  validate.py                  stops a run when a case no longer matches the guide, schedule or data
  run.py                       plays every case, prints each result and the scores by layer and split
  observe.py                   traces each case to the spadesk-evals LangSmith project
  summary.py                   pass rates per layer and split, time and tokens
  datasets.py                  copies golden.json into LangSmith datasets, one per target
  experiment.py                runs the cases as a LangSmith experiment, one score per layer
  annotation.py                the review rubric and queues, and what goes in them
  retrieval.py                 retrieval alone, no model calls: is the chunk holding each answer in the top k?
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

- Booking is for one person and one service at a time. Group classes,
  packages and cancelling from the site are not bookable yet (cancelling is
  a tool, for the chat). Phone numbers are not verified.
- The bookings data is generated, not real, and ends on a fixed date, so the
  dashboard's "last 30 days" means the last 30 days of the data.
- The owner is one password, not separate staff accounts. Customers are not
  signed in: they are identified only by the name and number they give.
- The owner's assistant (manage page) takes each question on its own, with
  no memory; Pebby, the customer chat, remembers its conversation.
- With RAG on, questions that need the whole document ("which service is
  cheapest?") can be answered wrongly, because only a few chunks are sent.

## Where later work plugs in

| Next step | Where |
|---|---|
| Another tool for the assistant | a description and a function in `reports/tools.py`, and one more case in the `chat` route's tool loop |
| Follow-up questions (memory) | the same route, keeping earlier messages per session |
| Another embedding provider, vector store or reranker | a file in `rag/embeddings/`, `rag/stores/` or `rag/rerankers/`, and one line in its `__init__.py` |
| A new dashboard chart | a metric in `reports/metrics/`, a line in `report.py`, a file in `js/sections/` |
| Another calendar (e.g. Outlook) | a file in `booking/calendars/` with the methods in `base.py`, and one line in its `__init__.py` |
