# spaDesk

spaDesk is a small web app for a spa, built around a fictional business called
Family Spa. It has two sides:

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
- **Charts**: revenue trend and traffic (bookings) trend as lines; revenue
  by category, top services, revenue by channel and new against returning
  customers as horizontal bars; and "How customers feel", a face for each
  star rating with the feedback tags behind each sentiment. Every chart has
  a table view.
- **Assistant**: a button that opens a chat window, available on both tabs. A
  question is sent to `POST /api/report/chat`, which gives Claude the business
  document and returns its reply. It answers questions about the business
  profile (services, prices, hours, policies) and says so when the document
  does not cover something. It does not see the reports yet and does not
  remember earlier messages.

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
| `data/family_spa_business_doc.md` | The business in words: services, prices, hours, policies, FAQs, offers | Customer site and the business profile tab |
| `data/spadesk_bookings.json` | One year of mock bookings, with services, customers, offers and channels as separate lists | Dashboard |
| `storage/booking_requests.jsonl` | Booking requests sent from the site (created on first use) | Nothing yet |

`scripts/generate_bookings.py` writes the bookings JSON. It copies services,
prices, hours and offers from the business document, so the dashboard and the
customer site agree, and invents the bookings around them. It is run by hand;
after editing the business document, run it again.

No service, price or label is written in application code. The dashboard reads
only the JSON; the customer site reads only the business document.

## How it works

### The customer site

1. The browser loads `static/index.html`, an empty shell, and `static/app.js`.
2. `app.js` requests `/api/spa`.
3. The server parses the business document once (`business/knowledge.py`) and
   returns the name, hours, services, policy highlights and contact details.
4. `app.js` fills the page and wires up the booking form.

### A dashboard view

1. The page asks `/api/report/options` what can be filtered and builds the
   filter row.
2. On every filter change it requests `/api/report` with the chosen filters.
3. The server filters the bookings, runs each calculation in
   `spadesk/reports/metrics/`, and returns the results as one JSON document.
4. Each file in `static/dashboard/js/sections/` draws one part of it.

The browser never reads the data file; filtering and maths happen on the server.

### A dashboard assistant question

1. `js/chat/widget.js` shows the question and calls `askAssistant`.
2. `js/chat/assistant.js` posts it to `/api/report/chat`.
3. The route in `reports/routes.py` reads the business document, sends it to
   Claude as context together with the question (using the API key from
   `.env`), and returns the reply text. The document is small enough to send
   whole, so no search step is needed.

The model is called from the server so the API key never reaches the browser.

## Layout

```
data/                          business document and bookings JSON
scripts/
  generate_bookings.py         writes the bookings JSON
spadesk/
  config.py                    settings; which parts of the document the site features
  app.py                       the FastAPI app and the customer site's routes
  business/
    knowledge.py               reads the business document
    store.py                   the one shared parsed copy (`get_knowledge`)
    profile.py                 lays it out as the manage page's business profile
  booking/
    service.py                 mock booking requests
  reports/
    dataset.py                 loads and filters the bookings JSON
    metrics/                   one file per calculation
    report.py                  assembles the full report
    routes.py                  /manage, /api/report and the assistant route
  static/
    index.html, styles.css, app.js    the customer site
    dashboard/
      index.html               the dashboard shell
      css/                     one stylesheet per area
      js/
        main.js                entry point
        tabs.js                switches between the two tabs
        profile.js             draws the business profile tab
        api.js, filters.js, presets.js, tiles.js, card.js, table.js, format.js, dom.js
        charts/                line and bar list charts, and their helpers
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

To follow the customer site: `spadesk/static/app.js`, then the `/api/spa`
route in `spadesk/app.py`, then `spadesk/business/knowledge.py`.

## Limits

- Booking is a mock: requests are stored, never confirmed.
- The bookings data is generated, not real, and ends on a fixed date, so the
  dashboard's "last 30 days" means the last 30 days of the data.
- Nothing has a login. Put the dashboard behind authentication before using it
  with real data.
- The assistant knows the business profile only. It cannot answer questions
  about the reports, and each question stands alone (no memory).

## Where later work plugs in

| Next step | Where |
|---|---|
| Let the assistant answer report questions (tool use) | the `chat` route in `reports/routes.py`, offering `build_report` to the model as a tool |
| Follow-up questions (memory) | the same route, keeping earlier messages per session |
| A new dashboard chart | a metric in `reports/metrics/`, a line in `report.py`, a file in `js/sections/` |
| Real calendar booking | `BookingDesk.submit` in `booking/service.py` |
