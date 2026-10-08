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
.venv/bin/pytest
```

The tests do not need an API key and do not call Claude.

## Changing things

| To change | Edit | Then |
|---|---|---|
| Services, prices, hours, policies | `data/family_spa_business_doc.md` | restart the server, then regenerate the bookings (below) |
| The mock bookings (volume, mix, period) | the constants at the top of `scripts/generate_bookings.py` | run `.venv/bin/python scripts/generate_bookings.py`, then restart |
| What the dashboard assistant sends to the model | the `chat` route in `spadesk/reports/routes.py` | saved automatically with `--reload` |
| Featured services, highlighted policies | `spadesk/config.py` | restart |
| Which model the assistant uses | `.env` | restart |
| Customer site text, layout, colours | `spadesk/static/` | refresh the browser |
| Dashboard tiles, charts, colours | `spadesk/static/dashboard/` | refresh the browser |

## Settings in `.env`

| Name | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Your API key. Required for the dashboard assistant. |
| `SPADESK_MODEL` | `claude-opus-5-5` | Which Claude model the assistant uses. |
| `SPADESK_DATA` | `data/family_spa_business_doc.md` | The business document. |
| `SPADESK_REPORT_DATA` | `data/spadesk_bookings.json` | The bookings file the dashboard reads. |

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
- **The dashboard shows an error instead of numbers**: the bookings file is
  missing or a booking points at a service that is not in its `services` list.
  Regenerate it with `.venv/bin/python scripts/generate_bookings.py`.
- **`command not found: uvicorn`**: use `.venv/bin/uvicorn`, or run step 1 again.
