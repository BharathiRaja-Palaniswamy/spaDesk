# Setting up Google Calendar for bookings

spaDesk keeps bookings in Google Calendar, one calendar per therapist
("Pebble Spa – Priya Nair", ...). The booking rules run in the app
(`spadesk/booking/`); Google Calendar is where bookings are kept and where
the owner sees them. This guide takes about 15 minutes. The Calendar API has
no charge.

The app signs in as a **service account**: a Google account for a program,
not a person. It creates the calendars, so it owns them, and shares them with
the owner. Nobody has to sign in from the server.

## 1. Create a Google Cloud project and turn on the Calendar API

1. Open [console.cloud.google.com](https://console.cloud.google.com), signed in
   with the owner's Google account.
2. Create a project, e.g. **spadesk**.
3. In **APIs & Services → Library**, search for **Google Calendar API** and
   click **Enable**.

## 2. Create the service account and its key

1. In **IAM & Admin → Service Accounts**, click **Create service account**.
   Name it e.g. **spadesk-booking**. It needs no roles: skip those steps.
2. Open it, go to **Keys → Add key → Create new key → JSON**. A `.json` file
   downloads. **It is a password**: never commit it, paste it in a chat or
   send it by email.
3. Move the file **outside the project folder**, e.g.
   `~/.config/spadesk/google-service-account.json`.

If Google answers **"An Organization Policy that blocks service accounts key
creation has been enforced"**, the project belongs to an organization that
forbids keys by default. To allow them for this project only: in **IAM &
Admin → Organization Policies**, with the project selected, open **Disable
service account key creation**, choose **Override parent's policy**, add a
rule with enforcement **Off**, and **Set policy** (this needs the
Organization Policy Administrator role, granted on the organization). Do the
same for its managed version (`iam.managed.disableServiceAccountKeyCreation`)
if it is listed. Wait a few minutes, then create the key.

## 3. Tell spaDesk where the key is, and who may see the calendars

On your laptop, in `.env`:

```
SPADESK_CALENDAR=google
GOOGLE_SERVICE_ACCOUNT=/Users/<you>/.config/spadesk/google-service-account.json
SPADESK_CALENDAR_OWNER=<a Gmail address>
```

`SPADESK_CALENDAR_OWNER` is optional. Booking works without it; it is only
needed to see the bookings in Google Calendar. Use an account made for the
spa (e.g. a new Gmail) rather than a personal one, so the spa's calendars
stay apart from your own.

On Replit, in **Secrets**: `GOOGLE_SERVICE_ACCOUNT` with the key file's
**whole JSON text** pasted as the value (the app accepts either the text or a
path), plus `SPADESK_CALENDAR=google` and, optionally, `SPADESK_CALENDAR_OWNER`.

## 4. Create the calendars

```
.venv/bin/python -m scripts.setup_calendars
```

It prints one line per therapist (`created` or `exists`), which shows the key
works, and shares every calendar with `SPADESK_CALENDAR_OWNER`: run it again
after setting that later. Then check that booking works:

```
.venv/bin/python -m scripts.check_calendar           # books, reads back and cancels a test appointment
.venv/bin/python -m scripts.check_calendar --keep    # leaves it on the calendar, to see it
```

 The app also creates a calendar by itself for anyone who joins the
team in the business document. Someone who leaves keeps their calendar, so
their bookings can still be found; delete it by hand once those are done.

## 5. See them in Google Calendar

Google emails the `SPADESK_CALENDAR_OWNER` account a link for each calendar;
open each to add it (or look under **Other calendars** at calendar.google.com). They
appear under **Other calendars**, each with its own colour and a tick box to
show or hide it. To give a therapist their own calendar on their phone: in
that calendar's **Settings and sharing**, add the therapist's Google account.

**Time off:** add an event to the therapist's calendar (e.g. "Priya –
dentist", 12:00–14:00). spaDesk stops offering those times. Events added by
hand are never listed as bookings.

## If something goes wrong

The chat answers "Booking is unavailable right now", and the server log has
a line starting `calendar unavailable:` with the reason:

- **GOOGLE_SERVICE_ACCOUNT is not set**: step 3.
- **403 / accessNotConfigured**: the Calendar API is not enabled (step 1).
- **invalid_grant**: the key was deleted or the clock is wrong; create a new key.
