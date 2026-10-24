"""Booking: appointments with the team.

    schedule.py      what can be booked, with whom, when (from the business document)
    calendars/       where bookings are kept, one calendar per therapist (in memory or Google Calendar)
    appointments.py  the booking rules: free times, book, cancel, the day's bookings
    actions.py       those rules in the shape a model's tool request takes (offered by agent/tools.py)
    routes.py        those rules as the booking form's routes
"""
