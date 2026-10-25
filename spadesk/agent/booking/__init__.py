"""Agent 2, the Booking agent: everything between Pebby and the calendar.

Pebby collects a booking's details in conversation and sends them with its
booking tool; from there:

    collect.py     checks them against the customer's own words, reads days and
                   times (when_words.py), says what is still missing
    yes.py         the questions for the customer's yes: the details read back,
                   then the plan found, or the bookings to cancel
    agent.py       the Booking agent: a BookingRequest in, a BookingResult out
                   (in code; its model only thinks about a request's notes)
    plan.py        the Plan, checking it can still be booked, and fitting the
                   services into a day
    executor.py    books the plan, all or none; carries out what was approved

The rules they follow are in ../rules.py.
"""
