"""How spaDesk's agents behave: six principles, and their settings.

Each principle is enforced in code (the files named), not left to a model's
instructions; the settings below it are imported from here by that code.
A new situation should fit one of these, not add a rule of its own.

  1. Nothing the customer says is dropped.
     Name and phone come first (on a call, the number calling stands in for
     the phone); whatever they ask meanwhile is answered as soon as those are
     given. A returning customer who has not yet said what they want is asked
     whether it is about an upcoming booking. (flow.py, pebby.py)

  2. Every detail is the customer's own.
     The services, day, time, therapist and the bookings to change or cancel
     must be picked out by what the customer wrote; a detail a model filled in
     is dropped and the customer is asked. A word picks out the one thing that
     has it ("prenatal", "6pm"); a kind ("massage" among NAMED_BY or more)
     picks out nothing; a package's name picks out all its parts; "all" or
     "both" (EVERYTHING) picks out every one. Days and times are read from the
     words by code (PARTS_OF_DAY, ANY_TIME). A request needs what NEEDS lists.
     (details.py, details.py)

  3. Nothing changes without a yes to exactly what will happen.
     Code writes each question: the details read back, the plan found, the
     bookings to cancel (or what a change cancels and books). A yes that also
     changes or questions it (YES, BUT) is not a yes. (yes.py)

  4. A booking is all or nothing, and never twice.
     A plan is re-checked at the yes and booked whole or not at all; a change
     books the new times before cancelling the old booking, whose time counts
     as free meanwhile. Within SPADESK_MIN_NOTICE_MINUTES nothing can be made,
     changed or cancelled. (executor.py, spadesk/booking/appointments.py)

  5. Work is bounded; past a bound, a person takes over.
     Too many rounds (MAX_TOOL_ROUNDS) or repeats (MAX_REPEATS) in a turn, an
     approved plan failing MAX_FAILED_PLANS times, a request beyond one booking
     (more than MAX_SERVICES services; a REPEATING day or time), changing part
     of a package,
     something the agents cannot do, or any failure: the request is written
     down for the staff and the customer is told, never shown an error.
     Feedback, complaints included, becomes a card the owner closes. No
     request is left waiting: every one is answered, the handover at worst
     (CHAT_TURN_LIMIT). (flow.py, executor.py, booking.py,
     staff.py, staff.py, agent/__init__.py, voice/routes.py)

  6. A model only where thinking is needed.
     Pebby talks and decides (PEBBY_MODEL); the Feedback agent reads
     feedback and answers it (FEEDBACK_MODEL); a small model reads a name
     (READER_MODEL). Everything exact (dates, times, fitting, booking) is code.

Who may use which tool is the access rules' (spadesk/access/permissions.py).
"""

import re
from datetime import time

# ---- 2: every detail is the customer's own --------------------------------------------------

# A booking request's intent -> the details it needs before anything is looked up.
NEEDS = {"new": ["services", "day", "time"], "change": ["references", "services", "day", "time"],
         "availability": ["services"], "cancel": ["references"], "my_bookings": []}
# A word shared by this many things or more is a kind ("massage", "facial", "wax"), not a name:
# "a facial" picks out no service; "deep tissue" picks out Deep Tissue Massage.
NAMED_BY = 3
# Words that name every booking Pebby listed: "cancel them all".
EVERYTHING = {"all", "both", "everything"}

# A part of the day -> (after, before): the times of day it means.
PARTS_OF_DAY = {"morning": (None, time(12)), "afternoon": (time(12), time(17)), "evening": (time(17), None),
                "after work": (time(17), None), "lunch": (time(12), time(14))}
# Words that mean any time of day is fine: an answer, not a missing detail.
ANY_TIME = re.compile(r"\b(any ?time|anytime|whenever|no preference|doesn'?t matter|any|flexible)\b")

# ---- 3: a yes to exactly what will happen -----------------------------------------------------

# A reply that starts like this is a yes...
YES = re.compile(r"^\s*(y|yes|yeah|yep|yup|sure|ok|okay|confirm(ed)?|please do|go ahead|do it|book it|cancel it|"
                 r"correct|that'?s right)\b", re.IGNORECASE)
# ...unless it goes on to change or question it: "yes, but at 4pm instead?" is not a yes.
BUT = re.compile(r"\b(but|instead|change|different|another|not|don'?t)\b|\?", re.IGNORECASE)

# ---- 5: bounds, past which a person takes over --------------------------------------------------

MAX_FAILED_PLANS = 2  # approved plans in a row that could not be booked (a time taken meanwhile)
# Beyond one booking request, arranged by the staff: more services than this at once (a bulk request),
# or a day or time that repeats ("every Friday", "weekly for two months").
MAX_SERVICES = 5
REPEATING = re.compile(r"\b(every|each|weekly|monthly|daily|fortnightly|recurring|regularly)\b", re.IGNORECASE)
MAX_TOOL_ROUNDS = 5  # Pebby's tool rounds for one customer message
MAX_REPEATS = 2  # rounds in one turn in which Pebby repeats a request (after the first, it must answer)

# ---- 6: a model only where thinking is needed ------------------------------------------------------

PEBBY_MODEL = "booking-agent"  # the front agent: talks with the customer, decides what to do
READER_MODEL = "cheap-router"  # small and fast: reads a name and number
FEEDBACK_MODEL = "cheap-router"  # the Feedback agent: reads feedback and writes the reply to it
FEEDBACK_QUESTIONS = 1  # questions the Feedback agent may ask about one piece of feedback (never an interrogation)

# ---- offering days (2: the customer chooses from them) -----------------------------------------

LOOK_AHEAD_DAYS = 14  # "which days fit": how far ahead to look
MAX_DAYS = 4  # and how many days to offer

# ---- 1: turns that overlap (on a call, a caller who keeps talking) -------------------------------

WAIT_FOR_NEWER = 10.0  # seconds a replaced turn waits for the newer turn's reply (turns.py)
# Every request is answered: a chat turn still running after this many seconds ends in the
# handover (routes.py), as a phone turn does after its own limit (voice/routes.py).
CHAT_TURN_LIMIT = 45.0
