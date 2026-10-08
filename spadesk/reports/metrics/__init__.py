"""The dashboard's calculations, one small file per topic.

Every function here takes a list of bookings (already filtered by
`Dataset.select`) and returns plain numbers, lists and dictionaries, ready to
be sent to the page as JSON. None of them read files or know about the web.

  common.py     shared helpers: the totals every breakdown reports
  period.py     date maths: the previous period, and day/week/month buckets
  summary.py    the headline numbers
  trend.py      bookings and revenue over time
  services.py   per category and per service
  channels.py   per source channel
  customers.py  new and returning customers, counted as people
  feedback.py   ratings, sentiment and feedback tags
"""
