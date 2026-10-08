"""Ratings, sentiment and feedback tags from completed visits."""

from collections import Counter
from typing import Any

from .common import Booking, average_rating


def feedback(rows: list[Booking]) -> dict[str, Any]:
    """How customers rated their visits: the number of rated visits, the
    average rating, a count per star rating, a count per sentiment, and a
    count per feedback tag (most common first).
    """
    rated = [b for b in rows if b["rating"] is not None]
    ratings = Counter(b["rating"] for b in rated)
    sentiments = Counter(b["sentiment"] for b in rated)
    # A tag can be given with different sentiments ("Expensive" by a neutral or an
    # unhappy customer), so tags are counted per sentiment.
    tags = Counter((b["sentiment"], b["feedback_tag"]) for b in rated if b["feedback_tag"])
    return {
        "rated": len(rated),
        "average_rating": average_rating(rated),
        "ratings": [{"rating": stars, "count": ratings.get(stars, 0)} for stars in (5, 4, 3, 2, 1)],
        "sentiments": [{"sentiment": s, "count": sentiments.get(s, 0)} for s in ("Happy", "Neutral", "Unhappy")],
        "tags": [{"tag": tag, "sentiment": sentiment, "count": count}
                 for (sentiment, tag), count in tags.most_common()],
    }

