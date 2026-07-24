"""
Shared local-timezone handling.

Not per-project config: this reflects the user's physical locations
(Ivanovo, RF and Turkey), both fixed UTC+3 year-round (neither observes DST),
so the same offset is correct for all three projects regardless of which
project's data is being viewed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

LOCAL_TZ = timezone(timedelta(hours=3))
LOCAL_TZ_LABEL = "MSK"


def to_local(ts: str) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp (optionally 'Z'-suffixed) into local time."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ)
    except ValueError:
        return None
