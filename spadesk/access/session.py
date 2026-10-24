"""The owner's sign-in, and the check every protected route makes.

Signing in: the owner types the password set in SPADESK_OWNER_PASSWORD on
/login. If it matches, the server answers with a cookie that says "owner
until <time>", signed with a key only the server knows, so the browser
cannot make or change one. Every later request carries the cookie; a valid,
unexpired one makes the request the owner's. Anything else (no cookie, an
expired, altered or made-up one) is a customer.

Without SPADESK_OWNER_PASSWORD, nobody can sign in, so the owner's pages
and data stay closed: locked is the safe default.

The signing key is SPADESK_SESSION_SECRET, or, if that is not set, a random
one made when the server starts, which signs everyone out on a restart.

`require(permission)` is used by routes: it answers 401 when the request's
role lacks the permission (permissions.py).
"""

import hashlib
import hmac
import secrets
import time
from functools import lru_cache

from fastapi import HTTPException, Request

from ..config import get_settings
from .permissions import Role, allowed

COOKIE = "spadesk_owner"
HOURS = 12  # how long a sign-in lasts


@lru_cache  # one key for the life of the server, if none is set
def _key() -> bytes:
    """The key cookies are signed with."""
    return (get_settings().session_secret or secrets.token_hex(32)).encode()


def _signature(until: int) -> str:
    """The signature of "owner until `until`": only the server can make it."""
    return hmac.new(_key(), f"owner|{until}".encode(), hashlib.sha256).hexdigest()


def password_matches(password: str) -> bool:
    """True when `password` is the owner's. False whenever no password is set."""
    expected = get_settings().owner_password
    # compare_digest takes the same time however much matches, so the password
    # cannot be guessed letter by letter from how long the answer takes.
    return bool(expected) and hmac.compare_digest(password.encode(), expected.encode())


def new_session() -> tuple[str, int]:
    """A fresh owner cookie value, and how many seconds it lasts."""
    until = int(time.time()) + HOURS * 3600
    return f"{until}.{_signature(until)}", HOURS * 3600


def role_of(request: Request) -> Role:
    """The owner if the request carries a valid, unexpired owner cookie; otherwise a customer."""
    until, _, signature = request.cookies.get(COOKIE, "").partition(".")
    if not until.isdigit() or int(until) < time.time():
        return Role.CUSTOMER
    if not hmac.compare_digest(signature, _signature(int(until))):
        return Role.CUSTOMER
    return Role.OWNER


def require(permission: str):
    """A route dependency that lets the request through only if its role has `permission`."""
    def check(request: Request) -> Role:
        role = role_of(request)
        if not allowed(role, permission):
            raise HTTPException(status_code=401, detail="Please sign in as the owner.")
        return role
    return check
