"""Tests for access checks (spadesk/access/): customers and the owner each
reach only what the permission table allows.

Run with:  .venv/bin/python -m pytest tests/test_access.py
"""

import time

import pytest
from conftest import signed_in
from fastapi.testclient import TestClient

from spadesk.access import session
from spadesk.access.permissions import PERMISSIONS, Role, allowed
from spadesk.agent.pebby import NEEDS, agent_tools
from spadesk.app import app

OWNER_ONLY = ["/api/report/options", "/api/report", "/api/voice-widget"]


@pytest.fixture
def visitor():
    """A test browser that has not signed in: a customer."""
    return TestClient(app)


# ---- the owner's pages and data ----------------------------------------------------------

def test_a_visitor_is_sent_to_sign_in_and_refused_the_owners_data(visitor):
    page = visitor.get("/manage", follow_redirects=False)
    assert page.status_code == 307 and page.headers["location"] == "/login?next=/manage"
    for path in OWNER_ONLY:
        assert visitor.get(path).status_code == 401, path
    assert visitor.post("/api/report/chat", json={"question": "What was revenue last month?"}).status_code == 401


def test_the_owner_signs_in_reaches_everything_and_signs_out():
    owner = signed_in(app)
    assert owner.get("/manage").status_code == 200
    assert all(owner.get(path).status_code == 200 for path in OWNER_ONLY)
    owner.post("/api/logout")
    assert owner.get("/api/report").status_code == 401


def test_a_wrong_password_is_refused(visitor):
    reply = visitor.post("/api/login", json={"password": "guess"})
    assert reply.status_code == 401 and session.COOKIE not in reply.cookies


def test_without_an_owner_password_nobody_can_sign_in(visitor, monkeypatch):
    monkeypatch.setenv("SPADESK_OWNER_PASSWORD", "")
    assert visitor.post("/api/login", json={"password": ""}).status_code == 401


def test_the_sign_in_cookie_cannot_be_read_by_scripts_or_sent_by_other_sites(visitor):
    reply = visitor.post("/api/login", json={"password": "test-owner-password"})
    cookie = reply.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie


def test_a_made_up_altered_or_expired_cookie_is_a_customer(visitor):
    later, earlier = int(time.time()) + 3600, int(time.time()) - 1
    for value in ["owner",  # made up
                  f"{later}.not-the-signature",  # altered: a time without the server's signature
                  f"{earlier}.{session._signature(earlier)}"]:  # genuine, but expired
        visitor.cookies.set(session.COOKIE, value)
        assert visitor.get("/api/report").status_code == 401, value


def test_the_customer_site_stays_open_to_everyone(visitor):
    for path in ["/", "/api/spa", "/api/profile", "/api/booking/options", "/login"]:
        assert visitor.get(path).status_code == 200, path


# ---- what the model may be given -----------------------------------------------------------

def test_the_owner_has_every_customer_permission_and_more():
    assert PERMISSIONS[Role.CUSTOMER] < PERMISSIONS[Role.OWNER]
    assert not allowed(Role.CUSTOMER, "reports") and not allowed(Role.CUSTOMER, "voice_test")
    assert not allowed(Role.OWNER, "not-a-permission")


def test_a_customers_conversation_never_holds_the_owners_tools():
    names = [tool.name for tool in agent_tools("Asha", "4085550101")]
    assert "get_report" not in names
    assert set(names) <= set(NEEDS)  # every tool offered names the permission it needs
