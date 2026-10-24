"""Signing the owner in and out.

  GET  /login        the sign-in page (static/login/)
  POST /api/login    {"password": "..."} -> sets the owner cookie, or 401
  POST /api/logout   removes it

The cookie is HttpOnly (page scripts cannot read it), SameSite=Strict (other
sites cannot make the browser send it) and, over HTTPS, Secure (never sent
unencrypted).
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .session import COOKIE, new_session, password_matches

PAGE = Path(__file__).resolve().parent.parent / "static" / "login" / "index.html"

router = APIRouter()


class SignIn(BaseModel):
    """What the sign-in form sends."""

    password: str = Field(max_length=200)


@router.get("/login")
def login_page() -> FileResponse:
    """The sign-in page."""
    return FileResponse(PAGE)


@router.post("/api/login")
def login(body: SignIn, request: Request, response: Response) -> dict:
    """Sign the owner in when the password matches."""
    if not password_matches(body.password):
        raise HTTPException(status_code=401, detail="That password is not right.")
    value, seconds = new_session()
    response.set_cookie(COOKIE, value, max_age=seconds, httponly=True, samesite="strict",
                        secure=request.url.scheme == "https")
    return {"signed_in": True}


@router.post("/api/logout")
def logout(response: Response) -> dict:
    """Sign the owner out."""
    response.delete_cookie(COOKIE)
    return {"signed_in": False}
