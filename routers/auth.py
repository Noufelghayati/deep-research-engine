import logging

import httpx
from fastapi import APIRouter, HTTPException, Depends

from models.auth import GoogleAuthRequest, AuthResponse
from database import get_db
from utils.auth import create_jwt, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["auth"])


@router.post("/auth/google", response_model=AuthResponse)
async def google_auth(body: GoogleAuthRequest):
    """Exchange a Google access token for a JWT + user record."""

    # Validate token with Google's userinfo endpoint
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {body.google_token}"},
        )

    if resp.status_code != 200:
        logger.warning("Google token validation failed: %s", resp.text)
        raise HTTPException(status_code=401, detail="Invalid Google token")

    info = resp.json()
    google_id = info.get("sub")
    email = info.get("email")
    name = info.get("name", "")
    picture = info.get("picture", "")

    if not google_id or not email:
        raise HTTPException(status_code=401, detail="Could not retrieve Google profile")

    # Upsert user in SQLite
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO users (google_id, email, name, picture)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(google_id) DO UPDATE SET
                email = excluded.email,
                name = excluded.name,
                picture = excluded.picture,
                last_login = datetime('now')
            """,
            (google_id, email, name, picture),
        )
        conn.commit()

    logger.info("User authenticated: %s (%s)", email, name)

    # Issue JWT
    token = create_jwt({"sub": google_id, "email": email, "name": name})

    return AuthResponse(
        token=token,
        user={"email": email, "name": name, "picture": picture},
    )


@router.get("/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Return current user info from JWT."""
    return {
        "email": user.get("email"),
        "name": user.get("name"),
        "google_id": user.get("sub"),
    }
