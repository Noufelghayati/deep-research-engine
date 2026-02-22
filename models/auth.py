from pydantic import BaseModel
from typing import Optional


class GoogleAuthRequest(BaseModel):
    google_token: str


class AuthResponse(BaseModel):
    token: str
    user: dict


class UserInfo(BaseModel):
    google_id: str
    email: str
    name: str
    picture: Optional[str] = None
