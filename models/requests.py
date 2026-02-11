from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class UserRole(str, Enum):
    AE = "account_executive"
    SDR = "sales_dev_rep"
    CSM = "customer_success_manager"
    EXEC = "executive"


class ResearchRequest(BaseModel):
    user_id: Optional[str] = Field(None, max_length=200)
    target_name: Optional[str] = Field(
        None, max_length=200,
        description="Full name of the person to research",
    )
    target_title: Optional[str] = Field(
        None, max_length=200,
        description="Job title if known",
    )
    target_company: str = Field(
        ..., min_length=1, max_length=200,
        description="Company the person is associated with (required)",
    )
    page_url: Optional[str] = Field(
        None, max_length=2000,
        description="LinkedIn or other page URL for context",
    )
    your_name: Optional[str] = Field(None, max_length=200)
    your_company: Optional[str] = Field(None, max_length=200)
    user_role: UserRole = Field(
        default=UserRole.AE,
        description="Caller's sales role — affects tone of output",
    )
    context: Optional[str] = Field(
        None, max_length=1000,
        description="Optional free text: deal stage, product interest, etc.",
    )
