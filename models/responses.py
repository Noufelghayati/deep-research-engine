from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SignalCategory(str, Enum):
    SCALING = "SCALING"
    INVESTING = "INVESTING"
    PRIORITY = "PRIORITY"
    CHALLENGE = "CHALLENGE"
    TRACTION = "TRACTION"
    BACKGROUND = "BACKGROUND"


CATEGORY_ICONS = {
    "SCALING": "\U0001f680",
    "INVESTING": "\U0001f4b0",
    "PRIORITY": "\U0001f527",
    "CHALLENGE": "\U0001f6a8",
    "TRACTION": "\U0001f4ca",
    "BACKGROUND": "\U0001f3af",
}


class SignalSource(BaseModel):
    type: str = Field(description="'video' or 'article'")
    title: str = ""
    url: str = ""
    timestamp: Optional[str] = Field(None, description="MM:SS for videos")
    date: Optional[str] = None


class Signal(BaseModel):
    id: int
    category: str
    icon: str
    signal: str = Field(description="One-line declarative statement, max 20 words")
    source_type: str = Field(description="VIDEO or ARTICLE")
    expandable: dict = Field(
        default_factory=dict,
        description="Contains 'quote' (15-30 words) and 'source' object",
    )


class PersonInfo(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    company: str


class ResearchResponse(BaseModel):
    person: PersonInfo
    signals: List[Signal] = Field(default_factory=list, max_length=5)
    sources_analyzed: Optional[dict] = Field(
        None,
        description="Summary: videos count, articles count, recency",
    )
    metadata: Optional[dict] = Field(
        None, description="Debug info: steps attempted, timings, etc."
    )


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    step_reached: Optional[str] = None
