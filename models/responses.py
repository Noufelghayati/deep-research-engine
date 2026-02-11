from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SourceType(str, Enum):
    YOUTUBE = "youtube"
    ARTICLE = "article"


class MatchType(str, Enum):
    PERSON = "person"
    COMPANY_LEADERSHIP = "company_leadership"
    COMPANY_CONTEXT = "company_context"


class SelectedSource(BaseModel):
    type: SourceType
    match_type: MatchType
    title: str
    url: str
    why_selected: str
    company_match_signals: List[str] = Field(default_factory=list)


class TalkingPoint(BaseModel):
    point: str
    source_url: str
    timestamp: Optional[str] = Field(
        None, description="MM:SS timestamp into video, if applicable"
    )


class ResearchResponse(BaseModel):
    selected_sources: List[SelectedSource]
    pre_read: List[str] = Field(
        default_factory=list,
        description="Max ~7 bullet points for quick pre-read",
    )
    talking_points: List[TalkingPoint] = Field(default_factory=list)
    draft_email: str = ""
    metadata: Optional[dict] = Field(
        None, description="Debug info: steps attempted, timings, etc."
    )


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    step_reached: Optional[str] = None
