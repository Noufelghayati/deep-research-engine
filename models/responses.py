from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SignalCategory(str, Enum):
    GROWTH = "GROWTH"
    MARKET = "MARKET"
    PRODUCT = "PRODUCT"
    CHALLENGE = "CHALLENGE"
    TRACTION = "TRACTION"
    BACKGROUND = "BACKGROUND"


CATEGORY_ICONS = {
    "GROWTH": "\U0001f680",
    "MARKET": "\U0001f4b0",
    "PRODUCT": "\U0001f527",
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
        description="Contains 'quote' (15-40 words) and 'source' object",
    )


class PersonInfo(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    company: str
    prior_role: Optional[str] = Field(
        None, description="e.g. 'CMO, Impossible Foods'"
    )


class DossierBackground(BaseModel):
    """Section 1: Identity & Background — max 6 bullets."""
    bullets: List[str] = Field(default_factory=list, max_length=6)


class DossierStrategicFocus(BaseModel):
    """Section 2: Current Strategic Focus — 3-6 synthesized themes."""
    themes: List[dict] = Field(
        default_factory=list,
        description="Each: {category, icon, title, bullets: [str], sources: [str]}",
    )


class DossierQuote(BaseModel):
    """A single quote in Section 3."""
    topic: str = Field(description="e.g. 'On Food Waste'")
    quote: str = Field(description="15-40 words, direct quote")
    source: str = Field(description="Source title + timestamp/page")


class DossierMomentum(BaseModel):
    """Section 4: Recent Company Momentum — max 6 bullets."""
    bullets: List[str] = Field(default_factory=list, max_length=6)


class DossierSource(BaseModel):
    """A single source in the appendix."""
    type: str = Field(description="'primary' or 'supporting'")
    icon: str = ""
    title: str = ""
    platform: str = ""
    date: str = ""
    duration: Optional[str] = None
    url: str = ""


class FullDossier(BaseModel):
    """VIEW 2: Full Dossier."""
    background: DossierBackground = Field(default_factory=DossierBackground)
    strategic_focus: DossierStrategicFocus = Field(
        default_factory=DossierStrategicFocus
    )
    quotes: List[DossierQuote] = Field(default_factory=list)
    momentum: DossierMomentum = Field(default_factory=DossierMomentum)
    sources: List[DossierSource] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    person: PersonInfo
    signals: List[Signal] = Field(default_factory=list, max_length=5)
    dossier: Optional[FullDossier] = Field(
        None, description="Full Dossier (VIEW 2)"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Edge-case warnings shown to user",
    )
    sources_analyzed: Optional[dict] = Field(
        None,
        description="Summary: videos count, articles count",
    )
    metadata: Optional[dict] = Field(
        None, description="Debug info: steps attempted, timings, etc."
    )


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    step_reached: Optional[str] = None
