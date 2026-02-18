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
    TENSION = "TENSION"


CATEGORY_ICONS = {
    "GROWTH": "\U0001f680",
    "MARKET": "\U0001f4b0",
    "PRODUCT": "\U0001f527",
    "CHALLENGE": "\U0001f6a8",
    "TRACTION": "\U0001f4ca",
    "BACKGROUND": "\U0001f3af",
    "TENSION": "\u2696\ufe0f",
}


class SignalSource(BaseModel):
    type: str = Field(description="'podcast', 'video', or 'article'")
    title: str = ""
    url: str = ""
    timestamp: Optional[str] = Field(None, description="MM:SS for videos")
    date: Optional[str] = None


class Signal(BaseModel):
    id: int
    category: str
    icon: str
    signal: str = Field(description="Action + context + revelation, max 25 words")
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
    executive_summary: Optional[str] = Field(
        None,
        description="1-sentence executive snapshot, max 25 words",
    )


# ── Executive Orientation (Quick Prep — between header and signals) ──

class ExecutiveOrientation(BaseModel):
    """3-5 flexible orientation bullets + key pressure line for Quick Prep."""
    bullets: List[str] = Field(
        default_factory=list,
        description="3-5 orientation bullets, each a different strategic dimension",
    )
    key_pressure: str = Field(
        default="",
        description="Single key pressure/vulnerability line (evidence-based)",
    )


# ── Research Confidence (top of Dossier) ──

class ResearchConfidence(BaseModel):
    level: str = "low"  # "high", "medium", "low"
    label: str = ""
    reasons: List[str] = Field(
        default_factory=list,
        description="Human-readable reasons supporting the confidence level",
    )


# ── Deep Dive Models ──

class DeepPressurePoint(BaseModel):
    """A single pressure point in the Deep Pressure section."""
    name: str = Field(default="", description="3-5 word label")
    description: str = Field(default="", description="2-3 sentences with citations")
    receptivity: str = Field(default="", description="What this makes them more/less receptive to")
    forward_facing: bool = Field(default=False, description="Is this a future-facing risk?")
    inferred: bool = Field(default=False, description="Is this inferred rather than directly stated?")


class QuoteEvidence(BaseModel):
    """Quote evidence for Pattern Evidence section."""
    quote: str = Field(default="", description="The strongest quote proving the thesis")
    source: str = Field(default="", description="Speaker | Platform - Date - Timestamp")
    source_url: str = Field(default="", description="URL to source")
    why_strongest: str = Field(default="", description="Why this quote is the strongest proof")


class PatternEvidence(BaseModel):
    """Pattern Evidence section proving Core Read thesis."""
    thesis: str = Field(default="", description="The Core Read thesis being proven")
    quote_evidence: Optional[QuoteEvidence] = None
    behavior_evidence: str = Field(default="", description="Behavioral proof, 2-3 sentences")
    company_evidence: str = Field(default="", description="Company-level proof, 2-3 sentences")


class OwnWordsQuote(BaseModel):
    """A single quote in the In Their Own Words section."""
    quote: str = Field(default="", description="Verbatim quote")
    source: str = Field(default="", description="Speaker | Platform | Date | Timestamp")
    source_url: str = Field(default="", description="URL to source")
    insight: str = Field(default="", description="What this reveals beyond synthesis")


class InTheirOwnWords(BaseModel):
    """In Their Own Words section."""
    core: List[OwnWordsQuote] = Field(default_factory=list, description="Quotes proving Core Read")
    supporting: List[OwnWordsQuote] = Field(default_factory=list, description="Context quotes")
    outlier: List[OwnWordsQuote] = Field(default_factory=list, description="Contradicting quotes")
    additional_count: int = Field(default=0, description="Extra quotes beyond displayed 10")
    limited_coverage_note: str = Field(default="", description="Note if fewer than 4 quotes")


class DossierSource(BaseModel):
    """A single source in the appendix."""
    type: str = Field(description="'primary' or 'supporting'")
    label: str = Field(default="", description="PODCAST / VIDEO / ARTICLE")
    title: str = ""
    platform: str = ""
    date: str = ""
    duration: Optional[str] = None
    url: str = ""
    context_only: bool = Field(default=False, description="Source provided context but no quotes")
    # Backward compat fields (old dossier sources had icon)
    icon: str = ""


class FullDossier(BaseModel):
    """VIEW 2: Deep Dive (replaces Full Dossier)."""
    research_confidence: Optional[ResearchConfidence] = None
    thin_signal_warning: Optional[str] = None
    deep_pressure: List[DeepPressurePoint] = Field(default_factory=list)
    pattern_evidence: Optional[PatternEvidence] = None
    in_their_own_words: Optional[InTheirOwnWords] = None
    sources: List[DossierSource] = Field(default_factory=list)
    # Backward compat — old fields kept as empty defaults so cached data doesn't crash
    background: Optional[dict] = None
    executive_profile: Optional[dict] = None
    strategic_focus: Optional[dict] = None
    quotes: List[dict] = Field(default_factory=list)
    momentum: Optional[dict] = None
    momentum_grouped: Optional[list] = None


class RecentMove(BaseModel):
    """A factual event from the last 90 days with tier, signal, and hook."""
    tier: str = Field(default="", description="THEIR WORDS, THEIR ATTENTION, COMPANY NEWS, or NONE")
    event: str = Field(description="Factual event description")
    date: str = Field(default="", description="Month Year, e.g. 'January 2026'")
    signal: str = Field(default="", description="What this tells you about their current priorities")
    hook: str = Field(default="", description="How an AE can reference this naturally")
    source_url: Optional[str] = None
    source_title: Optional[str] = None


class OpeningMove(BaseModel):
    """A single conversation opening direction."""
    angle: str = Field(description="Short label, e.g. 'Scaling Pain'")
    suggestion: str = Field(description="Max 25 words: '[Lead with X] — [ask about Y]'")


class ResearchResponse(BaseModel):
    person: PersonInfo
    executive_orientation: Optional[ExecutiveOrientation] = None
    core_read: str = Field(default='', description='Cross-source thesis statement')
    recent_moves: List[RecentMove] = Field(
        default_factory=list,
        description="Up to 4 factual events from the last 90 days",
    )
    signals: List[Signal] = Field(default_factory=list, max_length=5)
    opening_moves: List[OpeningMove] = Field(
        default_factory=list,
        description="3 conversation opening directions for the sales rep",
    )
    pull_quote: Optional[dict] = Field(
        None,
        description="Best direct quote: {quote, source}",
    )
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
