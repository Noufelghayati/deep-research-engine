import re
from typing import List, Optional, Tuple
from models.internal import YouTubeCandidate, ScoredVideo
from services.youtube_transcript import fetch_transcript
from config import settings
import logging

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _company_variants(company_name: str) -> tuple[List[str], List[str]]:
    """
    Generate strong and weak match variants for a company name.
    Returns (strong_variants, weak_variants).

    Strong: full name, no-space form — these are reliable.
    Weak: individual words — only used for single-word company names,
          otherwise too many false positives (e.g. "Hale" matching Lzzy Hale).
    """
    base = _normalize(company_name)
    words = base.split()

    strong = [base, base.replace(" ", "")]

    # For single-word companies (e.g. "Tesla"), the word IS the full name
    # For multi-word companies, individual words are unreliable
    weak = []
    if len(words) == 1:
        # Single-word company: strong match is the word itself, no weak needed
        pass
    else:
        # Multi-word: individual words are weak signals only
        for word in words:
            if len(word) > 4:
                weak.append(word)

    return list(set(strong)), list(set(weak) - set(strong))


def _count_strong_mentions(text: str, strong_variants: List[str]) -> int:
    """Count how many strong (full name) variants appear in text."""
    normalized = _normalize(text)
    return sum(1 for v in strong_variants if v in normalized)


def _count_weak_mentions(text: str, weak_variants: List[str]) -> int:
    """Count how many weak (single word) variants appear in text."""
    normalized = _normalize(text)
    return sum(1 for v in weak_variants if v in normalized)


# C-suite title priority: CEO/Founder > other C-suite > VP/Director > panel/keynote
_CSUITE_PRIORITY = [
    (0.10, ["ceo", "founder", "co-founder", "cofounder"]),
    (0.05, ["cro", "cfo", "cto", "coo", "cmo", "cio", "cso", "chro",
            "chief revenue", "chief financial", "chief technology",
            "chief operating", "chief marketing", "chief information",
            "chief strategy", "chief human resources"]),
    (0.02, ["vp ", "vice president", "director", "executive"]),
    (0.01, ["panel", "keynote", "leadership"]),
]


def _csuite_title_bonus(text: str) -> tuple[float, str]:
    """Return (bonus_score, signal_name) based on highest C-suite title found."""
    normalized = _normalize(text)
    for bonus, titles in _CSUITE_PRIORITY:
        for title in titles:
            if title in normalized:
                return bonus, f"csuite_priority_{title.strip().replace(' ', '_')}"
    return 0.0, ""


def _person_mentioned(text: str, person_name: Optional[str]) -> bool:
    if not person_name:
        return False
    normalized = _normalize(text)
    parts = _normalize(person_name).split()
    # Full name match
    if _normalize(person_name) in normalized:
        return True
    # Last name match
    if len(parts) >= 2 and parts[-1] in normalized and len(parts[-1]) > 2:
        return True
    return False


async def score_candidate(
    candidate: YouTubeCandidate,
    person_name: Optional[str],
    company_name: str,
    fetch_transcript_for_scoring: bool = True,
) -> ScoredVideo:
    """
    Score a YouTube candidate for company/person relevance.

    Scoring rubric (0.0 to 1.0):
      Company in title:        +0.30
      Person in title:         +0.25
      Company in description:  +0.15
      Person in description:   +0.10
      Channel matches company: +0.10
      Company in transcript:   +0.15
      Person in transcript:    +0.05
    """
    score = 0.0
    signals = []
    strong_variants, weak_variants = _company_variants(company_name)

    # Title signals — strong match = full points, weak-only = minimal
    title_strong = _count_strong_mentions(candidate.title, strong_variants)
    title_weak = _count_weak_mentions(candidate.title, weak_variants)
    if title_strong > 0:
        score += 0.30
        signals.append("title")
    elif title_weak > 0:
        score += 0.05  # Weak match alone is almost worthless
        signals.append("title_weak")

    is_person_match = False
    if _person_mentioned(candidate.title, person_name):
        score += 0.25
        signals.append("person_in_title")
        is_person_match = True

    # Description signals — same two-tier approach
    desc_strong = _count_strong_mentions(candidate.description, strong_variants)
    desc_weak = _count_weak_mentions(candidate.description, weak_variants)
    if desc_strong > 0:
        score += 0.15
        signals.append("description")
    elif desc_weak > 0:
        score += 0.03
        signals.append("description_weak")

    if _person_mentioned(candidate.description, person_name):
        score += 0.10
        signals.append("person_in_description")
        is_person_match = True

    # Channel name signal — channel/publication clearly associated with company
    if candidate.channel_title:
        channel_strong = _count_strong_mentions(candidate.channel_title, strong_variants)
        if channel_strong > 0:
            score += 0.10
            signals.append("channel_matches_company")

    # C-suite title priority bonus (CEO/Founder > C-suite > VP > panel)
    title_bonus, title_signal = _csuite_title_bonus(candidate.title)
    if title_bonus > 0:
        score += title_bonus
        signals.append(title_signal)

    # Transcript signals — only fetch if metadata already has STRONG signal
    transcript_text = None
    transcript_available = False

    if fetch_transcript_for_scoring and score >= 0.25:
        transcript_text, transcript_available = await fetch_transcript(
            candidate.video_id
        )
        if transcript_text:
            excerpt = transcript_text[:5000]
            if _count_strong_mentions(excerpt, strong_variants) > 0:
                score += 0.15
                signals.append("transcript")
            elif _count_weak_mentions(excerpt, weak_variants) > 0:
                score += 0.03
                signals.append("transcript_weak")
            if _person_mentioned(excerpt, person_name):
                score += 0.05
                signals.append("person_in_transcript")
                is_person_match = True

    scored = ScoredVideo(
        video_id=candidate.video_id,
        title=candidate.title,
        description=candidate.description,
        channel_title=candidate.channel_title,
        published_at=candidate.published_at,
        match_score=round(min(score, 1.0), 2),
        match_signals=signals,
        is_person_match=is_person_match,
        transcript_text=transcript_text,
        transcript_available=transcript_available,
        url=f"https://www.youtube.com/watch?v={candidate.video_id}",
    )

    logger.info(
        f"Scored '{candidate.title[:50]}': {scored.match_score} "
        f"signals={signals} person={is_person_match}"
    )
    return scored


async def score_and_filter(
    candidates: List[YouTubeCandidate],
    person_name: Optional[str],
    company_name: str,
    max_keep: int = 2,
) -> Tuple[List[ScoredVideo], List[ScoredVideo]]:
    """Score all candidates, filter by threshold. Returns (kept, all_scored)."""
    all_scored = []
    for c in candidates:
        sv = await score_candidate(
            c, person_name, company_name, fetch_transcript_for_scoring=False
        )
        all_scored.append(sv)

    passed = [s for s in all_scored if s.match_score >= settings.disambiguation_threshold]
    passed.sort(key=lambda x: x.match_score, reverse=True)
    kept = passed[:max_keep]

    logger.info(
        f"Disambiguation: {len(candidates)} candidates -> "
        f"{len(passed)} passed threshold -> keeping {len(kept)}"
    )
    return kept, all_scored
