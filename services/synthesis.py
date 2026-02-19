import anthropic
from google import genai
from google.genai import types
from models.internal import CollectedArtifacts
from models.requests import ResearchRequest
from models.responses import (
    ResearchResponse,
    PersonInfo,
    Signal,
    OpeningMove,
    RecentMove,
    FullDossier,
    DossierSource,
    ExecutiveOrientation,
    ResearchConfidence,
    DeepPressurePoint,
    PatternEvidence,
    QuoteEvidence,
    InTheirOwnWords,
    OwnWordsQuote,
    CATEGORY_ICONS,
)
from config import settings
from typing import List, Optional
from datetime import datetime, timedelta
import json
import re
import asyncio
import logging

logger = logging.getLogger(__name__)

gemini_client = genai.Client(api_key=settings.gemini_api_key)
claude_client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=600.0)

CATEGORY_PRIORITY = ["GROWTH", "CHALLENGE", "MARKET", "PRODUCT", "TENSION", "TRACTION", "BACKGROUND"]


# ═══════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════

def _is_within_90_days(date_str: str) -> bool:
    """Check if a date string like 'January 2026' or 'March 2025' is within 90 days of now."""
    if not date_str:
        return False
    # Common formats: "January 2026", "Jan 2026", "2026-01-15", "Q1 2026"
    today = datetime.utcnow()
    cutoff = today - timedelta(days=90)
    try:
        # Try "Month Year" format
        for fmt in ("%B %Y", "%b %Y", "%Y-%m-%d", "%B %d, %Y"):
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                # For month-only formats, use the 1st of the month
                return dt >= cutoff
            except ValueError:
                continue
        # Try extracting year and month from freeform text
        import calendar
        months = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
        months.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
        date_lower = date_str.lower()
        year = None
        month = None
        for word in date_lower.split():
            if word in months:
                month = months[word]
            elif word.isdigit() and len(word) == 4:
                year = int(word)
        if year and month:
            dt = datetime(year, month, 1)
            return dt >= cutoff
    except Exception:
        pass
    # Can't parse — keep it (don't filter ambiguous dates)
    return True

def _format_extracted_signals(signals) -> str:
    """Format pre-extracted signals for a single source."""
    if not signals:
        return ""
    lines = ["PRE-EXTRACTED SIGNALS:"]
    for sig in signals:
        line = f"  [{sig.signal_type.upper()}] {sig.content}"
        if sig.quote:
            line += f' \u2014 "{sig.quote}"'
        if sig.timestamp:
            line += f" ({sig.timestamp})"
        line += f" [confidence: {sig.confidence}]"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _validate_pull_quote(pq_raw, artifacts) -> Optional[dict]:
    """Validate and return pull quote only if it's from a video/podcast transcript.
    Rejects quotes sourced from articles/LinkedIn — Gemini often ignores prompt rules."""
    if not pq_raw:
        return None

    if isinstance(pq_raw, dict) and pq_raw.get("quote"):
        quote_text = pq_raw["quote"]
        source_str = (pq_raw.get("source") or "").lower()
        source_url = pq_raw.get("source_url") or ""
    elif isinstance(pq_raw, str) and pq_raw:
        quote_text = pq_raw
        source_str = ""
        source_url = ""
    else:
        return None

    source_display = pq_raw.get("source") or "" if isinstance(pq_raw, dict) else ""

    why_it_matters = pq_raw.get("why_it_matters") or "" if isinstance(pq_raw, dict) else ""

    def _build_result(matched_url: str = "") -> dict:
        """Build pull quote result, using matched_url as fallback if Gemini didn't provide one."""
        url = source_url or matched_url
        result = {"quote": quote_text, "source": source_display}
        if url:
            result["source_url"] = url
        if why_it_matters:
            result["why_it_matters"] = why_it_matters
        return result

    def _normalize_text(text: str) -> str:
        """Strip punctuation & collapse whitespace for fuzzy matching."""
        import re
        return re.sub(r'[^a-z0-9 ]', ' ', text.lower()).strip()

    def _normalize_words(text: str) -> list:
        """Return list of normalised words."""
        return _normalize_text(text).split()

    def _quote_in_transcript(transcript: str) -> bool:
        """Check if the quote plausibly comes from this transcript.
        Uses sliding window of consecutive words — robust against punctuation
        differences in auto-generated transcripts."""
        if not transcript:
            return False
        t_norm = _normalize_text(transcript)
        q_words = _normalize_words(quote_text)
        if len(q_words) < 4:
            return ' '.join(q_words) in t_norm
        # Sliding window: if any 5-consecutive-word sequence from the quote
        # appears in the transcript, it's a match
        window = min(5, len(q_words))
        for i in range(len(q_words) - window + 1):
            chunk = ' '.join(q_words[i:i + window])
            if chunk in t_norm:
                return True
        return False

    # Build person-matched source lists (only sources where the target is speaking)
    _pm_podcasts = [p for p in artifacts.podcasts if getattr(p, 'is_person_match', False)]
    _pm_videos = [v for v in artifacts.videos if getattr(v, 'is_person_match', False)]

    def _quote_in_person_transcript() -> bool:
        """Check if the quote exists in ANY person-matched transcript."""
        for p in _pm_podcasts:
            if _quote_in_transcript(p.transcript_text):
                return True
        for v in _pm_videos:
            if _quote_in_transcript(v.transcript_text):
                return True
        return False

    def _find_url_from_artifacts() -> str:
        """Find the source URL using multiple strategies.
        Only considers person-matched sources to avoid picking up
        unrelated videos (e.g. songs by bands with same company name)."""
        import re as _re
        _stop = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at',
                 'to', 'for', 'with', 'by', 'from', 'is', 'was', 'are'}

        # Strategy 1: Match quote word-sequences against person-matched transcripts
        for p in _pm_podcasts:
            if _quote_in_transcript(p.transcript_text):
                logger.info(f"Pull quote URL matched via transcript (podcast): {p.url}")
                return p.url or ""
        for v in _pm_videos:
            if _quote_in_transcript(v.transcript_text):
                logger.info(f"Pull quote URL matched via transcript (video): {v.url}")
                return v.url or ""

        # Strategy 2: Word-overlap between source attribution and person-matched titles
        if source_str:
            src_words = set(_normalize_words(source_str)) - _stop
            best_url = ""
            best_overlap = 0
            all_pm = [(a, 'podcast') for a in _pm_podcasts] + \
                     [(a, 'video') for a in _pm_videos]
            for art, kind in all_pm:
                title = (art.title or "").lower()
                title_words = set(_normalize_words(title)) - _stop
                if len(title_words) < 2:
                    continue
                overlap = len(src_words & title_words)
                if overlap >= 2 and overlap > best_overlap:
                    best_overlap = overlap
                    best_url = art.url or ""
            if best_url:
                logger.info(f"Pull quote URL matched via title overlap ({best_overlap} words): {best_url}")
                return best_url

        # Strategy 3: If only one person-matched source has a transcript, must be that
        transcript_sources = []
        for p in _pm_podcasts:
            if p.transcript_text:
                transcript_sources.append(p.url or "")
        for v in _pm_videos:
            if v.transcript_text:
                transcript_sources.append(v.url or "")
        if len(transcript_sources) == 1:
            logger.info(f"Pull quote URL matched via single-source fallback: {transcript_sources[0]}")
            return transcript_sources[0]

        return ""

    # Reject if source attribution mentions LinkedIn or article
    reject_keywords = ["linkedin", "article", "blog", "post"]
    if any(kw in source_str for kw in reject_keywords):
        logger.info(f"Pull quote rejected: source contains non-transcript keyword ({source_str[:80]})")
        return None

    # Accept if source mentions video/podcast/youtube/interview
    # BUT verify the quote actually comes from a person-matched source
    accept_keywords = ["youtube", "podcast", "interview", "keynote", "video", "panel", "talk", "conference"]
    if any(kw in source_str for kw in accept_keywords):
        if not _quote_in_person_transcript():
            logger.info(f"Pull quote rejected: not found in any person-matched transcript ({quote_text[:50]}...)")
            return None
        fallback_url = _find_url_from_artifacts() if not source_url else ""
        return _build_result(fallback_url)

    # No clear source — verify quote text exists in a person-matched transcript
    for p in _pm_podcasts:
        if _quote_in_transcript(p.transcript_text):
            return _build_result(p.url or "")
    for v in _pm_videos:
        if _quote_in_transcript(v.transcript_text):
            return _build_result(v.url or "")

    logger.info(f"Pull quote rejected: not found in any transcript ({quote_text[:60]}...)")
    return None


def _build_source_material(artifacts: CollectedArtifacts) -> str:
    """Build the source material block shared by both prompts.

    Source numbering: PODCAST 1, 2... → VIDEO N+1... → ARTICLE M+1...
    Each source includes pre-extracted signals (from signal_extraction step)
    followed by raw text for quote verification.
    """
    sections = []
    source_num = 0

    # Podcasts first
    for podcast in artifacts.podcasts:
        source_num += 1
        section = f"=== SOURCE {source_num}: Podcast Episode ===\n"
        section += f"Title: {podcast.title}\n"
        if podcast.podcast_title:
            section += f"Podcast: {podcast.podcast_title}\n"
        section += f"URL: {podcast.url}\n"
        if podcast.published_at:
            section += f"Published: {podcast.published_at}\n"
        if podcast.is_person_match:
            section += "Match type: PERSON-LEVEL (features the target person)\n"
        else:
            section += "Match type: COMPANY-LEVEL (features company leadership)\n"
        # Pre-extracted signals
        signals_block = _format_extracted_signals(podcast.extracted_signals)
        if signals_block:
            section += signals_block
        if podcast.transcript_text:
            section += f"Transcript:\n{podcast.transcript_text}\n"
        else:
            section += (
                "[No transcript available. Use title and description only.]\n"
                f"Description: {podcast.description}\n"
            )
        sections.append(section)

    # Then videos
    for video in artifacts.videos:
        source_num += 1
        section = f"=== SOURCE {source_num}: YouTube Video ===\n"
        section += f"Title: {video.title}\n"
        section += f"Channel: {video.channel_title}\n"
        section += f"URL: {video.url}\n"
        section += f"Published: {video.published_at}\n"
        if video.is_person_match:
            section += "Match type: PERSON-LEVEL (features the target person)\n"
        else:
            section += "Match type: COMPANY-LEVEL (features company leadership)\n"
        # Pre-extracted signals
        signals_block = _format_extracted_signals(video.extracted_signals)
        if signals_block:
            section += signals_block
        if video.transcript_text:
            section += f"Transcript:\n{video.transcript_text}\n"
        else:
            section += (
                "[No transcript available. Use title and description only.]\n"
                f"Description: {video.description}\n"
            )
        sections.append(section)

    # Then articles
    for article in artifacts.articles:
        source_num += 1
        section = f"=== SOURCE {source_num}: Article ===\n"
        section += f"Title: {article.title}\n"
        section += f"URL: {article.url}\n"
        # Pre-extracted signals
        signals_block = _format_extracted_signals(article.extracted_signals)
        if signals_block:
            section += signals_block
        section += f"Content:\n{article.text}\n"
        sections.append(section)

    if not sections:
        return "No sources were found."

    return "\n\n".join(sections)


def _call_gemini_sync(system_prompt: str, content_prompt: str, thinking_budget: int = 4096) -> str:
    """Synchronous Gemini call — used for QP sub-calls (fast, parallel)."""
    response = gemini_client.models.generate_content(
        model=settings.gemini_model,
        contents=content_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=settings.gemini_max_output_tokens,
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        ),
    )
    text = response.text
    finish = getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None
    logger.info(f"Gemini response: {len(text)} chars, finish_reason={finish}")
    if finish and str(finish) not in ('STOP', 'FinishReason.STOP', '1'):
        logger.warning(f"Gemini non-STOP finish: {finish} — response may be truncated")
    return text


def _call_claude_sync(system_prompt: str, content_prompt: str) -> str:
    """Synchronous Claude call — used for final Dossier synthesis (high quality)."""
    full_system = system_prompt + "\n\nIMPORTANT: Respond with ONLY the raw JSON object. No markdown, no code fences, no explanation."
    response = claude_client.messages.create(
        model=settings.claude_model,
        max_tokens=settings.claude_max_output_tokens,
        temperature=settings.claude_temperature,
        system=full_system,
        messages=[{"role": "user", "content": content_prompt}],
    )
    text = response.content[0].text
    stop = response.stop_reason
    logger.info(f"Claude response: {len(text)} chars, stop_reason={stop}")
    if stop != "end_turn":
        logger.warning(f"Claude non-end_turn stop: {stop} — response may be truncated")
    return text

def _parse_json_safe(raw: str):
    """Parse JSON from Gemini, handling common issues including truncation."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r'[\x00-\x1f\x7f]', lambda m: ' ' if m.group() not in ('\n', '\r') else m.group(), text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r'(?<!\\)\n', ' ', text)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Truncated JSON recovery — handles nested structures like {"signals": [{...}, ...
    logger.warning(f"Attempting truncated JSON recovery (input length: {len(cleaned)})")
    logger.warning(f"First 200 chars: {cleaned[:200]}")
    logger.warning(f"Last 200 chars: {cleaned[-200:]}")

    # Strategy 1: Find last complete object at any depth and close surrounding structures
    try:
        depth = 0
        last_complete_positions = {}  # depth -> last position where that depth closed
        in_string = False
        escape = False
        for i, ch in enumerate(cleaned):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ('{', '['):
                depth += 1
            elif ch in ('}', ']'):
                depth -= 1
                last_complete_positions[depth] = i
                if depth == 0:
                    # Full top-level object/array recovered
                    return json.loads(cleaned[:i + 1])

        # Top-level didn't close — try to close it manually
        # Find the last complete element inside the top-level structure
        if 1 in last_complete_positions:
            cut = last_complete_positions[1] + 1
            recovered = cleaned[:cut].rstrip().rstrip(',')
            # Close any open arrays and the outer object
            if cleaned.lstrip().startswith('{'):
                recovered += ']}'
            elif cleaned.lstrip().startswith('['):
                recovered += ']'
            try:
                result = json.loads(recovered)
                logger.info(f"Truncated JSON recovery succeeded at position {cut}")
                return result
            except json.JSONDecodeError:
                pass

        # Strategy 2: For {"signals": [...]} format, extract complete signal objects
        if 2 in last_complete_positions:
            cut = last_complete_positions[2] + 1
            recovered = cleaned[:cut].rstrip().rstrip(',')
            recovered += ']}'
            try:
                result = json.loads(recovered)
                logger.info(f"Truncated JSON recovery (deep) succeeded at position {cut}")
                return result
            except json.JSONDecodeError:
                pass

    except Exception as e:
        logger.warning(f"Truncated JSON recovery failed: {e}")

    return []


def _extract_list(parsed) -> list:
    """Extract a list from parsed JSON (may be dict with a key or direct list)."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("signals", "items", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        # Return any list value
        for v in parsed.values():
            if isinstance(v, list):
                return v
    return []


# ═══════════════════════════════════════════════════════════
#  Research Confidence Computation
# ═══════════════════════════════════════════════════════════

def _parse_date_lenient(date_str: str):
    """Try to parse a date string leniently. Returns datetime or None."""
    if not date_str:
        return None
    # Common formats from YouTube API and articles
    formats = [
        "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%b %d, %Y", "%B %d, %Y",
        "%d %b %Y", "%d %B %Y", "%Y-%m-%dT%H:%M:%S.%fZ",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    # Try extracting a year-month pattern
    m = re.search(r'(\d{4})[/-](\d{1,2})', date_str)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            pass
    return None


def _compute_research_confidence(
    artifacts: CollectedArtifacts,
    has_person_content: bool,
) -> ResearchConfidence:
    """Compute research confidence from source quality, with human-readable reasons."""
    now = datetime.now()
    six_months_ago = now - timedelta(days=180)

    podcast_person_matches = sum(1 for p in artifacts.podcasts if p.is_person_match)
    video_person_matches = sum(1 for v in artifacts.videos if v.is_person_match)
    direct_interviews = podcast_person_matches + video_person_matches
    total_sources = len(artifacts.podcasts) + len(artifacts.videos) + len(artifacts.articles)

    recent_count = 0
    for p in artifacts.podcasts:
        dt = _parse_date_lenient(p.published_at)
        if dt and dt >= six_months_ago:
            recent_count += 1
    for v in artifacts.videos:
        dt = _parse_date_lenient(v.published_at)
        if dt and dt >= six_months_ago:
            recent_count += 1
    for a in artifacts.articles:
        dt = _parse_date_lenient(a.published_date or "")
        if dt and dt >= six_months_ago:
            recent_count += 1

    # Build human-readable reasons list
    reasons = []
    if direct_interviews > 0:
        reasons.append(f"{direct_interviews} direct interview{'s' if direct_interviews != 1 else ''} found")
    else:
        reasons.append("No direct interviews found")

    if recent_count > 0:
        reasons.append(f"{recent_count} source{'s' if recent_count != 1 else ''} from the past 6 months")
    else:
        reasons.append("No recent sources (past 6 months)")

    if len(artifacts.podcasts) > 0:
        with_transcript = sum(1 for p in artifacts.podcasts if p.transcript_available)
        reasons.append(f"{with_transcript}/{len(artifacts.podcasts)} podcasts transcribed")

    if len(artifacts.videos) > 0:
        with_transcript = sum(1 for v in artifacts.videos if v.transcript_available)
        reasons.append(f"{with_transcript}/{len(artifacts.videos)} videos with transcripts")

    if len(artifacts.articles) > 0:
        reasons.append(f"{len(artifacts.articles)} article{'s' if len(artifacts.articles) != 1 else ''} analyzed")

    if direct_interviews >= 3 and recent_count >= 1:
        return ResearchConfidence(
            level="high",
            label=f"{direct_interviews} direct interviews with recent coverage",
            reasons=reasons,
        )
    elif direct_interviews >= 1 and total_sources >= 4:
        return ResearchConfidence(
            level="medium",
            label=f"{direct_interviews} interview(s) + {total_sources - direct_interviews} supporting sources",
            reasons=reasons,
        )
    else:
        return ResearchConfidence(
            level="low",
            label="No direct interviews; role-based synthesis only" if direct_interviews == 0
                  else f"Limited sources ({total_sources} total)",
            reasons=reasons,
        )


# ═══════════════════════════════════════════════════════════
#  Person Signal Strength Assessment
# ═══════════════════════════════════════════════════════════

def _assess_signal_strength(
    artifacts: CollectedArtifacts,
    request: ResearchRequest,
) -> str:
    """
    Assess how much direct person-level content we have.
    Returns: 'strong', 'moderate', or 'low'
    """
    if not request.target_name:
        return "strong"  # Company-only search, no person to assess

    direct_interviews = sum(
        1 for v in artifacts.videos
        if v.is_person_match and v.transcript_available
    )
    direct_interviews += sum(
        1 for p in artifacts.podcasts
        if p.is_person_match and p.transcript_available
    )
    person_match_videos = sum(1 for v in artifacts.videos if v.is_person_match)
    person_match_podcasts = sum(1 for p in artifacts.podcasts if p.is_person_match)

    # Check if person is mentioned substantively in articles
    person_name_lower = request.target_name.lower()
    person_articles = 0
    for a in artifacts.articles:
        if person_name_lower in a.text.lower():
            person_articles += 1

    if direct_interviews >= 1:
        return "strong"
    elif person_match_videos >= 1 or person_match_podcasts >= 1 or person_articles >= 2:
        return "moderate"
    else:
        return "low"


# ═══════════════════════════════════════════════════════════
#  CALL 1: Quick Prep signals + Executive Orientation
# ═══════════════════════════════════════════════════════════

def _build_quick_prep_system(request: ResearchRequest, has_person_content: bool) -> str:
    lines = [
        "You are an EXECUTIVE INTELLIGENCE engine for B2B sales reps.",
        "",
        "YOUR JOB: Reveal how this executive THINKS, what PRESSURES they're under,",
        "and where they're VULNERABLE. NOT company news summaries.",
        "",
        "The bar: An SDR reads this and thinks 'I understand how this person thinks",
        "and where they're vulnerable' — NOT 'I know what this company announced.'",
        "",
        "PERSON-FIRST RULE (MANDATORY when target person is specified):",
        "The individual MUST always be the primary object of analysis.",
        "Company context exists ONLY to illuminate the PERSON's world.",
        "Every signal and every orientation line must be ABOUT this person.",
        "NEVER let company strategy dominate — frame everything through the person's lens.",
        "",
        "SYNTHESIS PROCESS:",
        "1. Extract all signals from ALL sources",
        "2. For each signal, ask: What does this reveal about how they think?",
        "   What pressure are they under? Where are they vulnerable?",
        "3. Cluster overlapping themes across sources",
        "4. Rank by: strategic insight value > frequency > recency",
        "5. Generate final output from ranked clusters",
        "",
        f"TARGET COMPANY: {request.target_company}",
    ]
    if request.target_name:
        lines.append(f"TARGET PERSON: {request.target_name}")
    if request.target_title:
        lines.append(f"PERSON TITLE: {request.target_title}")

    if not has_person_content and request.target_name:
        lines.append("")
        lines.append("WARNING: No direct interviews with this person were found.")
        lines.append("Derive intelligence from company-level sources and role context.")
        lines.append(f"Focus on what someone in the role of {request.target_title or 'executive'}")
        lines.append("at this company would be navigating. Label role-inferred insights as such.")
        lines.append("Never fabricate quotes. Only use verifiable company statements.")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 1: LEAD ORIENTATION (Key Pressure + 5 dimensions)
═══════════════════════════════════════

FIRST, write the Key Pressure — the single most important structural tension this
person is navigating right now. This is the FIRST thing an AE reads. It should name
the tension specifically, not generically. Key Pressure should answer: what is the
one thing keeping this person up at night that an AE could credibly speak to?
2 sentences maximum.

Example Key Pressure:
✓ "He's managing the structural friction of scaling a $16B+ multi-product surface area while keeping application-specific and platform teams from pulling in opposite directions. Any vendor that adds integration complexity makes his job harder — any vendor that reduces it becomes an ally."
✓ "She's managing the psychological weight of asking studio owners to abandon a decade of Mindbody data — her entire sales motion is built around not traumatizing the customer. If your implementation story is even slightly unclear she'll walk."

THEN, write 5 Lead Orientation dimensions. Each dimension should:
1. Open with the IMPLICATION for an AE — start with what this means for how you sell,
   not what personality trait it describes
2. Support it with one specific piece of evidence from the sources — a quote, a decision
   they made, a pattern across multiple sources
3. Be written as 2 sentences maximum — no bullet labels, no headers

Examples:
✓ "He'll reject any pitch that positions your tool as a reporting layer or oversight mechanism — he's on record saying finance should be operating fabric, not auditor. Come in as a partner to his GTM motion, not a check on it."
✓ "She measures her own team's performance by technology adoption rates, not quota attainment alone — she'll evaluate your tool the same way. Have a concrete adoption and rollout plan ready before she asks."

RULES:
- Key Pressure: 2 sentences max, specific structural tension, leads the section
- Each dimension: 2 sentences max, AE implication first, evidence second
- No category labels or headers on dimensions
- No two dimensions may address the same strategic area
- NEVER fabricate or speculate without evidence
REJECTION FRAME: At least one of the five dimensions MUST be a rejection frame —
what this person is likely to reject, dismiss, or have zero patience for.
This dimension must:
- Be written as a direct statement: "He'll reject…" / "She has no patience for…"
- Cite a specific quote, decision, or pattern
- Avoid generic language. Never hedge.
- Be the sharpest, most direct line in the section.
If no rejection pattern is clearly supported by evidence, do not fabricate one.""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 2: INTEL (5 cards, ranked by tactical value)
═══════════════════════════════════════

Generate 5 Intel cards for this person. Each card should:

1. State the insight in plain language — no "reveals" formula, no category labels
   in the text. Just say what it means directly.

2. End with one sentence that makes the tactical implication explicit — what does
   someone walking into a meeting with this person actually DO with this signal?
   Never leave the insight without a so-what.

3. Rank the cards by tactical value — the highest leverage signal comes first,
   the most contextual last. The ordering itself is part of the intelligence.
CARD 1 ANCHOR RULE: Card 1 must be the strongest available tactical signal.
It must be one of:
- A direct quote-based signal that shows how this person thinks, or
- A behavior-backed signal (decision, repeated action, measurable pattern), or
- A cross-source synthesis signal (if distinct from the Core Read thesis)

Card 1 must:
- Add new leverage beyond the Executive Snapshot and Core Read
- Be immediately usable in a live conversation
- NEVER be company news, funding announcements, headcount data, or LinkedIn bio restatements
- NEVER apologize or state that signal depth is weak
If no cross-source synthesis exists, default to the strongest quote-backed or behavior-backed signal.

4. HARD RULE: No more than 2 cards from the same source type (video/article/podcast).
   If 3+ sources are the same type, skip the weakest and find signal from a different type.

NEVER use the word "reveals" more than once across all 5 cards.

PERSON-FOCUS: Every card is about how this PERSON thinks or decides — not what
their company shipped. Start with He/She, never the company name.
  WRONG: "NVIDIA is aggressively moving into the Software-as-a-Model space..."
  RIGHT: "He’s betting his next growth cycle on open-sourcing not just models but training data..."

WORD LIMIT: Max 40 words per card. Cut filler, keep specifics.

TACTICAL PRECISION: The so-what must be specific enough that it could ONLY apply
to this person. "Position your offerings" = too generic, fails the test.
"Come with conservative projections you can defend, not optimistic ones" = specific.

Examples:
✅ "His 2008 Goldman experience left him permanently skeptical of growth narratives — he's been burned by exuberance before and will probe your numbers harder than most. Come with conservative projections you can defend, not optimistic ones you hope he accepts."
✅ "Displacing Mindbody isn't just a technology switch for her customers, it's a psychological event. She's built her entire sales motion around not repeating the trauma of bad migrations — your implementation story needs to be airtight before she'll consider price."
✅ "He's launched 50+ brands but his pattern is always '0 to 1' — he gets restless with operational maintenance. If your pitch is about long-term management, reframe it around what it unlocks for the next build."

SIGNAL CATEGORIES (used for icon selection only, NOT shown in text):
🚀 GROWTH — expansion, scaling, market entry
💰 MARKET — partnerships, funding, competitive positioning
🔧 PRODUCT — tech priorities, product strategy, operational focus
🎯 BACKGROUND — previous roles, philosophy, approach
⚖️ TENSION — strategic trade-offs, competing priorities
🚨 CHALLENGE — stated problems, pain points, risks

REQUIRED COMPOSITION (5 signals, each a DIFFERENT category):
- Each signal must address a different strategic dimension
- NEVER repeat the same category

CRITICAL RULES:
1. Every signal must end with a tactical so-what for the AE
2. Be specific with numbers, timelines, stakes
3. Max 2-3 sentences per signal. Plain prose, no formulas.
4. Each quote must be 15-40 words from source material
5. DEDUPLICATE: same fact from multiple sources = ONE signal citing best source
6. Include source title, URL, date, and timestamp (MM:SS) for videos/podcasts
7. Diversify source types across cards (podcast, video, article)""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 2.5: PULL QUOTE (1 standout direct quote + tactical context)
═══════════════════════════════════════

Select the quote that reveals the most about how this person thinks, what they
value, or what they're reacting against — not the most impressive-sounding one.
Recency matters but insight matters more.

Then add a "why_it_matters" line — 1-2 sentences that translate the quote into a
tactical implication for the AE. This should complete the thought: "This tells you
that in your meeting, you should..." Always write from the AE's perspective —
what do they DO with this information.

RULES:
- Must be VERBATIM from a video transcript or podcast transcript (15-40 words)
- STRONGLY prefer quotes from podcast/interview transcripts over any other source
- NEVER use text from articles or LinkedIn posts — only spoken words from audio/video
- Choose a quote that reveals their thinking, philosophy, values, or pressure
- Pick quotes where the person speaks with conviction or candor — not generic platitudes
- NEVER select quotes containing generic business language like "customer-obsessed,"
  "results-driven," or "move the needle" unless surrounding context makes them
  tactically useful. If the only available quotes are generic, return null.
- Include the source as: "Source Title - Platform - Date - Timestamp"
- If no podcast/video transcripts are available, return null
- why_it_matters must be 1-2 sentences max, always from the AE's perspective

Examples of good why_it_matters:
✓ "He doesn't want a vendor pitch — he wants to know how you become part of his operational infrastructure. Lead with integration and workflow impact, not features."
✓ "He'll respect you if you know your craft and can articulate why you made the choices you made. Don't oversell — he'll see through it immediately."
✓ "She's been the person who has to make tools actually work for reps. She'll be evaluating whether you've thought through adoption, not just capability."

Output: {"quote": "verbatim text...", "source": "Source Title - Platform - Date - MM:SS", "why_it_matters": "1-2 sentence tactical implication for the AE"}
or null if no direct quotes available.""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 3: EXECUTIVE SNAPSHOT (2-3 sentences)
═══════════════════════════════════════

Based on everything gathered about this person, write a 2-3 sentence
Executive Snapshot that answers three things in flowing prose — not
as labeled sections:

1. What is this person reacting against or trying to prove based on
   their career trajectory?
2. What do they care about most in their current role based on how
   they talk publicly?
3. What does that mean tactically for how someone should sell to or
   engage with them?

Avoid generic descriptors like "passionate," "results-driven," or
"customer-obsessed." Every sentence should contain information an AE
couldn't get from reading the LinkedIn profile themselves. If you
can't find enough signal to answer all three, say so — do not
fabricate or generalize.

Examples:
✓ "Former CFO turned ecosystem builder — Adil thinks in systems and friction points, not features. He's been inside Rippling's operational engine long enough to have zero patience for tools that add complexity. Sell to the outcome he's accountable for: making founders operationally dangerous faster."
✓ "Gia has lived through bad software migrations and built her entire philosophy around not repeating them. She'll evaluate you on how realistic and role-specific your implementation process is before she ever considers price. Come in with a clear onboarding story or don't come in at all."
✓ "Creative director who's learned to speak in data to get what he wants — he bridges brand intuition with business rigor because he's had to. He's not a feelings-first creative, he's a systems thinker in a creative's clothing. Pitch him on outcomes and craft simultaneously or you'll lose him on one dimension."

RULES:
- 2-3 sentences, flowing prose
- Skip the person's name (it's already shown above)
- Must reveal HOW to engage, not just WHO they are
- Every sentence must contain signal beyond their LinkedIn profile
- Do NOT repeat orientation lines verbatim
- If insufficient signal, explicitly flag it rather than fabricating""")

    today = datetime.utcnow()
    cutoff = today - timedelta(days=90)
    lines.append("")
    lines.append(f"""═══════════════════════════════════════
PART 4: RECENT MOVES (last 90 days, max 4 events)
═══════════════════════════════════════

TODAY'S DATE: {today.strftime('%B %d, %Y')}
CUTOFF DATE: {cutoff.strftime('%B %d, %Y')} (90 days ago)

Identify up to 4 recent activities involving this person or their company
that occurred BETWEEN {cutoff.strftime('%B %d, %Y')} and {today.strftime('%B %d, %Y')}.

For each recent move, do four things:
1. Label the signal tier:
   - [THEIR WORDS]: They authored it, appeared in it, or were directly quoted
   - [THEIR ATTENTION]: They shared, commented on, or engaged with it on LinkedIn
   - [COMPANY NEWS]: Organizational announcement, funding, press coverage about their company
   NEVER present a [THEIR ATTENTION] signal as if it were [THEIR WORDS].

2. STATE what happened factually in one line with source and date

3. Add a "signal" line — what does this activity tell you about what's top of mind
   for this person RIGHT NOW

4. Add a "hook" line — one sentence on how an AE can reference this naturally in
   the first 5 minutes of a meeting without sounding like they Googled them

Examples:
{{"tier": "THEIR WORDS", "event": "Appeared on podcast discussing quantitative methodology for creative differentiation", "date": "January 2026", "signal": "He's evangelizing a framework he's built — he wants people to understand his methodology.", "hook": "I listened to your conversation with Luke Nelmes — the ecosystem mapping framework was different from how most creative directors talk about differentiation.", "source_url": "https://...", "source_title": "Podcast Title"}}
{{"tier": "THEIR ATTENTION", "event": "Shared conversation between Rippling CEO and Cursor CEO about scaling and AI infrastructure", "date": "December 2025", "signal": "He's actively thinking about how AI changes the cost of growth — this isn't abstract for him, it's operational.", "hook": "I noticed you shared that conversation between Parker and the Cursor CEO — what made you want to put that in front of your network?", "source_url": "https://...", "source_title": "LinkedIn"}}

RULES:
- ONLY include events dated AFTER {cutoff.strftime('%B %d, %Y')} — reject anything older
- If no events from the last 90 days exist in sources, return: [{{"tier": "NONE", "event": "No recent public activity found — rely on company-level signals instead", "date": "", "signal": "", "hook": "", "source_url": "", "source_title": ""}}]
- Never fabricate dates or events
- Signal and hook must be 1 sentence each
- Tier must be exactly one of: THEIR WORDS, THEIR ATTENTION, COMPANY NEWS, NONE""")

    lines.append("")
    lines.append("""
═══════════════════════════════════════
CORE READ (thesis statement)
═══════════════════════════════════════

Generate one Core Read statement for this person.
This appears between Recent Moves and Lead Orientation and serves as the thesis.

It must:
1. Begin with: "Across interviews, public statements, and company actions, a consistent pattern emerges:"
2. Identify a specific, evidence-backed tension, strategic posture, or recurring
   decision pattern visible in at least two distinct source types.
3. Focus on behavior and strategic posture — not personality speculation.
4. Avoid psychological interpretation unless directly supported by quotes or actions.
5. Be one to two sentences maximum.

If no genuine cross-source pattern exists, write:
"Sources present a consistent but narrow view — signals below are directional rather than cross-validated."

═══════════════════════════════════════
PART 5: CONVERSATION ANGLES (1 primary + 2 alternatives)
═══════════════════════════════════════

Generate one primary Conversation Angle and two alternative angles for this person.

The primary angle should be the single highest-leverage conversation starter based
on cross-source synthesis — something specific enough that the prospect will be
genuinely surprised the AE knows it.

Each alternative angle should use a DIFFERENT conversational approach — not just a
different topic. For example:
- One might be a direct question
- One might be an observation that invites pushback
- One might be a vulnerability-based opener that shows the AE did real homework

CRITICAL: Never use the same sentence structure twice across the three angles.
The formula "Reference X — ask how Y" can be used ONCE maximum.

Each angle should be 1-2 sentences. No internal labels like "Quantitative Creative"
or "Mental Recovery" — those are internal logic, not what the AE says or thinks.

RULES:
- 1-2 sentences per angle, natural prose
- Each angle uses a DIFFERENT conversational mode (curiosity, validation, tension, etc.)
- Reference specific facts/signals from the research
- The primary angle must be the single strongest opener""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
OUTPUT FORMAT (JSON object):
═══════════════════════════════════════
{
  "prior_role": "CMO, Impossible Foods" or null,
  // ^ MUST be their IMMEDIATELY PREVIOUS role (the one right before their current position).
  // NOT a role from 2-3 positions ago. If their current role is "CEO at Acme" and their
  // LinkedIn shows "VP Sales at BigCorp" right before that, use "VP Sales, BigCorp".
  // If they are currently at their ONLY known role, return null.
  "executive_summary": "2-3 sentence Executive Snapshot (see rules above)",
  "pull_quote": {
    "quote": "It's to me truly unacceptable that we continue to waste food at this scale...",
    "source": "Jordan Schenck | Flashfood - YouTube - Apr 2, 2025 - 04:45",
    "source_url": "https://youtube.com/watch?v=abgKopCIDOY",
    "why_it_matters": "She leads with mission, not margin — if you pitch cost savings first, you'll lose her. Frame your value in terms of waste reduction impact and she'll lean in."
  },
  "core_read": "Across interviews, public statements, and company actions, a consistent pattern emerges: he treats every strategic decision as an infrastructure bet — the through-line is building irreversible platform lock-in while appearing to give everything away.",
  "executive_orientation": {
    "key_pressure": "She's managing the psychological weight of asking studio owners to abandon a decade of Mindbody data — her entire sales motion is built around not traumatizing the customer. If your implementation story is even slightly unclear she'll walk.",
    "bullets": [
      "She measures her own team's performance by technology adoption rates, not quota attainment alone — she'll evaluate your tool the same way. Have a concrete adoption and rollout plan ready before she asks.",
      "He'll reject any pitch that positions your tool as a reporting layer or oversight mechanism — he's on record saying finance should be operating fabric, not auditor. Come in as a partner to his GTM motion, not a check on it.",
      "Don't lead with ROI projections — she's publicly stated she cares more about realistic migration timelines than cost savings. Show her your implementation blueprint first.",
      "She treats middle management as the make-or-break layer for any new tool deployment. If you can't articulate how your onboarding works for team leads specifically, you'll stall.",
      "His public content reveals a pattern of turning operational failures into teachable moments. Reference a specific challenge you've helped similar companies navigate — he'll engage with vulnerability over polish."
    ]
  },
  "recent_moves": [
    {"tier": "THEIR WORDS", "event": "Keynoted Davos 2026 on AI and labor", "date": "January 2026", "signal": "Publicly positioning AI as a labor story.", "hook": "Your Davos talk reframed how I think about workforce planning.", "source_url": "https://...", "source_title": "Davos 2026"},
    {"tier": "COMPANY NEWS", "event": "Unveiled new self-driving car models", "date": "January 2026", "signal": "Betting on automotive AI beyond data centers.", "hook": "The autonomous vehicle push is a big new surface area.", "source_url": "https://...", "source_title": "NVIDIA Live"}
  ],
  "opening_moves": [
    {"angle": "Primary", "suggestion": "Lead with his 'How I Screwed This Up' series — ask about the most common operational drag he sees in pre-IPO startups today."},
    {"angle": "Alternative", "suggestion": "Tell him you noticed Rippling just crossed $570M ARR and ask what operational bets he made 18 months ago that actually got them there — let him tell you what he's proud of."},
    {"angle": "Alternative", "suggestion": "Most vendors come in talking about efficiency. Open instead by acknowledging the tension he's described publicly — that platform teams and application teams want different things — and ask how he thinks about that."}
  ],
  "signals": [
    {
      "category": "GROWTH",
      "signal": "She's pushing a 25% store expansion (2,000 to 2,500) just 8 months into her CEO tenure. That's an aggressive bet with serious execution exposure — ask what's breaking at that growth rate and position your solution around the operational strain she's likely feeling right now.",
      "quote": "It's over 2,000 and on any given day probably getting closer to 2,500...",
      "source": {
        "type": "video",
        "title": "Jordan Schenck CEO Interview",
        "url": "https://youtube.com/watch?v=abgKopCIDOY",
        "timestamp": "08:30",
        "date": "Apr 2, 2025"
      }
    }
  ]
}

Return a JSON object with "prior_role", "executive_summary", "pull_quote", "core_read", "executive_orientation", "recent_moves", "opening_moves", and "signals".
If no quality signals found, return {"prior_role": null, "executive_summary": "...", "pull_quote": null, "executive_orientation": {"bullets": [], "key_pressure": ""}, "recent_moves": [], "opening_moves": [], "signals": []}.""")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  CALL 1b: Quick Prep — LOW SIGNAL MODE
# ═══════════════════════════════════════════════════════════

def _build_quick_prep_system_low_signal(request: ResearchRequest) -> str:
    """Build Quick Prep prompt for low-signal executives (no direct interviews)."""
    lines = [
        "You are an EXECUTIVE INTELLIGENCE engine for B2B sales prep.",
        "",
        "CRITICAL CONTEXT: Very limited direct public signal exists for this person.",
        "There are NO direct interviews, podcast appearances, or keynotes available.",
        "",
        "YOUR JOB: Build a useful profile by combining:",
        "1. Whatever identity/role information exists in sources",
        "2. Role-based inference (clearly labeled as inferred)",
        "3. Company context as SUPPORTING evidence (never leading)",
        "",
        "ABSOLUTE RULES:",
        "- The PERSON is always the primary subject, never the company",
        "- Label all inferences: 'Inferred from role context' or 'Based on company positioning'",
        "- NEVER fabricate quotes or attribute statements to this person",
        "- NEVER speculate on psychology, personality, or decision patterns without evidence",
        "- Be honest about what you don't know",
        "- Company context supports the person analysis — it does NOT replace it",
        "",
        f"TARGET PERSON: {request.target_name}",
        f"TARGET COMPANY: {request.target_company}",
    ]
    if request.target_title:
        lines.append(f"PERSON TITLE: {request.target_title}")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 1: LEAD ORIENTATION (Key Pressure + 5 dimensions — LOW SIGNAL MODE)
═══════════════════════════════════════

FIRST, write the Key Pressure — the most likely structural tension this person is
navigating based on role and company context. Acknowledge limited signal honestly.
2 sentences maximum.

THEN, write up to 5 Lead Orientation dimensions with honest role-based inference.
Each dimension should:
1. Open with the implication for an AE — what this means for how you sell
2. Support with whatever evidence is available, labeling inferences clearly
3. Be written as 2 sentences maximum — no bullet labels, no headers

RULES:
- Key Pressure: 2 sentences max, role-inferred if no direct evidence
- Each dimension: 2 sentences max, AE implication first
- Acknowledge limited signal where appropriate
- NEVER fabricate or speculate without evidence
REJECTION FRAME: If evidence supports it, one dimension should describe what this
person is likely to reject or dismiss. Written as "He'll reject…" / "She has no
patience for…" — cite specific evidence. Do not fabricate if unsupported.""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 2: INTEL (5 cards — LOW SIGNAL MODE)
═══════════════════════════════════════

With limited person-level content, generate 5 Intel cards that:
1. LEAD with what we know about the PERSON (even if minimal)
2. Connect role context to likely operational focus
3. End each with a tactical so-what for the AE
4. CLEARLY LABEL what is observed vs inferred

Frame EVERY signal around the person, even when using company data.
Start every card with He/She, never the company name.

WORD LIMIT: Max 40 words per card. Plain prose, no filler.
HARD RULE: No more than 2 cards from the same source type.
TACTICAL PRECISION: Each so-what must be specific to THIS person —
generic advice like "position your solution" fails the test.

CARD 1 ANCHOR RULE: Card 1 must be the strongest signal available, even with limited data.
Prefer a direct quote or observable behavior over role-inferred context.
Card 1 must add leverage beyond the Executive Snapshot — never restate the bio.
NEVER apologize or state that signal depth is weak.
Rank by tactical value.

SIGNAL CATEGORIES (for icon selection only):
🎯 BACKGROUND — identity, role context, career positioning
🔧 PRODUCT — likely operational/product focus based on role
💰 MARKET — market context this person operates in
🚨 CHALLENGE — role-typical pressures and challenges
⚖️ TENSION — likely tensions and trade-offs in this role

For quote field: use a relevant company quote if available, or write:
  "No direct quote available — inferred from role context"
For inferred signals, set source type to "article" and use the most relevant company source.""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 3: EXECUTIVE SNAPSHOT (2-3 sentences — LOW SIGNAL MODE)
═══════════════════════════════════════

Based on whatever limited information is available, write a 2-3 sentence
Executive Snapshot that answers in flowing prose:

1. What can we infer this person is reacting against or trying to prove
   based on their career trajectory?
2. What do they likely care about most based on their role and company context?
3. What does that mean tactically for how someone should approach them?

If you can't find enough signal to answer all three confidently, say so —
do not fabricate or generalize. Acknowledge limited signal honestly.

RULES:
- 2-3 sentences, flowing prose
- Acknowledge limited signal where applicable
- Do NOT speculate beyond what sources support
- If insufficient signal, explicitly flag it rather than fabricating""")

    today_ls = datetime.utcnow()
    cutoff_ls = today_ls - timedelta(days=90)
    lines.append("")
    lines.append(f"""═══════════════════════════════════════
PART 4: RECENT MOVES (last 90 days — LOW SIGNAL MODE)
═══════════════════════════════════════

TODAY'S DATE: {today_ls.strftime('%B %d, %Y')}
CUTOFF: {cutoff_ls.strftime('%B %d, %Y')}

Same rules as standard mode. Up to 4 events AFTER {cutoff_ls.strftime('%B %d, %Y')}.
Each move must include: tier (THEIR WORDS / THEIR ATTENTION / COMPANY NEWS), event, date,
signal (what's top of mind), hook (how AE references it naturally).
If no events exist, return: [{{"tier": "NONE", "event": "No recent public activity found — rely on company-level signals instead", "date": "", "signal": "", "hook": "", "source_url": "", "source_title": ""}}]
Never fabricate dates or events.""")

    lines.append("")
    lines.append("""
═══════════════════════════════════════
CORE READ (thesis statement — LOW SIGNAL)
═══════════════════════════════════════

Generate one Core Read statement based on available signal.
With limited sources, default to:
"Sources present a consistent but narrow view — signals below are directional rather than cross-validated."

Only write a pattern-based thesis if you genuinely see evidence from 2+ source types.

═══════════════════════════════════════
PART 5: CONVERSATION ANGLES (1 primary + 2 alternatives — LOW SIGNAL MODE)
═══════════════════════════════════════

Generate one primary and two alternative conversation angles even with limited signal.
Frame as genuine discovery approaches (we don't have deep signal, so ASK).

Each angle should use a DIFFERENT conversational approach — not just a different topic.
Never use the same sentence structure twice. 1-2 sentences per angle.

RULES:
- 1-2 sentences per angle, natural prose
- Each angle uses a different conversational mode
- Reference company context the person operates in
- The primary angle should be the strongest available opener""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
OUTPUT FORMAT (JSON object):
═══════════════════════════════════════
{
  "prior_role": null,
  // ^ Their IMMEDIATELY PREVIOUS role (right before current position), or null if unknown.
  // Must be the role directly before their current one — never a role from 2+ positions ago.
  "executive_summary": "2-3 sentence Executive Snapshot acknowledging limited signal",
  "pull_quote": null,
  "core_read": "Sources present a consistent but narrow view — signals below are directional rather than cross-validated.",
  "executive_orientation": {
    "key_pressure": "Limited direct signal — based on role context, the most likely structural tension is [specific role-typical pressure]. Any vendor approaching should [implication].",
    "bullets": [
      "Limited direct signal on this person's preferences — based on their [title] role, they're likely evaluating tools through [specific lens]. Come prepared with [approach].",
      "Role context suggests emphasis on [priorities] — frame your pitch around [implication] rather than [alternative].",
      "Without direct content from this person, lead with genuine discovery about their priorities at [company] rather than assumptions."
    ]
  },
  "recent_moves": [],
  "opening_moves": [
    {"angle": "Primary", "suggestion": "Lead with their role at [Company] and a specific company development — ask how it's shaped their priorities since joining."},
    {"angle": "Alternative", "suggestion": "You could also surface a competitive observation about [Company]'s market and ask how they think about differentiation from their seat."},
    {"angle": "Alternative", "suggestion": "Given limited public signal, open with genuine curiosity about what drew them to [Company] at this stage — the answer will tell you what they value."}
  ],
  "signals": [
    {
      "category": "BACKGROUND",
      "signal": "As [Title] at [Company], she joined right as [recent event] was unfolding. That timing suggests she was brought in specifically to manage [challenge] — lead with that context and ask what her first 90 days taught her.",
      "quote": "No direct quote available — inferred from role context",
      "source": {
        "type": "article",
        "title": "Source title",
        "url": "https://...",
        "timestamp": null,
        "date": "..."
      }
    }
  ]
}

Return a JSON object with "prior_role", "executive_summary", "pull_quote", "core_read", "executive_orientation", "recent_moves", "opening_moves", and "signals".
EVERY signal must be framed around the person, never the company alone.""")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Quick Prep Sub-Prompts (for parallel execution)
# ═══════════════════════════════════════════════════════════

def _qp_target_block(request: ResearchRequest) -> str:
    """Shared target context block for all QP sub-prompts."""
    lines = [f"TARGET COMPANY: {request.target_company}"]
    if request.target_name:
        lines.append(f"TARGET PERSON: {request.target_name}")
    if request.target_title:
        lines.append(f"PERSON TITLE: {request.target_title}")
    return "\n".join(lines)


def _build_qp_sub_a_system(
    request: ResearchRequest, has_person_content: bool, low_signal: bool = False
) -> str:
    """Sub-prompt A: Executive Snapshot + Recent Moves + prior_role (fast, ~3-5s)."""
    lines = [
        "You are an EXECUTIVE INTELLIGENCE engine for B2B sales reps.",
        "",
        _qp_target_block(request),
    ]

    if low_signal and request.target_name:
        lines.append("")
        lines.append("CRITICAL CONTEXT: Very limited direct public signal exists for this person.")
        lines.append("Be honest about confidence levels. Label inferences clearly.")
    elif not has_person_content and request.target_name:
        lines.append("")
        lines.append("WARNING: No direct interviews with this person were found.")
        lines.append("Derive intelligence from company-level sources and role context.")

    lines.append("")
    if low_signal:
        lines.append("""═══════════════════════════════════════
TASK 1: EXECUTIVE SNAPSHOT (2-3 sentences — LOW SIGNAL)
═══════════════════════════════════════

Based on whatever limited information is available, write a 2-3 sentence
Executive Snapshot answering in flowing prose:
1. What can we infer about their career trajectory?
2. What do they likely care about based on role and company context?
3. What does that mean tactically for how to engage them?

Acknowledge limited signal honestly. Do NOT fabricate or generalize.

RULES:
- 2-3 sentences, flowing prose
- Acknowledge limited signal where applicable
- If insufficient signal, explicitly flag it""")
    else:
        lines.append("""═══════════════════════════════════════
TASK 1: EXECUTIVE SNAPSHOT (2-3 sentences)
═══════════════════════════════════════

Based on everything gathered, write a 2-3 sentence Executive Snapshot
answering in flowing prose — not as labeled sections:
1. What is this person reacting against or trying to prove?
2. What do they care about most based on how they talk publicly?
3. What does that mean tactically for how to sell to or engage them?

Avoid generic descriptors like "passionate" or "results-driven."
Every sentence should contain information beyond their LinkedIn profile.
If insufficient signal, say so — do not fabricate.

RULES:
- 2-3 sentences, flowing prose
- Must reveal HOW to engage, not just WHO they are
- Do NOT repeat orientation lines verbatim""")

    today = datetime.utcnow()
    cutoff = today - timedelta(days=90)
    lines.append("")
    lines.append(f"""═══════════════════════════════════════
TASK 2: RECENT MOVES (last 90 days, max 4 events)
═══════════════════════════════════════

TODAY'S DATE: {today.strftime('%B %d, %Y')}
CUTOFF DATE: {cutoff.strftime('%B %d, %Y')} (90 days ago)

Identify up to 4 recent activities BETWEEN {cutoff.strftime('%B %d, %Y')} and {today.strftime('%B %d, %Y')}.

For each, include:
- tier: THEIR WORDS / THEIR ATTENTION / COMPANY NEWS
- event: what happened (1 line)
- date: Month Year
- signal: what this tells you about what's top of mind (1 sentence)
- hook: how an AE references this naturally (1 sentence)
- source_url and source_title

If no events, return: [{{"tier": "NONE", "event": "No recent public activity found", "date": "", "signal": "", "hook": "", "source_url": "", "source_title": ""}}]
Never fabricate dates or events.""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
TASK 3: PRIOR ROLE
═══════════════════════════════════════

Identify their IMMEDIATELY PREVIOUS role (the one right before their current position).
NOT a role from 2-3 positions ago. If unknown from sources, return null.""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
OUTPUT FORMAT (JSON object):
═══════════════════════════════════════
{
  "prior_role": "CMO, Impossible Foods",
  "executive_summary": "2-3 sentence Executive Snapshot",
  "recent_moves": [
    {"tier": "THEIR WORDS", "event": "Keynoted Davos 2026 on AI and labor", "date": "January 2026", "signal": "Publicly positioning AI as a labor transformation story.", "hook": "Your Davos talk reframed how I think about workforce planning.", "source_url": "https://...", "source_title": "Davos 2026"}
  ]
}

If no recent moves, return "recent_moves": [].
If prior role unknown, return "prior_role": null.""")

    return "\n".join(lines)


def _build_qp_sub_b_system(
    request: ResearchRequest, has_person_content: bool, low_signal: bool = False
) -> str:
    """Sub-prompt B: Signals + Orientation + Opening Moves (heavy analysis, ~10-15s)."""
    lines = [
        "You are an EXECUTIVE INTELLIGENCE engine for B2B sales reps.",
        "",
        "YOUR JOB: Reveal how this executive THINKS, what PRESSURES they're under,",
        "and where they're VULNERABLE. NOT company news summaries.",
        "",
        "The bar: An SDR reads this and thinks 'I understand how this person thinks",
        "and where they're vulnerable' — NOT 'I know what this company announced.'",
        "",
        "PERSON-FIRST RULE (MANDATORY when target person is specified):",
        "The individual MUST always be the primary object of analysis.",
        "Company context exists ONLY to illuminate the PERSON's world.",
        "Every signal and every orientation line must be ABOUT this person.",
        "",
        "SYNTHESIS PROCESS:",
        "1. Extract all signals from ALL sources",
        "2. For each signal, ask: What does this reveal about how they think?",
        "   What pressure are they under? Where are they vulnerable?",
        "3. Cluster overlapping themes across sources",
        "4. Rank by: strategic insight value > frequency > recency",
        "5. Generate final output from ranked clusters",
        "",
        _qp_target_block(request),
    ]

    if low_signal and request.target_name:
        lines.append("")
        lines.append("CRITICAL CONTEXT: Very limited direct public signal exists for this person.")
        lines.append("There are NO direct interviews, podcast appearances, or keynotes available.")
        lines.append("Build intelligence by combining role/title inference (clearly labeled) with company context.")
        lines.append("Label all inferences: 'Inferred from role context' or 'Based on company positioning'")
    elif not has_person_content and request.target_name:
        lines.append("")
        lines.append("WARNING: No direct interviews with this person were found.")
        lines.append("Derive intelligence from company-level sources and role context.")
        lines.append(f"Focus on what someone in the role of {request.target_title or 'executive'}")
        lines.append("at this company would be navigating. Label role-inferred insights as such.")

    # ── CORE READ ──
    lines.append("")
    if low_signal:
        lines.append("""═══════════════════════════════════════
CORE READ (thesis statement — LOW SIGNAL)
═══════════════════════════════════════

Generate one Core Read statement. With limited sources, default to:
"Sources present a consistent but narrow view — signals below are directional rather than cross-validated."
Only write a pattern-based thesis if you see evidence from 2+ source types.""")
    else:
        lines.append("""═══════════════════════════════════════
CORE READ (thesis statement)
═══════════════════════════════════════

Generate one Core Read statement for this person.
It must begin with: "Across interviews, public statements, and company actions, a consistent pattern emerges:"
Identify a specific tension, strategic posture, or recurring decision pattern
visible in at least 2 distinct source types. Focus on behavior, not personality.
1-2 sentences max.

If no genuine cross-source pattern exists, write:
"Sources present a consistent but narrow view — signals below are directional rather than cross-validated."
""")

    # ── PART 1: Orientation ──
    lines.append("")
    if low_signal:
        lines.append("""═══════════════════════════════════════
PART 1: LEAD ORIENTATION (Key Pressure + 5 dimensions — LOW SIGNAL)
═══════════════════════════════════════

FIRST, write the Key Pressure — most likely structural tension based on role context.
2 sentences max. Acknowledge limited signal.

THEN, write up to 5 orientation dimensions. Each: AE implication first, evidence second,
2 sentences max, no labels/headers. Acknowledge limited signal where appropriate.

REJECTION FRAME: If evidence supports it, one dimension should describe what this
person is likely to reject or dismiss. Written as "He'll reject…" / "She has no
patience for…" — cite specific evidence. Do not fabricate if unsupported.""")
    else:
        lines.append("""═══════════════════════════════════════
PART 1: LEAD ORIENTATION (Key Pressure + 5 dimensions)
═══════════════════════════════════════

FIRST, write the Key Pressure — the single most important structural tension this
person is navigating. 2 sentences max. Name the tension specifically, not generically.

THEN, write 5 orientation dimensions. Each dimension:
1. Opens with the IMPLICATION for an AE — what this means for how you sell
2. Supported by specific evidence (quote, decision, pattern)
3. 2 sentences max — no bullet labels, no headers

CRITICAL: Never use the same sentence structure twice. Each dimension should use a
different conversational mode.

RULES:
- Key Pressure: 2 sentences max, specific, leads the section
- Each dimension: 2 sentences max, AE implication first
- No category labels or headers
- NEVER fabricate or speculate without evidence

REJECTION FRAME: At least one of the five dimensions MUST be a rejection frame —
what this person is likely to reject, dismiss, or have zero patience for.
This dimension must:
- Be written as a direct statement: "He'll reject…" / "She has no patience for…"
- Cite a specific quote, decision, or pattern
- Avoid generic language. Never hedge.
- Be the sharpest, most direct line in the section.
If no rejection pattern is clearly supported by evidence, do not fabricate one.""")

    # ── PART 2: Signals ──
    lines.append("")
    if low_signal:
        lines.append("""═══════════════════════════════════════
PART 2: INTEL (5 cards — LOW SIGNAL)
═══════════════════════════════════════

With limited signal, generate 5 Intel cards. Lead with the person (start
every card with He/She, never the company name), connect role context to
likely challenges, end each with a person-specific tactical so-what.
Label inferences clearly. Max 40 words per card, plain prose.
Rank by tactical value. HARD RULE: No more than 2 cards from the same source type.
Generic so-whats like "position your solution" fail — be specific to this person.
CARD 1 ANCHOR RULE: Card 1 must be the strongest signal available, even with limited data.
Prefer a direct quote or observable behavior over role-inferred context.
Card 1 must add leverage beyond the Executive Snapshot — never restate the bio.
NEVER apologize or state that signal depth is weak.

SIGNAL CATEGORIES (for icon selection only):
🎯 BACKGROUND — identity, role context
🔧 PRODUCT — likely operational focus
💰 MARKET — market context
🚨 CHALLENGE — role-typical pressures
⚖️ TENSION — likely trade-offs""")
    else:
        lines.append("""═══════════════════════════════════════
PART 2: INTEL (5 cards, ranked by tactical value)
═══════════════════════════════════════

Generate 5 Intel cards. Each card:
1. States the insight in plain language — no "reveals" formula
2. Ends with person-specific tactical so-what for the AE
3. Ranked by tactical value (highest leverage first)
4. HARD RULE: No more than 2 cards from same source type (video/article/podcast)

NEVER use "reveals" more than once across all 5 cards.

Start every card with He/She, never the company name. Max 40 words per card.
Generic so-whats like "position your offerings" fail — make it specific to
this person's psychology and situation.
CARD 1 ANCHOR RULE: Card 1 must be the strongest available tactical signal.
It must be one of:
- A direct quote-based signal that shows how this person thinks, or
- A behavior-backed signal (decision, repeated action, measurable pattern), or
- A cross-source synthesis signal (if distinct from the Core Read thesis)

Card 1 must:
- Add new leverage beyond the Executive Snapshot and Core Read
- Be immediately usable in a live conversation
- NEVER be company news, funding announcements, headcount data, or LinkedIn bio restatements
- NEVER apologize or state that signal depth is weak
If no cross-source synthesis exists, default to the strongest quote-backed or behavior-backed signal.

SIGNAL CATEGORIES (for icon selection only):
🚀 GROWTH — expansion, scaling, market entry
💰 MARKET — partnerships, funding, competitive positioning
🔧 PRODUCT — tech priorities, product strategy
🎯 BACKGROUND — previous roles, philosophy
⚖️ TENSION — strategic trade-offs
🚨 CHALLENGE — stated problems, pain points

RULES:
1. Each signal: 2-3 sentences, plain prose, ends with AE tactical implication
2. Be specific with numbers, timelines, stakes
3. Each quote: 15-40 words from source material
4. DEDUPLICATE: same fact from multiple sources = ONE signal
5. Include source title, URL, date, timestamp for videos/podcasts
6. Include source title, URL, date, and timestamp (MM:SS) for videos/podcasts.""")

    # ── PART 3: Opening Moves ──
    lines.append("")
    if low_signal:
        lines.append("""═══════════════════════════════════════
PART 3: CONVERSATION ANGLES (1 primary + 2 alternatives — LOW SIGNAL)
═══════════════════════════════════════

Generate one primary and two alternative conversation angles with limited signal.
Frame as genuine discovery approaches. Each angle must use a DIFFERENT conversational
approach — not just a different topic. Never use the same sentence structure twice.

RULES:
- 1-2 sentences per angle, natural prose
- Each angle uses a different conversational mode
- Reference company context the person operates in""")
    else:
        lines.append("""═══════════════════════════════════════
PART 3: CONVERSATION ANGLES (1 primary + 2 alternatives)
═══════════════════════════════════════

Generate one primary Conversation Angle and two alternative angles.

The primary angle should be the single highest-leverage conversation starter —
something specific enough that the prospect will be genuinely surprised the AE knows it.

Each alternative should use a DIFFERENT conversational approach — not just a different
topic (e.g. curiosity, validation, acknowledged tension).

CRITICAL: Never use the same sentence structure twice. The formula
"Reference X — ask how Y" can be used ONCE maximum.

RULES:
- 1-2 sentences per angle, natural prose
- Each angle uses a different conversational mode
- Reference specific facts/signals from the research
- The primary angle must be the single strongest opener""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
OUTPUT FORMAT (JSON object):
═══════════════════════════════════════
{
  "core_read": "Across interviews, public statements, and company actions, a consistent pattern emerges: [specific cross-source pattern].",
  "executive_orientation": {
    "key_pressure": "She's managing the psychological weight of asking studio owners to abandon a decade of Mindbody data — her entire sales motion is built around not traumatizing the customer. If your implementation story is even slightly unclear she'll walk.",
    "bullets": [
      "She measures her own team's performance by technology adoption rates, not quota attainment alone — she'll evaluate your tool the same way. Have a concrete adoption and rollout plan ready before she asks.",
      "He'll reject any pitch that positions your tool as a reporting layer or oversight mechanism — he's on record saying finance should be operating fabric, not auditor. Come in as a partner to his GTM motion, not a check on it."
    ]
  },
  "opening_moves": [
    {"angle": "Primary", "suggestion": "Lead with his 'How I Screwed This Up' series — ask about the most common operational drag he sees in pre-IPO startups today."},
    {"angle": "Alternative", "suggestion": "Tell him you noticed Rippling just crossed $570M ARR and ask what operational bets he made 18 months ago that actually got them there."},
    {"angle": "Alternative", "suggestion": "Most vendors come in talking about efficiency. Open by acknowledging the tension he's described publicly — that platform teams and application teams want different things."}
  ],
  "signals": [
    {
      "category": "GROWTH",
      "signal": "She's pushing a 25% store expansion (2,000 to 2,500) just 8 months into her CEO tenure. That's an aggressive bet with serious execution exposure — ask what's breaking at that growth rate and position your solution around the operational strain she's likely feeling right now.",
      "quote": "It's over 2,000 and on any given day probably getting closer to 2,500...",
      "source": {
        "type": "video",
        "title": "Jordan Schenck CEO Interview",
        "url": "https://youtube.com/watch?v=abgKopCIDOY",
        "timestamp": "08:30",
        "date": "Apr 2, 2025"
      }
    }
  ]
}""")

    return "\n".join(lines)


def _build_qp_sub_c_system(request: ResearchRequest) -> str:
    """Sub-prompt C: Pull Quote (fast, ~3-5s). Only used when transcripts available."""
    lines = [
        "You are an EXECUTIVE INTELLIGENCE engine for B2B sales reps.",
        "",
        _qp_target_block(request),
        "",
        """═══════════════════════════════════════
TASK: Select the SINGLE most revealing direct quote + tactical context
═══════════════════════════════════════

Select the quote that reveals the most about how this person thinks, what they
value, or what they're reacting against — not the most impressive-sounding one.
Recency matters but insight matters more.

Then add a "why_it_matters" line — 1-2 sentences that translate the quote into a
tactical implication for the AE. Always write from the AE's perspective —
what do they DO with this information.

RULES:
- Must be VERBATIM from a video transcript or podcast transcript (15-40 words)
- STRONGLY prefer quotes from podcast/interview transcripts over any other source
- NEVER use text from articles or LinkedIn posts — only spoken words from audio/video
- Choose a quote that reveals their thinking, philosophy, values, or pressure
- Pick quotes where the person speaks with conviction or candor — not generic platitudes
- NEVER select quotes containing generic business language like "customer-obsessed,"
  "results-driven," or "move the needle" unless surrounding context makes them useful.
  If the only available quotes are generic, return null.
- Include the source as: "Source Title - Platform - Date - Timestamp"
- ALWAYS include source_url: the exact URL of the source video/podcast from the SOURCE MATERIAL
- If no podcast/video transcripts are available, return null
- why_it_matters: 1-2 sentences max, always from the AE's perspective

═══════════════════════════════════════
OUTPUT FORMAT (JSON object):
═══════════════════════════════════════
{
  "pull_quote": {"quote": "verbatim text...", "source": "Source Title - Platform - Date - MM:SS", "source_url": "https://youtube.com/watch?v=...", "why_it_matters": "1-2 sentence tactical implication for the AE"}
}

If no direct quotes available, return:
{
  "pull_quote": null
}""",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  CALL 2: Full Dossier
# ═══════════════════════════════════════════════════════════

def _build_dossier_system(request: ResearchRequest, has_person_content: bool) -> str:
    """Build Deep Dive prompt (replaces Full Dossier)."""
    lines = [
        "You are an EXECUTIVE INTELLIGENCE engine building a Deep Dive research brief for B2B sales prep.",
        "",
        "YOUR TASK: Build four evidence-backed sections that PROVE the intelligence in Quick Prep.",
        "Quick Prep is where you think. Deep Dive is where you prove.",
        "",
        "PERSON-FIRST RULE: The individual MUST always be the primary object of analysis.",
        "Company context exists ONLY to illuminate the PERSON.",
        "",
        f"TARGET COMPANY: {request.target_company}",
    ]
    if request.target_name:
        lines.append(f"TARGET PERSON: {request.target_name}")
    if request.target_title:
        lines.append(f"PERSON TITLE: {request.target_title}")

    if not has_person_content and request.target_name:
        lines.append("")
        lines.append("ROLE-BASED FALLBACK: No direct person interviews available.")
        lines.append("Label inferences clearly. Never fabricate quotes.")

    lines.append("")
    lines.append("""EVIDENCE & CITATION RULES:
- EVERY factual claim MUST include an inline citation: [PODCAST 1], [VIDEO 2], [ARTICLE 4], etc.
- Sources are numbered: podcasts first, then videos, then articles (continuous sequence)
- Place citations at the END of the specific claim they support
- Multiple sources for one claim: [VIDEO 1, ARTICLE 3]

RECENCY WEIGHTING:
- Prioritize content from the past 6 months
- De-weight content older than 18 months""")

    lines.append("")
    lines.append("""OUTPUT FORMAT (JSON object with 4 sections):

{
  "deep_pressure": [
    {
      "name": "Scaling Friction",
      "description": "He is managing the structural tension between application-specific teams and horizontal platform teams as the company scales past $16B. He described this publicly as one of the hardest operational problems at their stage. [VIDEO 2] Anyone selling to him needs to show how their tool reduces this tension, not adds to it.",
      "receptivity": "More receptive to tools that reduce cross-team friction; less receptive to anything that adds integration complexity.",
      "forward_facing": false
    },
    {
      "name": "AI Economics Inflection",
      "description": "He is publicly questioning whether current AI software economics hold as compute efficiency improves through 2030. [ARTICLE 5] Any AI-adjacent vendor he evaluates will face questions about long-term unit economics and business model durability.",
      "receptivity": "Come prepared to talk about where your pricing goes as underlying infrastructure costs drop.",
      "forward_facing": true,
      "inferred": true
    }
  ],
  "pattern_evidence": {
    "thesis": "He publicly advocates for eliminating friction while simultaneously creating it through his own constraint philosophy.",
    "quote_evidence": {
      "quote": "They are not the auditor of performance, they are not a check and balance...",
      "source": "CFO Thought Leader - 2022-10-19 - 00:45",
      "source_url": "https://...",
      "why_strongest": "This is the clearest on-record statement of his operating philosophy."
    },
    "behavior_evidence": "Despite advocating for eliminating overhead, he personally administers payroll at scale to identify product gaps. The pattern is consistent: he creates tension deliberately to stress-test systems. [VIDEO 2]",
    "company_evidence": "Rippling's compound product strategy — building deeply integrated HR, IT, and Finance simultaneously — is his philosophy operating at company scale. [VIDEO 3]"
  },
  "in_their_own_words": {
    "core": [
      {
        "quote": "If finance isn't playing the role of a true business partner, finance will fail.",
        "source": "Adil Syed | CFO Thought Leader | 2022-10-19 | 00:30",
        "source_url": "https://...",
        "insight": "He is not describing an aspiration — he is describing a standard he holds vendors to as well."
      }
    ],
    "supporting": [
      {
        "quote": "Watching that play out at Goldman Sachs really changed my perspective on risk.",
        "source": "Adil Syed | CFO Thought Leader | 2022-10-19 | 04:50",
        "source_url": "https://...",
        "insight": "The 2008 experience explains why he probes vendor stability harder than most buyers."
      }
    ],
    "outlier": [
      {
        "quote": "We rarely talk openly about the ways we screw up building companies.",
        "source": "Adil Syed | LinkedIn | 2024-05-01",
        "source_url": "https://...",
        "insight": "This complicates the ruthless operator framing — he is more self-aware about pressure than his positioning suggests."
      }
    ],
    "additional_count": 0
  },
  "sources": [
    {
      "type": "primary",
      "label": "PODCAST",
      "title": "Source title",
      "platform": "Platform",
      "date": "2026-01-23",
      "duration": "28:00",
      "url": "https://...",
      "context_only": false
    }
  ]
}

SECTION RULES:

1. DEEP PRESSURE (exactly 3 pressure points):
   - Evidence-backed situational tensions this person is navigating NOW
   - Specific to THIS person at THIS moment — not generic role pressures
   - Ordered by immediacy: most acute first, background context last
   - At least one must be forward-facing (set forward_facing: true)
   - Each: name (3-5 words), description (2-3 sentences with citations),
     receptivity (1 sentence on what this makes them more/less open to)
   - If inferred rather than directly stated, set inferred: true
   - Never restate Quick Prep content verbatim — go deeper

2. PATTERN EVIDENCE (proves the Core Read thesis):
   - thesis: restate the Core Read thesis being proven
   - quote_evidence: single strongest quote proving the thesis + why
   - behavior_evidence: a decision/action/pattern that proves it behaviorally (2-3 sentences)
   - company_evidence: organizational action reflecting the pattern at scale (2-3 sentences)
   - If evidence type missing: use "Insufficient source coverage for [type] evidence."
   - NEVER fabricate evidence. Only support — do not expand — the thesis.
   - If evidence contradicts the thesis, flag it clearly.

3. IN THEIR OWN WORDS (6-10 quotes organized by relevance):
   - core: quotes directly proving Core Read thesis (2-5 quotes)
   - supporting: quotes adding context/dimension (max 3)
   - outlier: quotes that complicate/contradict (max 2, omit if none)
   - Each: exact quote, source with date/timestamp, one insight line
   - NEVER summarize or paraphrase. Verbatim only.
   - Skip generic business language unless contextually essential
   - If fewer than 4 quotes: add note about limited coverage
   - additional_count: number of extra quotes beyond displayed 10

4. SOURCES:
   - type: "primary" (direct person interviews) or "supporting" (company/context)
   - label: PODCAST / VIDEO / ARTICLE (plain text, no emoji)
   - context_only: true if source provided context but no direct quotes
   - Final summary line is computed client-side from source counts""")

    return "\n".join(lines)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  CALL 2b: Full Dossier — LOW SIGNAL MODE
# ═══════════════════════════════════════════════════════════

def _build_dossier_system_low_signal(request: ResearchRequest) -> str:
    """Build Deep Dive prompt for low-signal executives."""
    lines = [
        "You are an EXECUTIVE INTELLIGENCE engine building a Deep Dive brief for B2B sales prep.",
        "",
        "CRITICAL: Very limited direct public signal exists for this person.",
        "Be honest about confidence levels. Label inferences clearly.",
        "NEVER fabricate quotes or attribute statements to this person.",
        "",
        f"TARGET PERSON: {request.target_name}",
        f"TARGET COMPANY: {request.target_company}",
    ]
    if request.target_title:
        lines.append(f"PERSON TITLE: {request.target_title}")

    lines.append("")
    lines.append("""OUTPUT FORMAT (JSON object with 4 sections — LOW SIGNAL):

{
  "deep_pressure": [
    {
      "name": "Role-Inferred Pressure",
      "description": "As [title] at [company], likely navigating [specific tension]. [ARTICLE N]",
      "receptivity": "Likely more receptive to [approach] based on role context.",
      "forward_facing": false,
      "inferred": true
    }
  ],
  "pattern_evidence": {
    "thesis": "Sources present a consistent but narrow view — thesis support is partial.",
    "quote_evidence": null,
    "behavior_evidence": "Insufficient source coverage for behavior evidence.",
    "company_evidence": "Company-level context suggests [pattern]. [ARTICLE N]"
  },
  "in_their_own_words": {
    "core": [],
    "supporting": [],
    "outlier": [],
    "additional_count": 0,
    "limited_coverage_note": "Limited direct quote coverage — signals in Quick Prep are partially inferred from context."
  },
  "sources": [...]
}

RULES:
- Deep Pressure: 2-3 role-inferred pressures, all marked inferred: true
- Pattern Evidence: be honest about gaps. Use null for missing evidence types.
- In Their Own Words: return empty arrays if no direct quotes. Include limited_coverage_note.
- Sources: mark all as "supporting" (no primary without direct interviews)""")

    return "\n".join(lines)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Build typed objects from raw JSON
# ═══════════════════════════════════════════════════════════

_CATEGORY_ALIASES = {
    "STRATEGY": "MARKET",
    "COMPETITIVE": "MARKET",
    "COMPETITION": "MARKET",
    "LEADERSHIP": "BACKGROUND",
    "INNOVATION": "PRODUCT",
    "TECHNOLOGY": "PRODUCT",
    "TECH": "PRODUCT",
    "AI": "PRODUCT",
    "RISK": "CHALLENGE",
    "THREAT": "CHALLENGE",
    "OPPORTUNITY": "GROWTH",
    "EXPANSION": "GROWTH",
    "REVENUE": "TRACTION",
    "PERFORMANCE": "TRACTION",
    "CULTURE": "BACKGROUND",
    "VISION": "GROWTH",
    "OPERATIONS": "TENSION",
    "EXECUTION": "TENSION",
    "PRESSURE": "TENSION",
}


def _build_signals(raw_signals: list) -> List[Signal]:
    """Convert raw Gemini output into Signal objects, enforcing max 5."""
    signals = []
    for i, raw in enumerate(raw_signals[:5], 1):
        if not isinstance(raw, dict):
            continue

        category = (raw.get("category") or "").upper()
        # Strip emoji prefixes Gemini sometimes adds
        category = re.sub(r"^[^\w]+", "", category).strip()
        # Map common Gemini category variants to our valid set
        if category not in CATEGORY_ICONS:
            mapped = _CATEGORY_ALIASES.get(category)
            if mapped:
                logger.info(f"Signal {i}: mapped category '{category}' -> '{mapped}'")
                category = mapped
            else:
                logger.warning(f"Signal {i} skipped: unknown category '{category}' (valid: {list(CATEGORY_ICONS.keys())})")
                continue

        signal_text = (raw.get("signal") or "").strip()
        if not signal_text:
            logger.warning(f"Signal {i} skipped: empty signal text")
            continue

        quote = (raw.get("quote") or "").strip()
        if not quote:
            logger.warning(f"Signal {i} skipped: empty quote")
            continue

        source_raw = raw.get("source") or {}
        source_type = (source_raw.get("type") or "video").lower()

        signals.append(Signal(
            id=i,
            category=category,
            icon=CATEGORY_ICONS[category],
            signal=signal_text,
            source_type=source_type.upper(),
            expandable={
                "quote": quote,
                "source": {
                    "type": source_type,
                    "title": source_raw.get("title") or "",
                    "url": source_raw.get("url") or "",
                    "timestamp": source_raw.get("timestamp"),
                    "date": source_raw.get("date") or "",
                },
            },
        ))

    return signals


def _build_dossier(raw: dict) -> FullDossier:
    """Convert raw Gemini Deep Dive JSON into a FullDossier object."""
    from models.responses import (
        DeepPressurePoint, PatternEvidence, QuoteEvidence,
        InTheirOwnWords, OwnWordsQuote, DossierSource,
    )

    # Deep Pressure
    deep_pressure = []
    for dp in (raw.get("deep_pressure") or [])[:3]:
        if isinstance(dp, dict):
            deep_pressure.append(DeepPressurePoint(
                name=dp.get("name") or "",
                description=dp.get("description") or "",
                receptivity=dp.get("receptivity") or "",
                forward_facing=bool(dp.get("forward_facing")),
                inferred=bool(dp.get("inferred")),
            ))

    # Pattern Evidence
    pattern_evidence = None
    pe_raw = raw.get("pattern_evidence")
    if isinstance(pe_raw, dict):
        qe_raw = pe_raw.get("quote_evidence")
        quote_ev = None
        if isinstance(qe_raw, dict) and qe_raw.get("quote"):
            quote_ev = QuoteEvidence(
                quote=qe_raw.get("quote") or "",
                source=qe_raw.get("source") or "",
                source_url=qe_raw.get("source_url") or "",
                why_strongest=qe_raw.get("why_strongest") or "",
            )
        pattern_evidence = PatternEvidence(
            thesis=pe_raw.get("thesis") or "",
            quote_evidence=quote_ev,
            behavior_evidence=pe_raw.get("behavior_evidence") or "",
            company_evidence=pe_raw.get("company_evidence") or "",
        )

    # In Their Own Words
    in_their_own_words = None
    itow_raw = raw.get("in_their_own_words")
    if isinstance(itow_raw, dict):
        def _parse_quotes(lst):
            result = []
            for q in (lst or []):
                if isinstance(q, dict) and q.get("quote"):
                    result.append(OwnWordsQuote(
                        quote=q.get("quote") or "",
                        source=q.get("source") or "",
                        source_url=q.get("source_url") or "",
                        insight=q.get("insight") or "",
                    ))
            return result

        in_their_own_words = InTheirOwnWords(
            core=_parse_quotes(itow_raw.get("core"))[:5],
            supporting=_parse_quotes(itow_raw.get("supporting"))[:3],
            outlier=_parse_quotes(itow_raw.get("outlier"))[:2],
            additional_count=itow_raw.get("additional_count") or 0,
            limited_coverage_note=itow_raw.get("limited_coverage_note") or "",
        )

    # Sources
    sources = []
    for s in (raw.get("sources") or []):
        if not isinstance(s, dict):
            continue
        sources.append(DossierSource(
            type=s.get("type") or "supporting",
            label=s.get("label") or "",
            icon=s.get("icon") or "",
            title=s.get("title") or "",
            platform=s.get("platform") or "",
            date=s.get("date") or "",
            duration=s.get("duration"),
            url=s.get("url") or "",
            context_only=bool(s.get("context_only")),
        ))

    return FullDossier(
        deep_pressure=deep_pressure,
        pattern_evidence=pattern_evidence,
        in_their_own_words=in_their_own_words,
        sources=sources,
    )


# ═══════════════════════════════════════════════════════════
#  Incremental Quick Prep (called after each pipeline step)
# ═══════════════════════════════════════════════════════════

async def run_quick_prep_only(
    artifacts: CollectedArtifacts,
    request: ResearchRequest,
    has_person_content: bool = True,
    on_section=None,
) -> Optional[dict]:
    """
    Run Quick Prep as 3 parallel Gemini calls for faster streaming.
    Each sub-call focuses on a subset of sections:
      Sub A (fast): Executive Snapshot + Recent Moves + prior_role
      Sub B (heavy): Signals + Orientation + Opening Moves
      Sub C (fast): Pull Quote
    As each completes, on_section(merged_dict) is called for SSE emission.
    Returns the final merged dict, or None on failure.
    """
    source_material = _build_source_material(artifacts)
    if source_material == "No sources were found.":
        return None

    loop = asyncio.get_event_loop()
    signal_strength = _assess_signal_strength(artifacts, request)
    low_signal = signal_strength == "low" and bool(request.target_name)

    # Build sub-prompts
    sys_a = _build_qp_sub_a_system(request, has_person_content, low_signal)
    sys_b = _build_qp_sub_b_system(request, has_person_content, low_signal)
    content = f"Analyze the following sources and extract executive intelligence:\n\n{source_material}"

    # Shared merge state — asyncio is single-threaded so no lock needed
    merged = {
        "person": {
            "name": request.target_name,
            "title": request.target_title,
            "company": request.target_company,
            "prior_role": None,
            "executive_summary": None,
        },
        "executive_orientation": None,
        "core_read": "",
        "recent_moves": [],
        "signals": [],
        "opening_moves": [],
        "pull_quote": None,
        "dossier": None,
        "warnings": [],
        "sources_analyzed": {
            "podcasts": len(artifacts.podcasts),
            "videos": len(artifacts.videos),
            "articles": len(artifacts.articles),
        },
    }
    has_any = False

    async def _run_sub_a():
        """Snapshot + Recent Moves + prior_role — fast."""
        nonlocal has_any
        try:
            raw = await loop.run_in_executor(
                None, _call_gemini_sync, sys_a, content, 1024
            )
            parsed = _parse_json_safe(raw.strip())
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if isinstance(parsed, dict):
                merged["person"]["prior_role"] = parsed.get("prior_role")
                merged["person"]["executive_summary"] = parsed.get("executive_summary")
                rm_raw = parsed.get("recent_moves") or []
                recent_moves = []
                for rm in rm_raw[:4]:
                    if isinstance(rm, dict) and rm.get("event"):
                        tier = rm.get("tier") or ""
                        if tier == "NONE":
                            recent_moves.append({
                                "tier": "NONE",
                                "event": rm["event"],
                                "date": "",
                                "signal": "",
                                "hook": "",
                                "source_url": "",
                                "source_title": "",
                            })
                            continue
                        if not rm.get("date"):
                            continue
                        if not _is_within_90_days(rm["date"]):
                            logger.info(f"Recent move filtered (too old): {rm['date']} — {rm['event'][:50]}")
                            continue
                        recent_moves.append({
                            "tier": tier,
                            "event": rm["event"],
                            "date": rm["date"],
                            "signal": rm.get("signal") or "",
                            "hook": rm.get("hook") or "",
                            "source_url": rm.get("source_url") or "",
                            "source_title": rm.get("source_title") or "",
                        })
                merged["recent_moves"] = recent_moves
                has_any = True
                logger.info("QP Sub A complete (snapshot + recent moves)")
                if on_section:
                    await on_section(dict(merged))
        except Exception as e:
            logger.error(f"QP Sub A error: {e}")

    async def _run_sub_b():
        """Signals + Orientation + Opening Moves — heavy analysis."""
        nonlocal has_any
        try:
            raw = await loop.run_in_executor(
                None, _call_gemini_sync, sys_b, content, 4096
            )
            parsed = _parse_json_safe(raw.strip())
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if isinstance(parsed, dict):
                # Orientation
                eo_raw = parsed.get("executive_orientation")
                if isinstance(eo_raw, dict):
                    raw_bullets = eo_raw.get("bullets") or []
                    if not raw_bullets and eo_raw.get("growth_posture"):
                        raw_bullets = [
                            v for v in [
                                eo_raw.get("growth_posture"),
                                eo_raw.get("functional_bias"),
                                eo_raw.get("role_context"),
                            ] if v
                        ]
                    eo = ExecutiveOrientation(
                        bullets=[b for b in raw_bullets if isinstance(b, str)][:5],
                        key_pressure=eo_raw.get("key_pressure") or eo_raw.get("vulnerable") or "",
                    )
                    merged["executive_orientation"] = eo.model_dump()

                # Core Read
                cr_raw = parsed.get("core_read")
                if isinstance(cr_raw, str) and cr_raw.strip():
                    merged["core_read"] = cr_raw.strip()

                # Signals
                raw_signals = parsed.get("signals") or []
                signals = _build_signals(raw_signals)
                for i, sig in enumerate(signals, 1):
                    sig.id = i
                merged["signals"] = [s.model_dump() for s in signals]

                # Opening moves
                raw_moves = parsed.get("opening_moves") or []
                om_list = []
                for move in raw_moves[:3]:
                    if isinstance(move, dict) and move.get("angle") and move.get("suggestion"):
                        om_list.append(OpeningMove(
                            angle=move["angle"],
                            suggestion=move["suggestion"],
                        ).model_dump())
                merged["opening_moves"] = om_list

                has_any = True
                logger.info(f"QP Sub B complete (signals={len(signals)}, orientation, opening moves)")
                if on_section:
                    await on_section(dict(merged))
        except Exception as e:
            logger.error(f"QP Sub B error: {e}")

    async def _run_sub_c():
        """Pull Quote — fast, skipped in low-signal mode."""
        nonlocal has_any
        if low_signal:
            return  # No transcripts to quote from
        try:
            sys_c = _build_qp_sub_c_system(request)
            raw = await loop.run_in_executor(
                None, _call_gemini_sync, sys_c, content, 1024
            )
            parsed = _parse_json_safe(raw.strip())
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if isinstance(parsed, dict):
                pq = _validate_pull_quote(parsed.get("pull_quote"), artifacts)
                if pq:
                    merged["pull_quote"] = pq
                    has_any = True
                    logger.info("QP Sub C complete (pull quote)")
                    if on_section:
                        await on_section(dict(merged))
                else:
                    logger.info("QP Sub C: no valid pull quote found")
        except Exception as e:
            logger.error(f"QP Sub C error: {e}")

    # Run all 3 sub-calls in parallel
    await asyncio.gather(_run_sub_a(), _run_sub_b(), _run_sub_c())

    if not has_any:
        return None

    return merged


# ═══════════════════════════════════════════════════════════
#  Main synthesis entry point
# ═══════════════════════════════════════════════════════════

async def synthesize(
    artifacts: CollectedArtifacts,
    request: ResearchRequest,
    has_person_content: bool = True,
    on_partial=None,
) -> ResearchResponse:
    """
    Multi-pass Gemini synthesis with progressive streaming:
      3 parallel QP sub-calls (Snapshot+Moves, Signals+Orientation, PullQuote)
      + 1 Full Dossier call — all 4 run concurrently.
    QP sections emit as each sub-call completes; Dossier awaited last.
    """
    source_material = _build_source_material(artifacts)
    loop = asyncio.get_event_loop()

    # Assess person signal strength for prompt routing
    signal_strength = _assess_signal_strength(artifacts, request)
    low_signal = signal_strength == "low" and bool(request.target_name)
    logger.info(
        f"Person signal strength: {signal_strength} "
        f"(has_person_content={has_person_content}, "
        f"podcasts={len(artifacts.podcasts)}, "
        f"videos={len(artifacts.videos)}, articles={len(artifacts.articles)})"
    )
    if low_signal:
        logger.info("Using LOW-SIGNAL prompt templates (person-first, role-inferred)")

    # Build sub-prompts for parallel QP
    sys_a = _build_qp_sub_a_system(request, has_person_content, low_signal)
    sys_b = _build_qp_sub_b_system(request, has_person_content, low_signal)
    qp_content = f"Analyze the following sources and extract executive intelligence:\n\n{source_material}"

    # Build dossier prompt
    if low_signal:
        dossier_system = _build_dossier_system_low_signal(request)
    else:
        dossier_system = _build_dossier_system(request, has_person_content)
    dossier_content = f"Build a full executive intelligence dossier from these sources:\n\n{source_material}"

    # Shared QP merge state
    prior_role = None
    signals = []
    executive_orientation = None
    core_read = ""
    executive_summary = None
    opening_moves = []
    qp_pull_quote = None
    recent_moves = []
    sources_info = {
        "podcasts": len(artifacts.podcasts),
        "videos": len(artifacts.videos),
        "articles": len(artifacts.articles),
    }

    def _build_partial():
        """Build partial data dict from current merged state."""
        return {
            "person": {
                "name": request.target_name,
                "title": request.target_title,
                "company": request.target_company,
                "prior_role": prior_role,
                "executive_summary": executive_summary,
            },
            "executive_orientation": executive_orientation.model_dump() if executive_orientation else None,
            "core_read": core_read,
            "recent_moves": recent_moves,
            "signals": [s.model_dump() for s in signals],
            "opening_moves": [m.model_dump() for m in opening_moves],
            "pull_quote": qp_pull_quote,
            "dossier": None,
            "warnings": [],
            "sources_analyzed": sources_info,
        }

    async def _emit_partial():
        """Emit current QP state as a partial SSE event."""
        if on_partial and (signals or executive_orientation or executive_summary or qp_pull_quote):
            try:
                await on_partial("quick_prep", _build_partial())
            except Exception as e:
                logger.warning(f"Failed to emit QP partial: {e}")

    # ── QP Sub A: Snapshot + Recent Moves + prior_role (fast) ──
    async def _run_sub_a():
        nonlocal prior_role, executive_summary, recent_moves
        try:
            raw = await loop.run_in_executor(
                None, _call_gemini_sync, sys_a, qp_content, 1024
            )
            parsed = _parse_json_safe(raw.strip())
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if isinstance(parsed, dict):
                prior_role = parsed.get("prior_role")
                executive_summary = parsed.get("executive_summary") or None
                rm_raw = parsed.get("recent_moves") or []
                rm_list = []
                for rm in rm_raw[:4]:
                    if isinstance(rm, dict) and rm.get("event"):
                        tier = rm.get("tier") or ""
                        if tier == "NONE":
                            rm_list.append({
                                "tier": "NONE",
                                "event": rm["event"],
                                "date": "",
                                "signal": "",
                                "hook": "",
                                "source_url": "",
                                "source_title": "",
                            })
                            continue
                        if not rm.get("date"):
                            continue
                        if not _is_within_90_days(rm["date"]):
                            logger.info(f"Recent move filtered (too old): {rm['date']} — {rm['event'][:50]}")
                            continue
                        rm_list.append({
                            "tier": tier,
                            "event": rm["event"],
                            "date": rm["date"],
                            "signal": rm.get("signal") or "",
                            "hook": rm.get("hook") or "",
                            "source_url": rm.get("source_url") or "",
                            "source_title": rm.get("source_title") or "",
                        })
                recent_moves = rm_list
                logger.info("Synthesis QP Sub A complete (snapshot + recent moves)")
                await _emit_partial()
        except Exception as e:
            logger.error(f"Synthesis QP Sub A error: {e}")

    # ── QP Sub B: Signals + Orientation + Opening Moves (heavy) ──
    async def _run_sub_b():
        nonlocal signals, executive_orientation, opening_moves, core_read
        try:
            raw = await loop.run_in_executor(
                None, _call_gemini_sync, sys_b, qp_content, 4096
            )
            parsed = _parse_json_safe(raw.strip())
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if isinstance(parsed, dict):
                logger.info(f"Synthesis QP Sub B parsed keys: {list(parsed.keys())}")

                # Orientation
                eo_raw = parsed.get("executive_orientation")
                if isinstance(eo_raw, dict):
                    raw_bullets = eo_raw.get("bullets") or []
                    if not raw_bullets and eo_raw.get("growth_posture"):
                        raw_bullets = [
                            v for v in [
                                eo_raw.get("growth_posture"),
                                eo_raw.get("functional_bias"),
                                eo_raw.get("role_context"),
                            ] if v
                        ]
                    executive_orientation = ExecutiveOrientation(
                        bullets=[b for b in raw_bullets if isinstance(b, str)][:5],
                        key_pressure=eo_raw.get("key_pressure") or eo_raw.get("vulnerable") or "",
                    )

                # Core Read
                cr_raw = parsed.get("core_read")
                if isinstance(cr_raw, str) and cr_raw.strip():
                    core_read = cr_raw.strip()

                # Signals
                raw_signals = parsed.get("signals") or []
                logger.info(f"Synthesis QP Sub B raw_signals count: {len(raw_signals)}")
                built = _build_signals(raw_signals)
                logger.info(f"Synthesis QP Sub B built signals: {len(built)} from {len(raw_signals)} raw")
                if raw_signals and not built:
                    for i, rs in enumerate(raw_signals[:3]):
                        if isinstance(rs, dict):
                            cat = (rs.get("category") or "").upper()
                            sig_text = bool((rs.get("signal") or "").strip())
                            quote = bool((rs.get("quote") or "").strip())
                            logger.warning(f"QP Sub B signal {i} filtered: category='{cat}' valid={cat in CATEGORY_ICONS}, has_signal={sig_text}, has_quote={quote}")
                for i, sig in enumerate(built, 1):
                    sig.id = i
                signals = built

                # Opening moves
                raw_moves = parsed.get("opening_moves") or []
                om_list = []
                for move in raw_moves[:3]:
                    if isinstance(move, dict) and move.get("angle") and move.get("suggestion"):
                        om_list.append(OpeningMove(
                            angle=move["angle"],
                            suggestion=move["suggestion"],
                        ))
                opening_moves = om_list

                logger.info(f"Synthesis QP Sub B complete (signals={len(signals)}, orientation, opening moves)")
                await _emit_partial()
        except Exception as e:
            logger.error(f"Synthesis QP Sub B error: {e}")

    # ── QP Sub C: Pull Quote (fast, skipped in low-signal mode) ──
    async def _run_sub_c():
        nonlocal qp_pull_quote
        if low_signal:
            return
        try:
            sys_c = _build_qp_sub_c_system(request)
            raw = await loop.run_in_executor(
                None, _call_gemini_sync, sys_c, qp_content, 1024
            )
            parsed = _parse_json_safe(raw.strip())
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if isinstance(parsed, dict):
                pq = _validate_pull_quote(parsed.get("pull_quote"), artifacts)
                if pq:
                    qp_pull_quote = pq
                    logger.info("Synthesis QP Sub C complete (pull quote)")
                    await _emit_partial()
                else:
                    logger.info("Synthesis QP Sub C: no valid pull quote found")
        except Exception as e:
            logger.error(f"Synthesis QP Sub C error: {e}")

    # ── Dossier call (runs in parallel with all QP sub-calls) ──
    async def _run_dossier():
        return await loop.run_in_executor(
            None, _call_claude_sync, dossier_system, dossier_content
        )

    # ── Launch all 4 calls in parallel ──
    dossier_task = asyncio.create_task(_run_dossier())
    await asyncio.gather(_run_sub_a(), _run_sub_b(), _run_sub_c())

    # Give SSE time to flush QP partials before dossier follows
    if on_partial and (signals or executive_orientation):
        await asyncio.sleep(0.3)

    # ── Await Full Dossier (may already be done) ──
    try:
        dossier_result = await dossier_task
    except Exception as e:
        dossier_result = e

    # ── Parse Full Dossier ──
    dossier = None
    dossier_pull_quote = None
    if isinstance(dossier_result, Exception):
        logger.error(f"Gemini Dossier error: {dossier_result}")
    else:
        try:
            parsed2 = _parse_json_safe(dossier_result.strip())
            # Unwrap single-element array (Gemini 3 sometimes wraps output in [])
            if isinstance(parsed2, list) and len(parsed2) == 1 and isinstance(parsed2[0], dict):
                parsed2 = parsed2[0]
            if isinstance(parsed2, dict):
                dossier = _build_dossier(parsed2)
                # Extract pull quote from dossier — validated against transcripts only
                dossier_pull_quote = _validate_pull_quote(parsed2.get("pull_quote"), artifacts)
            else:
                logger.warning("Dossier call returned non-dict, skipping")
        except Exception as e:
            logger.error(f"Dossier parse error: {e}")

    # Use QP pull quote, fall back to dossier pull quote
    pull_quote = qp_pull_quote or dossier_pull_quote

    # ── Compute Research Confidence & Thin Signal Warning ──
    if dossier:
        dossier.research_confidence = _compute_research_confidence(
            artifacts, has_person_content
        )
        if signal_strength == "low":
            dossier.thin_signal_warning = (
                f"No direct interviews or public executive content found for {request.target_name or 'this person'}. "
                "Analysis is role-inferred from title context and company information. "
                "Validate key assumptions in conversation."
            )
        else:
            strong_sources = sum(1 for p in artifacts.podcasts if p.is_person_match) + sum(1 for v in artifacts.videos if v.is_person_match) + len(artifacts.articles)
            if strong_sources < 2:
                dossier.thin_signal_warning = (
                    "Limited public executive signal. "
                    "Analysis primarily based on role context and company information."
                )

    # ── Build response ──
    podcast_count = len(artifacts.podcasts)
    video_count = len(artifacts.videos)
    article_count = len(artifacts.articles)

    return ResearchResponse(
        person=PersonInfo(
            name=request.target_name,
            title=request.target_title,
            company=request.target_company,
            prior_role=prior_role,
            executive_summary=executive_summary,
        ),
        executive_orientation=executive_orientation,
        core_read=core_read,
        recent_moves=[
            RecentMove(
                tier=rm.get("tier", ""),
                event=rm["event"],
                date=rm.get("date", ""),
                signal=rm.get("signal", ""),
                hook=rm.get("hook", ""),
                source_url=rm.get("source_url"),
                source_title=rm.get("source_title"),
            ) for rm in recent_moves
        ],
        signals=signals,
        opening_moves=opening_moves,
        pull_quote=pull_quote,
        dossier=dossier,
        sources_analyzed={
            "podcasts": podcast_count,
            "videos": video_count,
            "articles": article_count,
        },
        metadata={
            "steps_attempted": artifacts.steps_attempted,
            "total_podcasts": podcast_count,
            "total_videos": video_count,
            "total_articles": article_count,
            "podcasts_with_transcripts": sum(
                1 for p in artifacts.podcasts if p.transcript_available
            ),
            "videos_with_transcripts": sum(
                1 for v in artifacts.videos if v.transcript_available
            ),
            "article_search_log": [
                entry.model_dump() for entry in artifacts.article_search_log
            ],
        },
    )
