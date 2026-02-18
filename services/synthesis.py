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
    DossierBackground,
    DossierStrategicFocus,
    DossierQuote,
    DossierMomentum,
    DossierMomentumGroup,
    DossierSource,
    ExecutiveOrientation,
    ResearchConfidence,
    ExecutiveProfile,
    LeadershipOrientation,
    PressurePoint,
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

client = genai.Client(api_key=settings.gemini_api_key)

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

    def _build_result(matched_url: str = "") -> dict:
        """Build pull quote result, using matched_url as fallback if Gemini didn't provide one."""
        url = source_url or matched_url
        result = {"quote": quote_text, "source": source_display}
        if url:
            result["source_url"] = url
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
    """Synchronous Gemini call — run via executor for async compat."""
    response = client.models.generate_content(
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
PART 1: LEAD ORIENTATION (3-5 bullets + Key Pressure)
═══════════════════════════════════════

Generate 3-5 Lead Orientation bullets. Each bullet MUST:
- Represent a DIFFERENT strategic dimension (from: capital allocation, product strategy,
  market posture, execution risk, leadership bias, org maturity, competitive positioning, talent strategy)
- Be grounded in explicit evidence from sources
- Avoid repetition or paraphrasing of prior bullets
- Convert fact → implication (surface pressure or strategic tension)

If there is insufficient evidence for a dimension, omit it. Never fabricate.

Examples:
✓ "Fresh $15M capital signals aggressive 18-month expansion timeline — burn rate pressure rising"
✓ "Product-first operator prioritizing AI/ML features over enterprise sales motions"
✓ "Enterprise bet: Kroger pilot is make-or-break validation for platform viability"
✓ "Org maturity gap: founder-led culture hitting scaling limits as headcount doubles"
✗ Avoid restating the same insight in different words across bullets

Then add ONE Key Pressure line: the single most important pressure or vulnerability
this person faces, grounded in evidence. Cite the source if possible.

Example Key Pressure:
✓ "Execution risk from 25% store growth in 12 months with unproven enterprise infrastructure [VIDEO 1]"
✓ "Regulatory uncertainty: FCC drone policy shifts could stall core business expansion [PODCAST 1]"

RULES:
- Each bullet: one concise sentence, max 20 words
- No two bullets may address the same strategic dimension
- Key Pressure: evidence-based, cite source when possible
- If evidence is thin: "Limited direct signal — role-typical pressures include [X]"
- NEVER fabricate or speculate without evidence""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 2: EXECUTIVE INTELLIGENCE SIGNALS
═══════════════════════════════════════

You are extracting EXECUTIVE INTELLIGENCE, not company news.
Each signal must reveal how this executive THINKS, what pressures they're under,
and where they're vulnerable.

SIGNAL FORMULA: [Action/Decision] + [Context/Pressure/Stakes] + [What It Reveals]

BAD SIGNAL (just states what happened):
❌ "Flashfood is scaling from 2,000 to 2,500 stores"

GOOD SIGNAL (reveals patterns):
✅ "Aggressive 25% expansion (2,000→2,500 by EOY) despite 8 months as CEO — high risk/reward growth posture with execution exposure"

More examples:
❌ "Partnered with Kroger" → ✅ "Using Kroger pilot (16 stores) to prove enterprise viability — needs major chain validation to justify scale trajectory"
❌ "Investing in AI for images" → ✅ "Prioritizing UX quality (AI images) over expansion — signals awareness of quality debt from rapid scaling"
❌ "Focuses on sustainability" → ✅ "Leads with 'triple bottom line' mission — balancing impact narrative with commercial pressure"

SIGNAL CATEGORIES (choose based on what the signal REVEALS):

🚀 GROWTH — expansion, scaling, market entry, hiring
💰 MARKET — partnerships, funding, deals, competitive positioning
🔧 PRODUCT — tech priorities, product strategy, operational focus
🎯 BACKGROUND — previous roles, philosophy, approach that shapes decisions
⚖️ TENSION — strategic trade-offs, competing priorities, balance points
🚨 CHALLENGE — stated problems, pain points, risks, vulnerabilities

REQUIRED COMPOSITION (5 signals, each a DIFFERENT category):
- 1 signal showing growth/expansion approach + pressure it creates
- 1 signal showing strategic positioning/market action + stakes involved
- 1 signal showing operational/product priorities + trade-offs
- 1 signal showing background/philosophy + how it shapes decisions
- 1 signal showing commercial/market tension + vulnerability
NEVER repeat the same category. Icons chosen by what signal reveals.

CRITICAL RULES:
1. Every signal MUST show PRESSURE, RISK, or TRADE-OFF.
   Not just "they're doing X" but "they're doing X because Y, which means Z"
2. Be specific with numbers, timelines, stakes.
3. Show what could go wrong.
4. Reveal psychology through actions.
5. Max 25 words per signal. Each signal must be a different dimension.
6. Each quote must be 15-40 words from source material.
7. DEDUPLICATE: same fact from multiple sources = ONE signal citing best source.
8. Include source title, URL, date, and timestamp (MM:SS) for videos/podcasts.

WHAT TO LOOK FOR IN SOURCES:
- Pressure signals: "needs to prove...", "must demonstrate...", timeline pressure
- Risk signals: new in role, aggressive timelines, unproven at scale
- Trade-off signals: "over [alternative]", "while maintaining...", priority shifts
- Psychology signals: how they talk, what they emphasize, their background lens""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 2.5: PULL QUOTE (1 standout direct quote)
═══════════════════════════════════════

Select the single most revealing, powerful direct quote from the executive.
This is the "wow" moment — displayed prominently at the top of the report.

RULES:
- Must be VERBATIM from a video transcript or podcast transcript (15-40 words)
- STRONGLY prefer quotes from podcast/interview transcripts over any other source
- NEVER use text from articles or LinkedIn posts — only spoken words from audio/video
- Choose a quote that reveals their thinking, philosophy, values, or pressure
- Pick quotes where the person speaks with conviction or candor — not generic platitudes
- Include the source as: "Source Title - Platform - Date - Timestamp"
- If no podcast/video transcripts are available, return null

Output: {"quote": "verbatim text...", "source": "Source Title - Platform - Date - MM:SS"}
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

Identify up to 4 concrete, factual events involving this person or their company
that occurred BETWEEN {cutoff.strftime('%B %d, %Y')} and {today.strftime('%B %d, %Y')}.

Examples:
{{"event": "Secured $15M strategic funding round", "date": "January 2026", "source_url": "https://...", "source_title": "TechCrunch"}}
{{"event": "Launched Progress AI product line", "date": "December 2025", "source_url": "https://...", "source_title": "Company Blog"}}

RULES:
- ONLY include events dated AFTER {cutoff.strftime('%B %d, %Y')} — reject anything older
- If no events from the last 90 days exist in sources, return an EMPTY array []
- Never fabricate dates or events
- Max 15 words per event description
- Each event must be factual (funding, product launch, hire, partnership, milestone)
- Include source URL and title for each event""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 5: OPENING MOVES (3 conversation directions)
═══════════════════════════════════════

Generate exactly 3 conversation opening DIRECTIONS (not scripts).
Each move has an "angle" (2-4 word label) and a "suggestion" (max 25 words).

The suggestion is a tactical direction the sales rep adapts to their own voice.
Format: "[Lead with X] — [ask about / reference Y]."

Examples:
{"angle": "Scaling Pain", "suggestion": "Lead with their 25% store growth — ask how infrastructure is keeping pace."}
{"angle": "Enterprise Bet", "suggestion": "Reference the Kroger pilot — ask what success looks like for that partnership."}
{"angle": "AI Investment", "suggestion": "Lead with the $15M AI/robotics reinvestment — ask about scaling across diverse environments."}

RULES:
- Maximum 25 words per suggestion
- Direction, not a script — tell the rep WHAT to reference and WHAT to ask
- Format: "[Lead with X] — [ask about / reference Y]"
- Each angle must target a DIFFERENT dimension
- Reference specific facts/signals from the research""")

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
    "source_url": "https://youtube.com/watch?v=abgKopCIDOY"
  },
  "executive_orientation": {
    "bullets": [
      "Aggressive expansion: scaling from 2,000 to 2,500 stores in 12 months",
      "Marketing-led operator: leads with sustainability narrative over cost savings",
      "Enterprise bet: Kroger pilot is make-or-break validation for platform viability",
      "Quality investment: prioritizing UX (AI images) despite scaling pressure"
    ],
    "key_pressure": "Execution risk from 25% store growth in 12 months with unproven enterprise infrastructure [VIDEO 1]"
  },
  "recent_moves": [
    {"event": "Secured $15M strategic funding round", "date": "January 2026", "source_url": "https://...", "source_title": "TechCrunch"},
    {"event": "Launched AI product line", "date": "December 2025", "source_url": "https://...", "source_title": "Company Blog"}
  ],
  "opening_moves": [
    {"angle": "Scaling Pain", "suggestion": "Lead with their 25% store growth — ask how infrastructure is keeping pace."},
    {"angle": "Enterprise Bet", "suggestion": "Reference the Kroger pilot — ask what success looks like."},
    {"angle": "Mission vs. Margin", "suggestion": "Lead with triple bottom line narrative — ask how impact and margin compete."}
  ],
  "signals": [
    {
      "category": "GROWTH",
      "signal": "Aggressive 25% expansion despite 8 months as CEO — execution exposure",
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

Return a JSON object with "prior_role", "executive_summary", "pull_quote", "executive_orientation", "recent_moves", "opening_moves", and "signals".
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
PART 1: LEAD ORIENTATION (3-5 bullets + Key Pressure — LOW SIGNAL MODE)
═══════════════════════════════════════

Generate 3-5 Lead Orientation bullets with honest role-based inference.
Each bullet MUST:
- Represent a DIFFERENT strategic dimension
- Acknowledge limited direct signal where appropriate
- Label inferences: "Inferred from role context" or "Based on company positioning"
- Convert available facts → implications

Then add ONE Key Pressure line (role-inferred if no direct evidence).

Examples:
✓ "Limited direct signal — [title] at [company] during [growth phase] suggests expansion-focused posture"
✓ "[Title]-focused operator — role context suggests emphasis on [likely priorities]"
✓ "Key Pressure: Limited direct signal — role-typical exposures include [role-based pressures]"

RULES:
- Each bullet: max 20 words
- No two bullets may address the same dimension
- Be honest about confidence level
- NEVER fabricate or speculate without evidence""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 2: EXECUTIVE INTELLIGENCE SIGNALS (LOW SIGNAL MODE)
═══════════════════════════════════════

With limited person-level content, generate signals that:
1. LEAD with what we know about the PERSON (even if minimal)
2. Connect role context to likely operational focus
3. Use company context to illuminate the person's likely challenges
4. CLEARLY LABEL what is observed vs inferred

SIGNAL COMPOSITION (5 signals):
- Signal 1: Identity & role positioning (what we actually know about this person)
- Signal 2: Likely operational focus based on role + company context (labeled inferred)
- Signal 3: Market/competitive context THIS PERSON navigates (framed around the person)
- Signal 4: Role-typical challenges and pressures (labeled inferred)
- Signal 5: Company momentum that shapes this person's priorities (framed around person)

CRITICAL: Frame EVERY signal around the person, even when using company data.
  ✓ "As [title], likely navigating [company challenge] — role demands [focus area]"
  ✗ "[Company] announced [thing]" — this is company news, not person intelligence

SIGNAL CATEGORIES:
🎯 BACKGROUND — identity, role context, career positioning
🔧 PRODUCT — likely operational/product focus based on role
💰 MARKET — market context this person operates in
🚨 CHALLENGE — role-typical pressures and challenges
⚖️ TENSION — likely tensions and trade-offs in this role

Each signal: max 25 words.
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

Same rules as standard mode. Up to 4 factual events AFTER {cutoff_ls.strftime('%B %d, %Y')}.
If no dated events exist within this window, return an EMPTY array [].
Never fabricate dates or events.""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 5: OPENING MOVES (3 conversation directions — LOW SIGNAL MODE)
═══════════════════════════════════════

Generate 3 conversation directions even with limited signal.
Frame as genuine discovery questions (we don't have deep signal, so ASK).

Format: "[Lead with X] — [ask about Y]" (max 25 words per suggestion)

Example:
{"angle": "Role Context", "suggestion": "Lead with their role at [Company] — ask how priorities have evolved since joining."}

RULES:
- Maximum 25 words per suggestion
- Direction, not a script
- Reference company context the person operates in
- Each angle targets a different dimension""")

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
  "executive_orientation": {
    "bullets": [
      "Limited direct signal — [title] at [company] suggests [orientation]",
      "[Title]-focused operator — role context suggests emphasis on [priorities]",
      "Role details limited to [title] at [company]; tenure not publicly confirmed"
    ],
    "key_pressure": "Limited direct signal — role-typical exposures include [role-based pressures]"
  },
  "recent_moves": [],
  "opening_moves": [
    {"angle": "Role Context", "suggestion": "Lead with their role at [Company] — ask how priorities have evolved."},
    {"angle": "Company Momentum", "suggestion": "Reference recent company activity — ask how it shapes their focus."},
    {"angle": "Market Position", "suggestion": "Lead with competitive landscape — ask about differentiation strategy."}
  ],
  "signals": [
    {
      "category": "BACKGROUND",
      "signal": "[Person]-focused signal text, max 25 words",
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

Return a JSON object with "prior_role", "executive_summary", "pull_quote", "executive_orientation", "recent_moves", "opening_moves", and "signals".
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

Identify up to 4 concrete, factual events involving this person or their company
that occurred BETWEEN {cutoff.strftime('%B %d, %Y')} and {today.strftime('%B %d, %Y')}.

RULES:
- ONLY include events dated AFTER {cutoff.strftime('%B %d, %Y')} — reject anything older
- If no events from the last 90 days exist in sources, return an EMPTY array []
- Never fabricate dates or events
- Max 15 words per event description
- Each event must be factual (funding, product launch, hire, partnership, milestone)
- Include source URL and title for each event""")

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
    {"event": "Secured $15M strategic funding round", "date": "January 2026", "source_url": "https://...", "source_title": "TechCrunch"}
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

    # ── PART 1: Orientation ──
    lines.append("")
    if low_signal:
        lines.append("""═══════════════════════════════════════
PART 1: LEAD ORIENTATION (3-5 bullets + Key Pressure)
═══════════════════════════════════════

Generate 3-5 Lead Orientation bullets with honest role-based inference.
Each bullet MUST:
- Represent a DIFFERENT strategic dimension
- Acknowledge limited direct signal where appropriate
- Label inferences: "Inferred from role context" or "Based on company positioning"
- Convert available facts → implications

Then add ONE Key Pressure line (role-inferred if no direct evidence).

RULES:
- Each bullet: max 20 words
- No two bullets may address the same dimension
- Be honest about confidence level
- NEVER fabricate or speculate without evidence""")
    else:
        lines.append("""═══════════════════════════════════════
PART 1: LEAD ORIENTATION (3-5 bullets + Key Pressure)
═══════════════════════════════════════

Generate 3-5 Lead Orientation bullets. Each bullet MUST:
- Represent a DIFFERENT strategic dimension (from: capital allocation, product strategy,
  market posture, execution risk, leadership bias, org maturity, competitive positioning, talent strategy)
- Be grounded in explicit evidence from sources
- Avoid repetition or paraphrasing of prior bullets
- Convert fact → implication (surface pressure or strategic tension)

If there is insufficient evidence for a dimension, omit it. Never fabricate.

Then add ONE Key Pressure line: the single most important pressure or vulnerability
this person faces, grounded in evidence. Cite the source if possible.

RULES:
- Each bullet: one concise sentence, max 20 words
- No two bullets may address the same strategic dimension
- Key Pressure: evidence-based, cite source when possible
- NEVER fabricate or speculate without evidence""")

    # ── PART 2: Signals ──
    lines.append("")
    if low_signal:
        lines.append("""═══════════════════════════════════════
PART 2: EXECUTIVE INTELLIGENCE SIGNALS (5 signals)
═══════════════════════════════════════

With limited person-level content, generate signals that:
1. LEAD with what we know about the PERSON (even if minimal)
2. Connect role context to likely operational focus
3. Use company context to illuminate the person's likely challenges
4. CLEARLY LABEL what is observed vs inferred

SIGNAL COMPOSITION (5 signals):
- Signal 1: Identity & role positioning (what we actually know about this person)
- Signal 2: Likely operational focus based on role + company context (labeled inferred)
- Signal 3: Market/competitive context THIS PERSON navigates (framed around the person)
- Signal 4: Role-typical challenges and pressures (labeled inferred)
- Signal 5: Company momentum that shapes this person's priorities (framed around person)

SIGNAL CATEGORIES:
🎯 BACKGROUND — identity, role context, career positioning
🔧 PRODUCT — likely operational/product focus based on role
💰 MARKET — market context this person operates in
🚨 CHALLENGE — role-typical pressures and challenges
⚖️ TENSION — likely tensions and trade-offs in this role

Each signal: max 25 words.
For quote field: use a relevant company quote if available, or write:
  "No direct quote available — inferred from role context"
For inferred signals, set source type to "article" and use the most relevant company source.""")
    else:
        lines.append("""═══════════════════════════════════════
PART 2: EXECUTIVE INTELLIGENCE SIGNALS (5 signals)
═══════════════════════════════════════

You are extracting EXECUTIVE INTELLIGENCE, not company news.
Each signal must reveal how this executive THINKS, what pressures they're under,
and where they're vulnerable.

SIGNAL FORMULA: [Action/Decision] + [Context/Pressure/Stakes] + [What It Reveals]

BAD SIGNAL (just states what happened):
❌ "Flashfood is scaling from 2,000 to 2,500 stores"

GOOD SIGNAL (reveals patterns):
✅ "Aggressive 25% expansion (2,000→2,500 by EOY) despite 8 months as CEO — high risk/reward growth posture with execution exposure"

SIGNAL CATEGORIES (choose based on what the signal REVEALS):
🚀 GROWTH — expansion, scaling, market entry, hiring
💰 MARKET — partnerships, funding, deals, competitive positioning
🔧 PRODUCT — tech priorities, product strategy, operational focus
🎯 BACKGROUND — previous roles, philosophy, approach that shapes decisions
⚖️ TENSION — strategic trade-offs, competing priorities, balance points
🚨 CHALLENGE — stated problems, pain points, risks, vulnerabilities

REQUIRED COMPOSITION (5 signals, each a DIFFERENT category):
- 1 signal showing growth/expansion approach + pressure it creates
- 1 signal showing strategic positioning/market action + stakes involved
- 1 signal showing operational/product priorities + trade-offs
- 1 signal showing background/philosophy + how it shapes decisions
- 1 signal showing commercial/market tension + vulnerability
NEVER repeat the same category.

CRITICAL RULES:
1. Every signal MUST show PRESSURE, RISK, or TRADE-OFF.
2. Be specific with numbers, timelines, stakes.
3. Max 25 words per signal. Each signal must be a different dimension.
4. Each quote must be 15-40 words from source material.
5. DEDUPLICATE: same fact from multiple sources = ONE signal citing best source.
6. Include source title, URL, date, and timestamp (MM:SS) for videos/podcasts.""")

    # ── PART 3: Opening Moves ──
    lines.append("")
    if low_signal:
        lines.append("""═══════════════════════════════════════
PART 3: OPENING MOVES (3 conversation directions)
═══════════════════════════════════════

Generate 3 conversation directions even with limited signal.
Frame as genuine discovery questions (we don't have deep signal, so ASK).

Format: "[Lead with X] — [ask about Y]" (max 25 words per suggestion)

RULES:
- Maximum 25 words per suggestion
- Direction, not a script
- Reference company context the person operates in
- Each angle targets a different dimension""")
    else:
        lines.append("""═══════════════════════════════════════
PART 3: OPENING MOVES (3 conversation directions)
═══════════════════════════════════════

Generate exactly 3 conversation opening DIRECTIONS (not scripts).
Each move has an "angle" (2-4 word label) and a "suggestion" (max 25 words).

The suggestion is a tactical direction the sales rep adapts to their own voice.
Format: "[Lead with X] — [ask about / reference Y]."

RULES:
- Maximum 25 words per suggestion
- Direction, not a script — tell the rep WHAT to reference and WHAT to ask
- Format: "[Lead with X] — [ask about / reference Y]"
- Each angle must target a DIFFERENT dimension
- Reference specific facts/signals from the research""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
OUTPUT FORMAT (JSON object):
═══════════════════════════════════════
{
  "executive_orientation": {
    "bullets": [
      "Aggressive expansion: scaling from 2,000 to 2,500 stores in 12 months",
      "Marketing-led operator: leads with sustainability narrative over cost savings"
    ],
    "key_pressure": "Execution risk from 25% store growth with unproven infrastructure [VIDEO 1]"
  },
  "opening_moves": [
    {"angle": "Scaling Pain", "suggestion": "Lead with their 25% store growth — ask how infrastructure is keeping pace."},
    {"angle": "Enterprise Bet", "suggestion": "Reference the Kroger pilot — ask what success looks like."},
    {"angle": "Mission vs. Margin", "suggestion": "Lead with triple bottom line narrative — ask how impact and margin compete."}
  ],
  "signals": [
    {
      "category": "GROWTH",
      "signal": "Aggressive 25% expansion despite 8 months as CEO — execution exposure",
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
TASK: Select the SINGLE most revealing direct quote from the executive
═══════════════════════════════════════

Select the single most revealing, powerful direct quote from the executive.
This is the "wow" moment — displayed prominently at the top of the report.

RULES:
- Must be VERBATIM from a video transcript or podcast transcript (15-40 words)
- STRONGLY prefer quotes from podcast/interview transcripts over any other source
- NEVER use text from articles or LinkedIn posts — only spoken words from audio/video
- Choose a quote that reveals their thinking, philosophy, values, or pressure
- Pick quotes where the person speaks with conviction or candor — not generic platitudes
- Include the source as: "Source Title - Platform - Date - Timestamp"
- ALWAYS include source_url: the exact URL of the source video/podcast from the SOURCE MATERIAL
- If no podcast/video transcripts are available, return null

═══════════════════════════════════════
OUTPUT FORMAT (JSON object):
═══════════════════════════════════════
{
  "pull_quote": {"quote": "verbatim text...", "source": "Source Title - Platform - Date - MM:SS", "source_url": "https://youtube.com/watch?v=... or podcast URL"}
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
    lines = [
        "You are an EXECUTIVE INTELLIGENCE engine building a deep research dossier for B2B sales prep.",
        "",
        "YOUR TASK: Reveal how this executive THINKS, what pressures they navigate,",
        "and where they're vulnerable. This is EXECUTIVE PSYCHOLOGY, not company news.",
        "",
        "PERSON-FIRST RULE (MANDATORY when target person is specified):",
        "The individual MUST always be the primary object of analysis.",
        "Company context exists ONLY to illuminate the PERSON's world.",
        "Every section must be framed through the person's lens.",
        "NEVER lead with company strategy — lead with the PERSON.",
        "",
        "Analyze ALL sources and produce a structured dossier with 6 sections.",
        "This must be SYNTHESIZED intelligence — cluster themes across sources, not per-source summaries.",
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
        lines.append(f"- Focus on what someone in the role of {request.target_title or 'executive'} at {request.target_company} would be navigating")
        lines.append("- Generate role-based insights, labeling them as 'Derived from role context, not direct statement'")
        lines.append("- Never fabricate quotes or attribute statements to the target person")

    lines.append("")
    lines.append("""SYNTHESIS QUALITY (MANDATORY):
- INTERPRET, don't just summarize. Your job is to reveal MEANING behind facts.
  BAD: "Company grew 25% last year" (just restating a fact)
  GOOD: "25% growth in 12 months creates execution risk — infrastructure must scale faster than revenue"
- Every bullet should answer "so what?" — connect facts to pressures, trade-offs, and implications.
- Cross-reference sources: if VIDEO 1 says X and ARTICLE 3 says Y, synthesize the pattern.
- Never produce a grocery list of disconnected facts. Weave a narrative about the PERSON.

EVIDENCE & CITATION RULES (MANDATORY):
- EVERY factual claim MUST include an inline citation: [PODCAST 1], [VIDEO 2], [ARTICLE 4], etc.
- Citation format: [PODCAST N] for podcast sources, [VIDEO N] for YouTube sources, [ARTICLE N] for article sources
- N corresponds to the source number in the SOURCE MATERIAL
- Sources are numbered: podcasts first, then videos, then articles (continuous sequence)
- So if there are 1 podcast and 2 videos: PODCAST 1 = SOURCE 1, VIDEO 2 = SOURCE 2, VIDEO 3 = SOURCE 3, ARTICLE 4 = SOURCE 4, etc.
- Place citations at the END of the specific claim they support, not at the end of a paragraph
- Multiple sources for one claim: [VIDEO 1, ARTICLE 3]
- If inferring from role context (no source), label: "Inferred from role context"
- No personality traits unless directly stated in sources
- Vulnerabilities must be based on: timelines, role transitions, stated priorities,
  competitive context, scaling velocity — NOT psychological speculation

RECENCY & TENURE WEIGHTING:
- Prioritize content from the past 6 months
- De-weight content older than 18 months
- If executive changed roles within past 12 months, prioritize post-transition content""")

    lines.append("")
    lines.append("""OUTPUT FORMAT (JSON object with 7 sections):

⚠️ CRITICAL: The example below is ONLY to show JSON structure and field names.
DO NOT copy any text, names, quotes, facts, or numbers from this example into your output.
Your output must contain ONLY information from the SOURCE MATERIAL provided above.

--- FORMAT EXAMPLE (structure only, do NOT copy content) ---
{
  "pull_quote": "<verbatim quote from transcript>",
  "background": [
    "<role> at <company> (<date>, promoted from <prior role>) [SOURCE N]",
    "Previously: <prior role> at <prior company> (<years>) [SOURCE N]",
    "Career arc: <domain1> → <domain2> → <current domain>"
  ],
  "executive_profile": {
    "leadership_orientation": {
      "growth_stage": "<phase description> — <specific metrics from sources> [SOURCE N]",
      "strategic_posture": "<approach/philosophy> — <how they lead, evidence-based> [SOURCE N]",
      "decision_making_bias": "<bias description> — <specific evidence of this pattern> [SOURCE N]",
      "strategic_implication": "<5-10 word observational note>"
    },
    "pressure_points": [
      {
        "name": "<short label>",
        "why_it_matters": "<1-2 sentences on business impact> [SOURCE N]",
        "evidence": "<specific quote or fact proving this> [SOURCE N]"
      }
    ]
  },
  "strategic_focus": [
    {
      "category": "GROWTH",
      "title": "<THEME IN CAPS>",
      "bullets": [
        "<strategic initiative with specific detail> [SOURCE N, SOURCE M]"
      ],
      "strategic_implication": "<5-10 word so-what insight>"
    }
  ],
  "quotes": [
    {
      "topic": "<topic label>",
      "quote": "<verbatim quote from source>",
      "source": "<speaker> | <platform> - <date> - <timestamp>"
    }
  ],
  "momentum_grouped": [
    {
      "period": "2025-Present",
      "bullets": [
        "<event with date and context> [SOURCE N]"
      ]
    },
    {
      "period": "2024",
      "bullets": [
        "<event with date and context> [SOURCE N]"
      ]
    },
    {
      "period": "Established Traction",
      "bullets": [
        "<long-term achievement or foundation metric> [SOURCE N]"
      ]
    }
  ],
  "sources": [
    {
      "type": "primary",
      "icon": "📹",
      "title": "<source title>",
      "platform": "<platform>",
      "date": "<date>",
      "duration": "<duration>",
      "url": "<url>"
    },
    {
      "type": "supporting",
      "icon": "📄",
      "title": "<source title>",
      "platform": "<platform>",
      "date": "<date>",
      "duration": null,
      "url": "<url>"
    }
  ]
}
--- END FORMAT EXAMPLE ---

═══════════════════════════════════════
SECTION RULES:
═══════════════════════════════════════

0. PULL QUOTE (1 standout direct quote):
   - The single most revealing, powerful direct quote from the executive
   - Must be VERBATIM from a podcast transcript or video transcript (15-40 words)
   - STRONGLY prefer quotes from podcast/interview transcripts over any other source
   - NEVER use text from articles or LinkedIn posts — only spoken words from audio/video
   - Choose a quote that reveals their thinking, philosophy, or pressure
   - This is displayed prominently in the UI — pick the one that makes a sales rep say "now I understand this person"
   - If no podcast/video transcripts exist (company-only search or no interviews), set to null
   - Do NOT fabricate or paraphrase — verbatim only

1. BACKGROUND (max 6 bullets):
   - Role, prior companies, years of experience
   - Include narrative context (not just resume bullets)
   - Career arc showing how they got here
   - No speculation — only verified facts from sources

2. EXECUTIVE PROFILE (NEW — critical section):

   LEADERSHIP ORIENTATION:
   - growth_stage: What phase? Specific metrics + trajectory + tenure context
   - strategic_posture: Primary approach/philosophy, what they lead with, how they balance priorities
   - decision_making_bias: Growth vs efficiency, speed vs quality, risk tolerance — show evidence
   - strategic_implication: 5-10 word observational note (e.g. "Suggests openness to scalable infrastructure")

   PRESSURE POINTS & VULNERABILITIES (3-4 thematic clusters):
   Each cluster MUST be EVIDENCE-BASED:
   - Name the pressure/vulnerability specifically
   - Explain why it matters (stakes, consequences)
   - Cite evidence from sources with [VIDEO X] or [ARTICLE X] references
   - Be specific about what could break
   - NEVER speculate on vulnerabilities without direct evidence
   - If evidence is thin: label as "Inferred from role context" and explain reasoning

   Common themes to consider: Execution Risk, Credibility Window, Commercial Tension,
   Market Position, Technical Debt, Organizational Strain

   QUALITY BAR:
   ✓ Specific (numbers, timelines, named pressures)
   ✓ Honest (don't sugarcoat vulnerabilities)
   ✓ Evidence-based (cite sources for every claim)
   ✓ Connected to the PERSON (not just company challenges)
   ✗ Generic ("facing market challenges")
   ✗ Safe ("well-positioned for growth")
   ✗ Speculative psychology without direct quotes or evidence
   ✗ Company challenges presented as personal vulnerabilities without connecting to the person

3. STRATEGIC FOCUS (3-6 themes):
   - Synthesized across multiple sources using cross-source pattern recognition
   - Group by category with inline source citations
   - Each theme gets a strategic_implication (5-10 word observational note)
   - MUST use ONLY these category values:
     GROWTH, MARKET, PRODUCT, CHALLENGE, TRACTION, BACKGROUND, TENSION
   - Do NOT invent new category names

4. QUOTES (3-5 direct quotes, 15-40 words each):
   - DIRECT quotes only, no paraphrasing
   - Topic label and source citation with timestamp
   - Prefer quotes that reveal psychology, priorities, or pressure

5. MOMENTUM GROUPED (organized by recency):
   - Group into periods: "2025-Present", "2024", "Established Traction"
   - Most recent first
   - Include specific dates where possible

6. SOURCES (all sources used):
   - type: "primary" (person interviews) or "supporting" (company/press)
   - icon: 🎧 for podcast, 📹 for video, 📄 for article
   - Include title, platform, date, duration (video/podcast only), url""")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  CALL 2b: Full Dossier — LOW SIGNAL MODE
# ═══════════════════════════════════════════════════════════

def _build_dossier_system_low_signal(request: ResearchRequest) -> str:
    """Build Dossier prompt for low-signal executives (no direct interviews)."""
    lines = [
        "You are an EXECUTIVE INTELLIGENCE engine building a research dossier for B2B sales prep.",
        "",
        "CRITICAL CONTEXT: Very limited direct public signal exists for this person.",
        "There are NO direct interviews, podcast appearances, or keynotes indexed.",
        "",
        "YOUR APPROACH:",
        "1. PERSON is always the primary subject — never let company context dominate",
        "2. Use role/title to infer likely focus areas (clearly labeled)",
        "3. Include company context ONLY as supporting evidence for the person's world",
        "4. Be honest about confidence levels throughout",
        "5. NEVER fabricate quotes or attribute statements to this person",
        "6. NEVER speculate on personality or psychology without evidence",
        "",
        f"TARGET PERSON: {request.target_name}",
        f"TARGET COMPANY: {request.target_company}",
    ]
    if request.target_title:
        lines.append(f"PERSON TITLE: {request.target_title}")

    lines.append("")
    lines.append("""OUTPUT FORMAT (JSON object — LOW SIGNAL MODE):
{
  "pull_quote": null,
  "background": [
    "[Name] serves as [Title] at [Company]",
    "[Any verified background facts from sources]",
    "Role context: [What this title typically involves at a company like this]",
    "Limited public executive content available — analysis primarily role-inferred"
  ],
  "executive_profile": {
    "leadership_orientation": {
      "growth_stage": "[Company]'s current phase: [what we know] — [person]'s role in this context",
      "strategic_posture": "Inferred from role: [title] typically focuses on [likely priorities]",
      "decision_making_bias": "Insufficient direct signal to assess individual decision-making patterns",
      "strategic_implication": "Role suggests focus on [likely area] — requires validation"
    },
    "pressure_points": [
      {
        "name": "Role-Typical Pressure",
        "why_it_matters": "[What someone in this role typically navigates]",
        "evidence": "Inferred from role context and company positioning"
      }
    ]
  },
  "strategic_focus": [
    {
      "category": "PRODUCT",
      "title": "LIKELY FOCUS: [Area]",
      "bullets": [
        "As [title], likely responsible for [focus area] — [company context supports this]",
        "[Another role-inferred focus with company evidence]"
      ],
      "strategic_implication": "Inferred — requires validation in conversation"
    }
  ],
  "quotes": [],
  "momentum_grouped": [
    {
      "period": "Company Context",
      "bullets": [
        "[Recent company developments that shape this person's operating environment]"
      ]
    }
  ],
  "sources": [...]
}

═══════════════════════════════════════
SECTION RULES (LOW SIGNAL MODE):
═══════════════════════════════════════

1. BACKGROUND (max 6 bullets):
   - LEAD with person's identity and role — this is about THEM
   - Include any verified facts from sources (LinkedIn, press mentions, etc.)
   - Add role context: what this title typically involves at a company like this
   - Acknowledge signal limitations honestly
   - NEVER pad with company history as if it's person background
   - Career arc if available, otherwise: "Background details not publicly confirmed"

2. EXECUTIVE PROFILE:

   LEADERSHIP ORIENTATION:
   - Frame around the person's role, not the company's strategy
   - Label inferences clearly: "Inferred from role context"
   - "Insufficient direct signal" is an acceptable and honest answer
   - strategic_implication: always note inference needs validation

   PRESSURE POINTS (2-3 role-inferred):
   - Focus on role-typical pressures, not speculative personal ones
   - Every point MUST be labeled: "Inferred from role context"
   - Frame as: what someone in THIS role at THIS company likely navigates
   - NEVER present company challenges as personal vulnerabilities
   - NEVER speculate on psychology or decision patterns

3. STRATEGIC FOCUS (3-4 themes):
   - Title each as "LIKELY FOCUS: [Area]" to signal inference
   - Frame around the person's role responsibilities
   - Use company context as supporting evidence, not the headline
   - strategic_implication: always note "Inferred — requires validation"
   - MUST use ONLY these category values:
     GROWTH, MARKET, PRODUCT, CHALLENGE, TRACTION, BACKGROUND, TENSION

4. QUOTES:
   - Return EMPTY array [] — do not fabricate quotes
   - Being honest is better than fabricated content

5. MOMENTUM GROUPED:
   - Use "Company Context" as period label
   - Include company developments that directly affect this person's role
   - Frame as: developments shaping the person's operating environment
   - Second group can be "Market Context" for competitive landscape

6. SOURCES:
   - List all sources used
   - Mark ALL as "supporting" (none are "primary" without direct interviews)
   - icon: 🎧 for podcasts, 📹 for videos, 📄 for articles""")

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
    """Convert raw Gemini dossier JSON into a FullDossier object."""
    background = DossierBackground(
        bullets=[b for b in (raw.get("background") or []) if isinstance(b, str)][:6]
    )

    # Executive Profile
    executive_profile = None
    ep_raw = raw.get("executive_profile")
    if isinstance(ep_raw, dict):
        lo_raw = ep_raw.get("leadership_orientation") or {}
        leadership_orientation = LeadershipOrientation(
            growth_stage=lo_raw.get("growth_stage", ""),
            strategic_posture=lo_raw.get("strategic_posture", ""),
            decision_making_bias=lo_raw.get("decision_making_bias", ""),
            strategic_implication=lo_raw.get("strategic_implication", ""),
        )
        pressure_points = []
        for pp in (ep_raw.get("pressure_points") or [])[:4]:
            if isinstance(pp, dict):
                pressure_points.append(PressurePoint(
                    name=pp.get("name", ""),
                    why_it_matters=pp.get("why_it_matters", ""),
                    evidence=pp.get("evidence", ""),
                ))
        executive_profile = ExecutiveProfile(
            leadership_orientation=leadership_orientation,
            pressure_points=pressure_points,
        )

    # Strategic Focus (with strategic_implication)
    themes = []
    for t in (raw.get("strategic_focus") or [])[:6]:
        if not isinstance(t, dict):
            continue
        category = (t.get("category") or "").upper()
        # Deterministic icon: always use our mapping, never trust Gemini's icon
        icon = CATEGORY_ICONS.get(category, "")
        themes.append({
            "category": category,
            "icon": icon,
            "title": t.get("title") or "",
            "bullets": [b for b in (t.get("bullets") or []) if isinstance(b, str)],
            "strategic_implication": t.get("strategic_implication", ""),
        })
    strategic_focus = DossierStrategicFocus(themes=themes)

    # Quotes
    quotes = []
    for q in (raw.get("quotes") or [])[:5]:
        if not isinstance(q, dict):
            continue
        quote_text = (q.get("quote") or "").strip()
        if not quote_text:
            continue
        quotes.append(DossierQuote(
            topic=q.get("topic") or "",
            quote=quote_text,
            source=q.get("source") or "",
        ))

    # Momentum (flat — backward compat)
    momentum = DossierMomentum(
        bullets=[b for b in (raw.get("momentum") or []) if isinstance(b, str)][:6]
    )

    # Momentum Grouped (new)
    momentum_grouped = None
    mg_raw = raw.get("momentum_grouped")
    if isinstance(mg_raw, list) and len(mg_raw) > 0:
        momentum_grouped = []
        for group in mg_raw:
            if isinstance(group, dict):
                momentum_grouped.append(DossierMomentumGroup(
                    period=group.get("period", ""),
                    bullets=[b for b in (group.get("bullets") or []) if isinstance(b, str)],
                ))

    # Sources
    sources = []
    for s in (raw.get("sources") or []):
        if not isinstance(s, dict):
            continue
        sources.append(DossierSource(
            type=s.get("type") or "supporting",
            icon=s.get("icon") or "",
            title=s.get("title") or "",
            platform=s.get("platform") or "",
            date=s.get("date") or "",
            duration=s.get("duration"),
            url=s.get("url") or "",
        ))

    return FullDossier(
        background=background,
        executive_profile=executive_profile,
        strategic_focus=strategic_focus,
        quotes=quotes,
        momentum=momentum,
        momentum_grouped=momentum_grouped,
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
                    if isinstance(rm, dict) and rm.get("event") and rm.get("date"):
                        if not _is_within_90_days(rm["date"]):
                            logger.info(f"Recent move filtered (too old): {rm['date']} — {rm['event'][:50]}")
                            continue
                        recent_moves.append({
                            "event": rm["event"],
                            "date": rm["date"],
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
                    if isinstance(rm, dict) and rm.get("event") and rm.get("date"):
                        if not _is_within_90_days(rm["date"]):
                            logger.info(f"Recent move filtered (too old): {rm['date']} — {rm['event'][:50]}")
                            continue
                        rm_list.append({
                            "event": rm["event"],
                            "date": rm["date"],
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
        nonlocal signals, executive_orientation, opening_moves
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
            None, _call_gemini_sync, dossier_system, dossier_content
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
        recent_moves=[
            RecentMove(
                event=rm["event"],
                date=rm["date"],
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
