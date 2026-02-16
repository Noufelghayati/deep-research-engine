from google import genai
from google.genai import types
from models.internal import CollectedArtifacts
from models.requests import ResearchRequest
from models.responses import (
    ResearchResponse,
    PersonInfo,
    Signal,
    OpeningMove,
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
from typing import List
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


def _call_gemini_sync(system_prompt: str, content_prompt: str) -> str:
    """Synchronous Gemini call — run via executor for async compat."""
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=content_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=settings.gemini_max_output_tokens,
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
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
PART 1: EXECUTIVE ORIENTATION (4 lines)
═══════════════════════════════════════

Synthesize a 4-line Executive Orientation that reveals how this person thinks and operates.

LINE 1 — GROWTH POSTURE: What is their orientation toward growth vs efficiency?
  Write naturally based on evidence. Examples:
  ✓ "Aggressive expansion leader in proof-of-scale phase"
  ✓ "Efficiency-focused operator consolidating after rapid growth"
  ✗ Avoid generic: "growth-minded" or "results-driven"

LINE 2 — FUNCTIONAL BIAS & TENSION: What is their primary lens? What are they balancing?
  ✓ "Marketing-led operator balancing impact narrative with commercial viability"
  ✓ "Product-obsessed builder managing tension between quality and speed"
  ✗ Avoid: "strategic leader focused on growth and innovation"

LINE 3 — ROLE CONTEXT: How long in role? What inflection point? Where placing bets?
  ✓ "New CEO (8 months) investing in product quality during rapid scaling"
  ✓ "Long-tenured founder (10 years) shifting from growth to profitability"

LINE 4 — VULNERABILITY (MANDATORY, must be EVIDENCE-BASED):
  - ONLY include vulnerabilities supported by evidence from sources
  - Cite the specific evidence: timelines, role transitions, stated priorities, competitive context
  - If strong evidence: "Vulnerable: [specific pressure with evidence]"
  - If limited evidence: "Limited direct signal — role-typical exposures include [role-based pressures]"
  ✓ "Vulnerable: Execution risk from 25% store growth in 12 months, retailer ROI pressure [VIDEO 1]"
  ✓ "Vulnerable: Platform stability during 3x growth, accumulating technical debt [ARTICLE 3]"
  ✗ NEVER speculate on vulnerabilities without evidence from sources
  ✗ NEVER present company challenges as personal vulnerabilities without connecting to the person

RULES:
- Base everything on evidence from sources (don't speculate)
- Be specific (not "focused on growth" but "25% expansion in 12 months")
- Show tension and trade-offs
- If evidence is thin: "Limited public signal — appears to be..."
- For company-only searches (no person): orient around the company's leadership posture
- NEVER fabricate psychology or decision patterns without direct signal""")

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
PART 3: EXECUTIVE SUMMARY (2-3 sentences)
═══════════════════════════════════════

Write a concise 2-3 sentence executive summary that captures:
- Who this person is and their current role context
- Their primary strategic focus or what defines their leadership right now
- The key tension or pressure they're navigating

This appears directly under the person's name/title in the Quick Prep header.
It should give a sales rep instant context before reading signals.

Example:
"Jordan Schenck is an impact-driven CEO navigating Flashfood's transition from startup to enterprise scale.
She's betting on aggressive retail expansion (25% store growth) while maintaining a triple-bottom-line narrative.
The core tension: proving commercial viability fast enough to justify the growth trajectory."

RULES:
- Write in third person
- Be specific (numbers, timelines, named pressures)
- Show the central tension or trade-off
- Do NOT repeat the orientation lines verbatim""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 4: OPENING MOVES (3 conversation starters)
═══════════════════════════════════════

Generate exactly 3 conversation opening suggestions for the sales rep.
Each move has an "angle" (2-4 word label) and a "suggestion" (1-2 sentence opener).

These should be smart, insight-led openers — NOT generic small talk.
Each should reference a specific signal or pressure from the research.

Example:
{
  "angle": "Scaling Pain",
  "suggestion": "Your 25% store expansion this year is aggressive — how are you thinking about operational infrastructure keeping pace?"
}
{
  "angle": "Enterprise Bet",
  "suggestion": "The Kroger pilot feels like a make-or-break proof point. What does success look like for that partnership?"
}
{
  "angle": "Mission vs. Margin",
  "suggestion": "You've talked about the triple bottom line — how do you handle board conversations when impact and margin compete?"
}

RULES:
- Each angle must target a DIFFERENT dimension (don't repeat themes)
- Reference specific facts/signals from the research
- Frame as genuine curiosity, not sales pitch
- Make the exec think "this person did their homework"
- 1-2 sentences per suggestion, conversational tone""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
OUTPUT FORMAT (JSON object):
═══════════════════════════════════════
{
  "prior_role": "CMO, Impossible Foods" or null,
  "executive_summary": "2-3 sentence summary...",
  "executive_orientation": {
    "growth_posture": "Aggressive expansion leader in proof-of-scale phase",
    "functional_bias": "Marketing-led operator balancing impact narrative with commercial viability",
    "role_context": "New CEO (8 months) investing in product quality during rapid scaling",
    "vulnerable": "Vulnerable: Execution risk, retailer ROI pressure, unproven at enterprise scale"
  },
  "opening_moves": [
    {"angle": "Scaling Pain", "suggestion": "Your 25% store expansion..."},
    {"angle": "Enterprise Bet", "suggestion": "The Kroger pilot feels like..."},
    {"angle": "Mission vs. Margin", "suggestion": "You've talked about..."}
  ],
  "signals": [
    {
      "category": "GROWTH",
      "signal": "Aggressive 25% expansion (2,000→2,500 by EOY) despite 8 months as CEO — high risk/reward growth posture with execution exposure",
      "quote": "It's over 2,000 and on any given day probably getting closer to 2,500... there's a lot more growth happening this year.",
      "source": {
        "type": "video",  // "podcast", "video", or "article"
        "title": "Jordan Schenck CEO Interview",
        "url": "https://youtube.com/watch?v=abgKopCIDOY",
        "timestamp": "08:30",
        "date": "Apr 2, 2025"
      }
    }
  ]
}

Return a JSON object with "prior_role", "executive_summary", "executive_orientation", "opening_moves", and "signals".
If no quality signals found, return {"prior_role": null, "executive_summary": "...", "executive_orientation": {...}, "opening_moves": [...], "signals": []}.""")

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
PART 1: EXECUTIVE ORIENTATION (4 lines — LOW SIGNAL MODE)
═══════════════════════════════════════

Generate an honest, role-based Executive Orientation. Each line MUST acknowledge limited signal.

LINE 1 — GROWTH POSTURE (role-inferred):
  ✓ "Limited direct signal — [title] at [company] during [growth phase] suggests [orientation]"
  ✗ Don't claim to know their growth philosophy without evidence

LINE 2 — FUNCTIONAL BIAS (role-inferred):
  ✓ "[Title]-focused operator — role context suggests emphasis on [likely priorities]"
  ✗ Don't attribute personality traits without direct quotes

LINE 3 — ROLE CONTEXT:
  ✓ "[What we actually know about their role, tenure, background from sources]"
  ✓ If limited: "Role details limited to [title] at [company]; tenure and background not publicly confirmed"

LINE 4 — VULNERABILITY:
  ✓ "Limited direct signal — role-typical exposures include [role-based pressures]"
  ✗ NEVER: Speculative personal vulnerabilities without evidence
  ✗ NEVER: Generic corporate challenges presented as personal vulnerability""")

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
PART 3: EXECUTIVE SUMMARY (2-3 sentences — LOW SIGNAL MODE)
═══════════════════════════════════════

Write a concise 2-3 sentence summary that honestly frames what we know.
- Lead with the person's identity and role
- Acknowledge limited direct signal
- Frame what they're likely navigating based on role context

Example:
"[Name] serves as [Title] at [Company], a [brief company context].
Limited direct executive content is available; analysis is primarily role-inferred.
As [title], they're likely navigating [key role-typical challenge]."

RULES:
- Write in third person
- Be honest about confidence level
- Do NOT speculate on psychology""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
PART 4: OPENING MOVES (3 conversation starters — LOW SIGNAL MODE)
═══════════════════════════════════════

Generate 3 conversation openers even with limited signal.
Use role context and company information to craft smart questions.

Example:
{"angle": "Role Context", "suggestion": "I'd love to understand how your priorities have evolved since joining [Company] — what's top of mind for you right now?"}

RULES:
- Frame as genuine discovery questions (we don't have deep signal, so ASK)
- Reference company context the person operates in
- Each angle targets a different dimension""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
OUTPUT FORMAT (JSON object):
═══════════════════════════════════════
{
  "prior_role": null,
  "executive_summary": "2-3 sentence summary acknowledging limited signal...",
  "executive_orientation": {
    "growth_posture": "Limited direct signal — ...",
    "functional_bias": "...",
    "role_context": "...",
    "vulnerable": "Limited direct signal — role-typical exposures include ..."
  },
  "opening_moves": [
    {"angle": "Role Context", "suggestion": "..."},
    {"angle": "Company Momentum", "suggestion": "..."},
    {"angle": "Market Position", "suggestion": "..."}
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

Return a JSON object with "prior_role", "executive_summary", "executive_orientation", "opening_moves", and "signals".
EVERY signal must be framed around the person, never the company alone.""")

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
{
  "pull_quote": "It's to me truly unacceptable that we continue to fund and fuel more when we have yet to solve the practical reality of all of this amazing stuff that we produce isn't sold.",
  "background": [
    "CEO at Flashfood (appointed May 2025, promoted from President & COO)",
    "Joined Flashfood early 2023 as Chief Customer Officer & Chief Brand Officer",
    "Rapid internal ascent: CCO/CBO → President & COO (Oct 2024) → CEO (May 2025)",
    "Previously: Head of Global Marketing at Impossible Foods (2019-2022)",
    "Career arc: Entrepreneurship → CPG marketing innovation → Food tech sustainability"
  ],
  "executive_profile": {
    "leadership_orientation": {
      "growth_stage": "Proof-of-scale expansion phase — scaling from 2,000 to 2,500+ stores (25% growth in 1 year)",
      "strategic_posture": "Marketing-led growth with impact narrative — leads with sustainability mission, not just cost savings",
      "decision_making_bias": "Growth over efficiency — aggressive expansion timeline despite operational risk, investing in UX during scaling",
      "strategic_implication": "Suggests openness to scalable infrastructure partners"
    },
    "pressure_points": [
      {
        "name": "Execution Risk",
        "why_it_matters": "25% store growth in 12 months requires operational infrastructure that may not exist",
        "evidence": "Platform stability and retailer ROI must hold during rapid expansion [VIDEO 1, ARTICLE 3]"
      },
      {
        "name": "Credibility Window",
        "why_it_matters": "New CEO promoted internally after only 2 years at company — must prove enterprise viability quickly",
        "evidence": "Kroger pilot (16 stores) is key validation test [ARTICLE 4]"
      },
      {
        "name": "Commercial Tension",
        "why_it_matters": "Impact-driven mission vs retailer profitability requirements — all three bottom lines must work simultaneously",
        "evidence": "Publicly frames as 'triple bottom line' company [ARTICLE 2, VIDEO 1]"
      }
    ]
  },
  "strategic_focus": [
    {
      "category": "GROWTH",
      "title": "SCALING & EXPANSION",
      "bullets": [
        "Driving aggressive growth from 2,000 to 2,500+ stores by EOY — targeting full US saturation [VIDEO 1, ARTICLE 3]",
        "Kroger partnership pilot (16 stores in Richmond) to prove enterprise viability [ARTICLE 4]"
      ],
      "strategic_implication": "Prioritizes velocity; may sacrifice short-term margin"
    }
  ],
  "quotes": [
    {
      "topic": "On Food Waste",
      "quote": "It's to me truly unacceptable that we continue to fund and fuel more when we have yet to solve the practical reality of all of this amazing stuff that we produce isn't sold.",
      "source": "Jordan Schenck | Flashfood - YouTube - Apr 2, 2025 - 04:45"
    }
  ],
  "momentum_grouped": [
    {
      "period": "2025-Present",
      "bullets": [
        "Jordan appointed CEO (May 2025) — promoted from President & COO",
        "Kroger partnership launched in Richmond (July 2025) — 16 store pilot"
      ]
    },
    {
      "period": "2024",
      "bullets": [
        "Transformational rebrand and consumer app relaunch (January 2024)",
        "Loblaw partnership delivered $50M in customer savings"
      ]
    },
    {
      "period": "Established Traction",
      "bullets": [
        "140M+ pounds of food diverted from landfills",
        "$355M+ saved by shoppers to date"
      ]
    }
  ],
  "sources": [
    {
      "type": "primary",
      "icon": "📹",
      "title": "Jordan Schenck | Flashfood",
      "platform": "YouTube",
      "date": "Apr 2, 2025",
      "duration": "29:00",
      "url": "https://youtube.com/watch?v=..."
    },
    {
      "type": "supporting",
      "icon": "📄",
      "title": "Flashfood Appoints Jordan Schenck as CEO",
      "platform": "flashfood.com",
      "date": "May 1, 2025",
      "duration": null,
      "url": "https://..."
    }
  ]
}

═══════════════════════════════════════
SECTION RULES:
═══════════════════════════════════════

0. PULL QUOTE (1 standout direct quote):
   - The single most revealing, powerful direct quote from the executive
   - Must be VERBATIM from a video transcript or article (15-40 words)
   - Choose a quote that reveals their thinking, philosophy, or pressure
   - This is displayed prominently in the UI — pick the one that makes a sales rep say "now I understand this person"
   - If no direct quotes exist (company-only search or no interviews), set to null
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

def _build_signals(raw_signals: list) -> List[Signal]:
    """Convert raw Gemini output into Signal objects, enforcing max 5."""
    signals = []
    for i, raw in enumerate(raw_signals[:5], 1):
        if not isinstance(raw, dict):
            continue

        category = (raw.get("category") or "").upper()
        if category not in CATEGORY_ICONS:
            continue

        signal_text = (raw.get("signal") or "").strip()
        if not signal_text:
            continue

        quote = (raw.get("quote") or "").strip()
        if not quote:
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
) -> dict | None:
    """
    Run Quick Prep synthesis only (no dossier).
    Called incrementally as sources are gathered.
    Returns a dict suitable for SSE partial emission, or None on failure.
    """
    source_material = _build_source_material(artifacts)
    if source_material == "No sources were found.":
        return None

    loop = asyncio.get_event_loop()
    signal_strength = _assess_signal_strength(artifacts, request)

    if signal_strength == "low" and request.target_name:
        quick_system = _build_quick_prep_system_low_signal(request)
    else:
        quick_system = _build_quick_prep_system(request, has_person_content)

    quick_content = f"Analyze the following sources and extract executive intelligence:\n\n{source_material}"

    try:
        raw = await loop.run_in_executor(
            None, _call_gemini_sync, quick_system, quick_content
        )
    except Exception as e:
        logger.error(f"Incremental Quick Prep Gemini error: {e}")
        return None

    # Parse
    prior_role = None
    signals = []
    executive_orientation = None
    executive_summary = None
    opening_moves = []

    try:
        parsed = _parse_json_safe(raw.strip())
        if isinstance(parsed, dict):
            prior_role = parsed.get("prior_role")
            raw_signals = parsed.get("signals") or []

            eo_raw = parsed.get("executive_orientation")
            if isinstance(eo_raw, dict):
                executive_orientation = ExecutiveOrientation(
                    growth_posture=eo_raw.get("growth_posture", ""),
                    functional_bias=eo_raw.get("functional_bias", ""),
                    role_context=eo_raw.get("role_context", ""),
                    vulnerable=eo_raw.get("vulnerable", ""),
                )

            executive_summary = parsed.get("executive_summary") or None

            raw_moves = parsed.get("opening_moves") or []
            for move in raw_moves[:3]:
                if isinstance(move, dict) and move.get("angle") and move.get("suggestion"):
                    opening_moves.append(OpeningMove(
                        angle=move["angle"],
                        suggestion=move["suggestion"],
                    ))

        elif isinstance(parsed, list):
            raw_signals = parsed
        else:
            raw_signals = []

        signals = _build_signals(raw_signals)
        for i, sig in enumerate(signals, 1):
            sig.id = i
    except Exception as e:
        logger.error(f"Incremental Quick Prep parse error: {e}")
        return None

    if not signals and not executive_orientation:
        return None

    return {
        "person": {
            "name": request.target_name,
            "title": request.target_title,
            "company": request.target_company,
            "prior_role": prior_role,
            "executive_summary": executive_summary,
        },
        "executive_orientation": executive_orientation.model_dump() if executive_orientation else None,
        "signals": [s.model_dump() for s in signals],
        "opening_moves": [m.model_dump() for m in opening_moves],
        "pull_quote": None,
        "dossier": None,
        "warnings": [],
        "sources_analyzed": {
            "podcasts": len(artifacts.podcasts),
            "videos": len(artifacts.videos),
            "articles": len(artifacts.articles),
        },
    }


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
    Two-pass Gemini synthesis with progressive streaming:
      Call 1: Quick Prep signals + Executive Orientation
      Call 2: Full Dossier (6 sections)
    Both calls start in parallel. Quick Prep emits as soon as ready
    via on_partial callback, before Dossier finishes.
    """
    source_material = _build_source_material(artifacts)
    loop = asyncio.get_event_loop()

    # Assess person signal strength for prompt routing
    signal_strength = _assess_signal_strength(artifacts, request)
    logger.info(
        f"Person signal strength: {signal_strength} "
        f"(has_person_content={has_person_content}, "
        f"podcasts={len(artifacts.podcasts)}, "
        f"videos={len(artifacts.videos)}, articles={len(artifacts.articles)})"
    )

    # Build prompts — use low-signal templates when person data is thin
    if signal_strength == "low" and request.target_name:
        logger.info("Using LOW-SIGNAL prompt templates (person-first, role-inferred)")
        quick_system = _build_quick_prep_system_low_signal(request)
        dossier_system = _build_dossier_system_low_signal(request)
    else:
        quick_system = _build_quick_prep_system(request, has_person_content)
        dossier_system = _build_dossier_system(request, has_person_content)

    quick_content = f"Analyze the following sources and extract executive intelligence:\n\n{source_material}"
    dossier_content = f"Build a full executive intelligence dossier from these sources:\n\n{source_material}"

    # ── Run both Gemini calls in parallel, emit Quick Prep first ──
    async def _run_quick_prep():
        return await loop.run_in_executor(
            None, _call_gemini_sync, quick_system, quick_content
        )

    async def _run_dossier():
        return await loop.run_in_executor(
            None, _call_gemini_sync, dossier_system, dossier_content
        )

    quick_task = asyncio.create_task(_run_quick_prep())
    dossier_task = asyncio.create_task(_run_dossier())

    # Await Quick Prep first (Dossier continues running in background)
    try:
        quick_result = await quick_task
    except Exception as e:
        quick_result = e

    # ── Parse Quick Prep ──
    prior_role = None
    signals = []
    executive_orientation = None
    executive_summary = None
    opening_moves = []
    if isinstance(quick_result, Exception):
        logger.error(f"Gemini Quick Prep error: {quick_result}")
    else:
        try:
            parsed = _parse_json_safe(quick_result.strip())
            if isinstance(parsed, dict):
                prior_role = parsed.get("prior_role")
                raw_signals = parsed.get("signals") or []

                # Parse Executive Orientation
                eo_raw = parsed.get("executive_orientation")
                if isinstance(eo_raw, dict):
                    executive_orientation = ExecutiveOrientation(
                        growth_posture=eo_raw.get("growth_posture", ""),
                        functional_bias=eo_raw.get("functional_bias", ""),
                        role_context=eo_raw.get("role_context", ""),
                        vulnerable=eo_raw.get("vulnerable", ""),
                    )

                # Parse Executive Summary
                executive_summary = parsed.get("executive_summary") or None

                # Parse Opening Moves
                raw_moves = parsed.get("opening_moves") or []
                for move in raw_moves[:3]:
                    if isinstance(move, dict) and move.get("angle") and move.get("suggestion"):
                        opening_moves.append(OpeningMove(
                            angle=move["angle"],
                            suggestion=move["suggestion"],
                        ))

            elif isinstance(parsed, list):
                raw_signals = parsed
            else:
                raw_signals = []
            signals = _build_signals(raw_signals)
            for i, sig in enumerate(signals, 1):
                sig.id = i
        except Exception as e:
            logger.error(f"Quick Prep parse error: {e}")

    # ── Emit Quick Prep partial (progressive streaming) ──
    if on_partial and (signals or executive_orientation):
        try:
            partial_data = {
                "person": {
                    "name": request.target_name,
                    "title": request.target_title,
                    "company": request.target_company,
                    "prior_role": prior_role,
                    "executive_summary": executive_summary,
                },
                "executive_orientation": executive_orientation.model_dump() if executive_orientation else None,
                "signals": [s.model_dump() for s in signals],
                "opening_moves": [m.model_dump() for m in opening_moves],
                "pull_quote": None,
                "dossier": None,
                "warnings": [],
                "sources_analyzed": {
                    "podcasts": len(artifacts.podcasts),
                    "videos": len(artifacts.videos),
                    "articles": len(artifacts.articles),
                },
            }
            await on_partial("quick_prep", partial_data)
            logger.info("Emitted Quick Prep partial result")
            # Give SSE time to flush partial to client before dossier follows
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.warning(f"Failed to emit Quick Prep partial: {e}")

    # ── Await Full Dossier (may already be done) ──
    try:
        dossier_result = await dossier_task
    except Exception as e:
        dossier_result = e

    # ── Parse Full Dossier ──
    dossier = None
    pull_quote = None
    if isinstance(dossier_result, Exception):
        logger.error(f"Gemini Dossier error: {dossier_result}")
    else:
        try:
            parsed2 = _parse_json_safe(dossier_result.strip())
            if isinstance(parsed2, dict):
                dossier = _build_dossier(parsed2)
                # Extract pull quote from dossier response
                pull_quote = parsed2.get("pull_quote") or None
            else:
                logger.warning("Dossier call returned non-dict, skipping")
        except Exception as e:
            logger.error(f"Dossier parse error: {e}")

    # ── Compute Research Confidence & Thin Signal Warning ──
    if dossier:
        dossier.research_confidence = _compute_research_confidence(
            artifacts, has_person_content
        )
        # Thin signal warning based on signal strength
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
