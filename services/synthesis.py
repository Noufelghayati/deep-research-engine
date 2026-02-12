from google import genai
from google.genai import types
from models.internal import CollectedArtifacts
from models.requests import ResearchRequest
from models.responses import (
    ResearchResponse,
    PersonInfo,
    Signal,
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

def _build_source_material(artifacts: CollectedArtifacts) -> str:
    """Build the source material block shared by both prompts."""
    sections = []

    for i, video in enumerate(artifacts.videos, 1):
        section = f"=== SOURCE {i}: YouTube Video ===\n"
        section += f"Title: {video.title}\n"
        section += f"Channel: {video.channel_title}\n"
        section += f"URL: {video.url}\n"
        section += f"Published: {video.published_at}\n"
        if video.is_person_match:
            section += "Match type: PERSON-LEVEL (features the target person)\n"
        else:
            section += "Match type: COMPANY-LEVEL (features company leadership)\n"
        if video.transcript_text:
            section += f"Transcript:\n{video.transcript_text}\n"
        else:
            section += (
                "[No transcript available. Use title and description only.]\n"
                f"Description: {video.description}\n"
            )
        sections.append(section)

    for i, article in enumerate(artifacts.articles, len(artifacts.videos) + 1):
        section = f"=== SOURCE {i}: Article ===\n"
        section += f"Title: {article.title}\n"
        section += f"URL: {article.url}\n"
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
    """Compute research confidence from source quality."""
    now = datetime.now()
    six_months_ago = now - timedelta(days=180)

    direct_interviews = sum(1 for v in artifacts.videos if v.is_person_match)
    total_sources = len(artifacts.videos) + len(artifacts.articles)

    recent_count = 0
    for v in artifacts.videos:
        dt = _parse_date_lenient(v.published_at)
        if dt and dt >= six_months_ago:
            recent_count += 1
    for a in artifacts.articles:
        dt = _parse_date_lenient(a.published_date or "")
        if dt and dt >= six_months_ago:
            recent_count += 1

    if direct_interviews >= 3 and recent_count >= 1:
        return ResearchConfidence(
            level="high",
            label=f"{direct_interviews} direct interviews with recent coverage",
        )
    elif direct_interviews >= 1 and total_sources >= 4:
        return ResearchConfidence(
            level="medium",
            label=f"{direct_interviews} interview(s) + {total_sources - direct_interviews} supporting sources",
        )
    else:
        return ResearchConfidence(
            level="low",
            label="No direct interviews; role-based synthesis only" if direct_interviews == 0
                  else f"Limited sources ({total_sources} total)",
        )


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

LINE 4 — VULNERABILITY (MANDATORY, must be specific):
  ✓ "Vulnerable: Execution risk, retailer ROI pressure, unproven at enterprise scale"
  ✓ "Vulnerable: Platform stability during 3x growth, accumulating technical debt"
  ✗ Avoid vague: "market challenges" or "competitive pressure"

RULES:
- Base everything on evidence from sources (don't speculate)
- Be specific (not "focused on growth" but "25% expansion in 12 months")
- Show tension and trade-offs
- If evidence is thin: "Limited public signal — appears to be..."
- For company-only searches (no person): orient around the company's leadership posture""")

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
8. Include source title, URL, date, and timestamp (MM:SS) for videos.

WHAT TO LOOK FOR IN SOURCES:
- Pressure signals: "needs to prove...", "must demonstrate...", timeline pressure
- Risk signals: new in role, aggressive timelines, unproven at scale
- Trade-off signals: "over [alternative]", "while maintaining...", priority shifts
- Psychology signals: how they talk, what they emphasize, their background lens""")

    lines.append("")
    lines.append("""═══════════════════════════════════════
OUTPUT FORMAT (JSON object):
═══════════════════════════════════════
{
  "prior_role": "CMO, Impossible Foods" or null,
  "executive_orientation": {
    "growth_posture": "Aggressive expansion leader in proof-of-scale phase",
    "functional_bias": "Marketing-led operator balancing impact narrative with commercial viability",
    "role_context": "New CEO (8 months) investing in product quality during rapid scaling",
    "vulnerable": "Vulnerable: Execution risk, retailer ROI pressure, unproven at enterprise scale"
  },
  "signals": [
    {
      "category": "GROWTH",
      "signal": "Aggressive 25% expansion (2,000→2,500 by EOY) despite 8 months as CEO — high risk/reward growth posture with execution exposure",
      "quote": "It's over 2,000 and on any given day probably getting closer to 2,500... there's a lot more growth happening this year.",
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

Return a JSON object with "prior_role", "executive_orientation", and "signals".
If no quality signals found, return {"prior_role": null, "executive_orientation": {...}, "signals": []}.""")

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
    lines.append("""EVIDENCE & INFERENCE RULES (MANDATORY):
- Every factual claim must be tied to a cited source (e.g. [VIDEO 1], [ARTICLE 5])
- If inferring from role context, label: "Inferred from role context"
- No personality traits unless directly stated in sources
- Vulnerabilities must be based on: timelines, role transitions, stated priorities,
  competitive context, scaling velocity — NOT psychological speculation

RECENCY & TENURE WEIGHTING:
- Prioritize content from the past 6 months
- De-weight content older than 18 months
- If executive changed roles within past 12 months, prioritize post-transition content""")

    lines.append("")
    lines.append("""OUTPUT FORMAT (JSON object with 6 sections):
{
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
   Each cluster must:
   - Name the pressure/vulnerability specifically
   - Explain why it matters (stakes, consequences)
   - Cite evidence from sources
   - Be specific about what could break

   Common themes to consider: Execution Risk, Credibility Window, Commercial Tension,
   Market Position, Technical Debt, Organizational Strain

   QUALITY BAR:
   ✓ Specific (numbers, timelines, named pressures)
   ✓ Honest (don't sugarcoat vulnerabilities)
   ✓ Evidence-based (cite sources)
   ✗ Generic ("facing market challenges")
   ✗ Safe ("well-positioned for growth")

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
   - icon: 📹 for video, 📄 for article
   - Include title, platform, date, duration (video only), url""")

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
#  Main synthesis entry point
# ═══════════════════════════════════════════════════════════

async def synthesize(
    artifacts: CollectedArtifacts,
    request: ResearchRequest,
    has_person_content: bool = True,
) -> ResearchResponse:
    """
    Two-pass Gemini synthesis:
      Call 1: Quick Prep signals + Executive Orientation
      Call 2: Full Dossier (6 sections)
    Both calls receive the same source material.
    """
    source_material = _build_source_material(artifacts)
    loop = asyncio.get_event_loop()

    # Build both prompts upfront
    quick_system = _build_quick_prep_system(request, has_person_content)
    quick_content = f"Analyze the following sources and extract executive intelligence:\n\n{source_material}"
    dossier_system = _build_dossier_system(request, has_person_content)
    dossier_content = f"Build a full executive intelligence dossier from these sources:\n\n{source_material}"

    # ── Run both Gemini calls in parallel ──
    async def _run_quick_prep():
        return await loop.run_in_executor(
            None, _call_gemini_sync, quick_system, quick_content
        )

    async def _run_dossier():
        return await loop.run_in_executor(
            None, _call_gemini_sync, dossier_system, dossier_content
        )

    quick_result, dossier_result = await asyncio.gather(
        _run_quick_prep(),
        _run_dossier(),
        return_exceptions=True,
    )

    # ── Parse Quick Prep ──
    prior_role = None
    signals = []
    executive_orientation = None
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
            elif isinstance(parsed, list):
                raw_signals = parsed
            else:
                raw_signals = []
            signals = _build_signals(raw_signals)
            for i, sig in enumerate(signals, 1):
                sig.id = i
        except Exception as e:
            logger.error(f"Quick Prep parse error: {e}")

    # ── Parse Full Dossier ──
    dossier = None
    if isinstance(dossier_result, Exception):
        logger.error(f"Gemini Dossier error: {dossier_result}")
    else:
        try:
            parsed2 = _parse_json_safe(dossier_result.strip())
            if isinstance(parsed2, dict):
                dossier = _build_dossier(parsed2)
            else:
                logger.warning("Dossier call returned non-dict, skipping")
        except Exception as e:
            logger.error(f"Dossier parse error: {e}")

    # ── Compute Research Confidence & Thin Signal Warning ──
    if dossier:
        dossier.research_confidence = _compute_research_confidence(
            artifacts, has_person_content
        )
        # Thin signal warning: fewer than 2 strong sources
        strong_sources = sum(1 for v in artifacts.videos if v.is_person_match) + len(artifacts.articles)
        if strong_sources < 2:
            dossier.thin_signal_warning = (
                "Limited public executive signal. "
                "Analysis primarily based on role context and company information."
            )

    # ── Build response ──
    video_count = len(artifacts.videos)
    article_count = len(artifacts.articles)

    return ResearchResponse(
        person=PersonInfo(
            name=request.target_name,
            title=request.target_title,
            company=request.target_company,
            prior_role=prior_role,
        ),
        executive_orientation=executive_orientation,
        signals=signals,
        dossier=dossier,
        sources_analyzed={
            "videos": video_count,
            "articles": article_count,
        },
        metadata={
            "steps_attempted": artifacts.steps_attempted,
            "total_videos": video_count,
            "total_articles": article_count,
            "videos_with_transcripts": sum(
                1 for v in artifacts.videos if v.transcript_available
            ),
            "article_search_log": [
                entry.model_dump() for entry in artifacts.article_search_log
            ],
        },
    )
