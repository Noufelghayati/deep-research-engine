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
    DossierSource,
    CATEGORY_ICONS,
)
from config import settings
from typing import List
import json
import re
import asyncio
import logging

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.gemini_api_key)

CATEGORY_PRIORITY = ["GROWTH", "CHALLENGE", "MARKET", "PRODUCT", "TRACTION", "BACKGROUND"]


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
    return response.text


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

    # Truncated JSON recovery
    logger.warning("Attempting truncated JSON recovery")
    try:
        depth = 0
        last_complete_end = -1
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
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    last_complete_end = i
        if last_complete_end > 0:
            recovered = cleaned[:last_complete_end + 1]
            if not recovered.rstrip().endswith(']'):
                recovered = recovered.rstrip().rstrip(',') + ']'
            return json.loads(recovered)
    except (json.JSONDecodeError, Exception) as e:
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
#  CALL 1: Quick Prep signals
# ═══════════════════════════════════════════════════════════

def _build_quick_prep_system(request: ResearchRequest, has_person_content: bool) -> str:
    lines = [
        "You are a synthesis engine extracting executive intelligence for B2B sales reps.",
        "",
        "YOUR TASK: Analyze ALL sources, cluster overlapping themes, rank by importance,",
        "then extract UP TO 5 commercial signals. This is SYNTHESIS, not summarization.",
        "",
        "SYNTHESIS PROCESS:",
        "1. Extract all signals from ALL sources",
        "2. Tag each by category",
        "3. Cluster overlapping themes (e.g. 3 articles about same partnership = 1 signal)",
        "4. Rank by: frequency across sources > recency > executive attribution strength",
        "5. Generate final signals from ranked clusters",
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
        lines.append("Derive signals from company-level sources. Focus on signals relevant")
        lines.append(f"to someone in the role of {request.target_title or 'executive'}.")
        lines.append("Never fabricate quotes. Only use verifiable company statements.")

    lines.append("")
    lines.append("""SIGNAL CATEGORIES (use EXACTLY these names and icons):

\U0001f680 GROWTH - Expansion, scaling, hiring
   Examples: "scaling to X stores", "expanding into Y market", "doubled headcount"

\U0001f4b0 MARKET - Partnerships, funding, deals, market moves
   Examples: "partnered with Kroger", "raised $X series B", "entered European market"

\U0001f527 PRODUCT - Tech priorities, operational focus, product strategy
   Examples: "top priority: in-app image quality", "building AI matching engine"

\U0001f6a8 CHALLENGE - Stated problems, pain points
   Examples: "struggling with X", "40% food waste challenge", "supply chain bottleneck"

\U0001f4ca TRACTION - Metrics, growth stats, proof points
   Examples: "70% of users report healthier diets", "3x revenue growth", "$355M saved"

\U0001f3af BACKGROUND - Previous roles, expertise, track record
   Examples: "ex-CMO at Impossible Foods", "15 years in consumer marketing"

STRICT RULES:

1. MAXIMUM 5 signals. Return FEWER if quality threshold not met.
   3 strong signals > 5 weak ones.
2. Each signal = ONE declarative sentence, max 20 words.
3. REQUIRED COMPOSITION (when possible):
   - At least 1 GROWTH signal
   - At least 1 MARKET or PRODUCT signal
   - At least 1 different category
   - 5th signal is optional (only if genuinely strong)
4. Must include specific details: numbers, timelines, percentages, dollar amounts, names.
5. IGNORE: generic statements, philosophy without action, marketing copy, press release fluff.
6. Each signal must answer: "Why does a sales rep care about this?"
7. Prefer person-attributed signals, but allow company-level when directly relevant.
8. Each quote must be 15-40 words from the source material. Drop signal if quote is weak.
9. DEDUPLICATE: If multiple sources mention the same fact, create ONE signal citing the best source.
10. For source, include video/article title, URL, date, and timestamp (MM:SS) for videos.""")

    lines.append("")
    lines.append("""OUTPUT FORMAT (JSON object):
{
  "prior_role": "CMO, Impossible Foods" or null,
  "signals": [
    {
      "category": "GROWTH",
      "signal": "Scaling Flashfood from 2,000 to 2,500 stores by year-end 2026",
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

Return a JSON object with "prior_role" (string or null) and "signals" (array).
If no quality signals found, return {"prior_role": null, "signals": []}.""")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  CALL 2: Full Dossier
# ═══════════════════════════════════════════════════════════

def _build_dossier_system(request: ResearchRequest, has_person_content: bool) -> str:
    lines = [
        "You are a synthesis engine building an executive research dossier for B2B sales prep.",
        "",
        "YOUR TASK: Analyze ALL sources and produce a structured dossier with 5 sections.",
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
        lines.append("WARNING: No direct interviews with this person were found.")
        lines.append("Build the dossier from company-level sources. Focus on role-relevant signals.")
        lines.append("Never fabricate quotes or attribute statements to the target person.")

    lines.append("")
    lines.append("""OUTPUT FORMAT (JSON object):
{
  "background": [
    "President & COO at Flashfood (promoted January 2023)",
    "Previously: Chief Marketing Officer at Impossible Foods (2019-2022)",
    "Led marketing transformation focused on movement-building",
    "15+ years in consumer marketing and sustainability sectors"
  ],
  "strategic_focus": [
    {
      "category": "GROWTH",
      "icon": "\U0001f680",
      "title": "SCALING & EXPANSION",
      "bullets": [
        "Driving growth from 2,000 to 2,500+ stores by EOY 2026, targeting full US saturation [VIDEO, ARTICLE]",
        "Launched Kroger partnership pilot in Richmond market with 16 stores [ARTICLE]"
      ]
    },
    {
      "category": "MARKET",
      "icon": "\U0001f4b0",
      "title": "INVESTMENT PRIORITIES",
      "bullets": [
        "AI platform optimization for marketplace matching [IMPACT REPORT]",
        "Technology infrastructure investments to support rapid scaling [VIDEO]"
      ]
    }
  ],
  "quotes": [
    {
      "topic": "On Food Waste",
      "quote": "The 'unacceptable' 40% of food waste in grocery stores, especially fresh items, is often driven by merchandising expectations.",
      "source": "Jordan Schenck CEO Interview - YouTube - Apr 2, 2025 - 04:00"
    }
  ],
  "momentum": [
    "Released 2024 Impact Report highlighting 70%+ healthier diet adoption (December 2024)",
    "Announced Kroger partnership pilot in Richmond, Virginia with 16 stores (January 2026)",
    "Diverted 30M+ pounds of food in 2025 alone"
  ],
  "sources": [
    {
      "type": "primary",
      "icon": "\U0001f4f9",
      "title": "Jordan Schenck CEO Interview",
      "platform": "YouTube",
      "date": "Apr 2, 2025",
      "duration": "29:00",
      "url": "https://youtube.com/watch?v=..."
    },
    {
      "type": "supporting",
      "icon": "\U0001f4c4",
      "title": "Flashfood 2024 Impact Report",
      "platform": "Yahoo Finance",
      "date": "Dec 2024",
      "duration": null,
      "url": "https://..."
    }
  ]
}

SECTION RULES:

1. BACKGROUND (max 6 bullets):
   - Role, prior companies, years of experience, location
   - No speculation. Only verified facts from sources.
   - Skip fields not found in sources.

2. STRATEGIC FOCUS (3-6 themes):
   - Synthesized across multiple sources (NOT per-source summaries)
   - Group by category with icon. Cite sources inline.
   - May be longer than Quick Prep signals.
   - Use categories: GROWTH, MARKET, PRODUCT, CHALLENGE, TRACTION, BACKGROUND

3. QUOTES (3-5 direct quotes, 15-40 words each):
   - DIRECT quotes only, no paraphrasing
   - Each needs a topic label and source citation with timestamp
   - Reject marketing copy. If no quality quotes exist, return empty array.

4. MOMENTUM (max 6 bullets):
   - Recent company events: partnerships, funding, expansion, product launches, metrics
   - Chronological or thematic organization
   - Don't repeat Quick Prep signals unless adding depth

5. SOURCES (all sources used):
   - type: "primary" (person interviews) or "supporting" (company/press)
   - icon: \U0001f4f9 for video, \U0001f4c4 for article
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

    themes = []
    for t in (raw.get("strategic_focus") or [])[:6]:
        if not isinstance(t, dict):
            continue
        category = (t.get("category") or "").upper()
        themes.append({
            "category": category,
            "icon": t.get("icon") or CATEGORY_ICONS.get(category, ""),
            "title": t.get("title") or "",
            "bullets": [b for b in (t.get("bullets") or []) if isinstance(b, str)],
        })
    strategic_focus = DossierStrategicFocus(themes=themes)

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

    momentum = DossierMomentum(
        bullets=[b for b in (raw.get("momentum") or []) if isinstance(b, str)][:6]
    )

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
        strategic_focus=strategic_focus,
        quotes=quotes,
        momentum=momentum,
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
      Call 1: Quick Prep signals (5 max)
      Call 2: Full Dossier (5 sections)
    Both calls receive the same source material.
    """
    source_material = _build_source_material(artifacts)
    loop = asyncio.get_event_loop()

    # ── Call 1: Quick Prep ──
    quick_system = _build_quick_prep_system(request, has_person_content)
    quick_content = f"Analyze the following sources and extract commercial signals:\n\n{source_material}"

    prior_role = None
    signals = []
    try:
        raw_text = await loop.run_in_executor(
            None, _call_gemini_sync, quick_system, quick_content
        )
        parsed = _parse_json_safe(raw_text.strip())

        if isinstance(parsed, dict):
            prior_role = parsed.get("prior_role")
            raw_signals = parsed.get("signals") or []
        elif isinstance(parsed, list):
            raw_signals = parsed
        else:
            raw_signals = []

        signals = _build_signals(raw_signals)
        for i, sig in enumerate(signals, 1):
            sig.id = i

    except Exception as e:
        logger.error(f"Gemini Quick Prep error: {e}")

    # ── Call 2: Full Dossier ──
    dossier = None
    try:
        dossier_system = _build_dossier_system(request, has_person_content)
        dossier_content = f"Build a full research dossier from these sources:\n\n{source_material}"

        raw_text2 = await loop.run_in_executor(
            None, _call_gemini_sync, dossier_system, dossier_content
        )
        parsed2 = _parse_json_safe(raw_text2.strip())

        if isinstance(parsed2, dict):
            dossier = _build_dossier(parsed2)
        else:
            logger.warning("Dossier call returned non-dict, skipping")

    except Exception as e:
        logger.error(f"Gemini Dossier error: {e}")

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
