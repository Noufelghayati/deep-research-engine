from google import genai
from google.genai import types
from models.internal import CollectedArtifacts
from models.requests import ResearchRequest
from models.responses import (
    ResearchResponse,
    PersonInfo,
    Signal,
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

# Priority order for signal categories
CATEGORY_PRIORITY = ["SCALING", "CHALLENGE", "INVESTING", "PRIORITY", "TRACTION", "BACKGROUND"]


def _build_system_prompt(request: ResearchRequest) -> str:
    lines = [
        "You are extracting sales intelligence for a B2B rep preparing for a call.",
        "",
        "Extract UP TO 5 commercial signals about this person/company. Return in priority order.",
        "",
        f"TARGET COMPANY: {request.target_company}",
    ]
    if request.target_name:
        lines.append(f"TARGET PERSON: {request.target_name}")
    if request.target_title:
        lines.append(f"PERSON TITLE: {request.target_title}")

    lines.append("")
    lines.append("""SIGNAL CATEGORIES (use these icons):

🚀 SCALING - expansion, growth, hiring
   Examples: "scaling to X stores", "expanding into Y", "doubled headcount"

💰 INVESTING - budget, tech builds, funding
   Examples: "investing in AI", "raised $X", "building platform"

🔧 PRIORITY - current focus, top initiative
   Examples: "top priority: X", "focused on Y", "working on Z"

🚨 CHALLENGE - stated problems, pain points
   Examples: "struggling with X", "challenge is Y", "pain point: Z"

📊 TRACTION - metrics, growth stats, proof points
   Examples: "70% of users", "3x revenue growth", "10K customers"

🎯 BACKGROUND - previous company, expertise
   Examples: "ex-CMO at X", "10 years in Y", "previously led Z"

STRICT RULES:

1. Return MAXIMUM 5 signals (fewer if quality threshold not met)
2. Each signal = ONE SENTENCE, max 20 words
3. Must include specific details:
   - Numbers (2,000 → 2,500)
   - Timelines (by EOY 2026, in Q2)
   - Percentages (70%+)
   - Dollar amounts ($10M)
   - Company names (Impossible Foods)

4. IGNORE completely:
   - Generic statements ("passionate about changing the world")
   - Philosophy without action ("believes in sustainability")
   - Company descriptions not tied to a person
   - Anything without a commercial angle
   - Marketing copy or press release fluff

5. Each signal must answer: "Why does an SDR care?"

6. Only show signals that reference leadership voice directly.
   Not generic company stats, not blog fluff.
   If it's not attributable to a person speaking, exclude it.

7. Each quote must be 15-30 words, taken from the source material.
   Reject quotes that are marketing copy, third-person blog voice,
   or contain no concrete detail. If the quote is weak, drop
   the signal entirely. 5 strong signals > 5 weak ones.

8. For the source, include the video/article title, URL, and date.
   For video sources include an approximate timestamp (MM:SS) if possible.

Priority order for signals:
1. SCALING (highest value)
2. CHALLENGE (direct pain)
3. INVESTING (budget signal)
4. PRIORITY (current focus)
5. TRACTION (proof points)
6. BACKGROUND (context)""")

    lines.append("")
    lines.append("""OUTPUT FORMAT (JSON array):
[
  {
    "category": "SCALING",
    "icon": "🚀",
    "signal": "Scaling from 2,000 to 2,500 stores by EOY 2026",
    "quote": "We're in a rapid growth phase, aiming for full saturation across most US states by year-end",
    "source": {
      "type": "video",
      "title": "Jordan Schenck CEO Interview",
      "url": "https://youtube.com/watch?v=abgKopCIDOY",
      "timestamp": "08:30",
      "date": "Jan 14, 2026"
    }
  }
]

Return a JSON array of signal objects. If you cannot find any quality signals, return an empty array [].""")

    return "\n".join(lines)


def _build_content_prompt(artifacts: CollectedArtifacts) -> str:
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
        return (
            "No sources were found. Return an empty JSON array: []"
        )

    return (
        "Analyze the following sources and extract commercial signals as a JSON array:\n\n"
        + "\n\n".join(sections)
    )


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
    # Strip markdown code blocks
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # First try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Clean control characters
    cleaned = re.sub(r'[\x00-\x1f\x7f]', lambda m: ' ' if m.group() not in ('\n', '\r') else m.group(), text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Replace unescaped newlines
    cleaned = re.sub(r'(?<!\\)\n', ' ', text)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Truncated JSON recovery: find the last complete object in the array
    # This handles when Gemini output is cut off mid-object
    logger.warning("Attempting truncated JSON recovery")
    try:
        # Find all complete {...} objects
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
            # Close the array if needed
            if not recovered.rstrip().endswith(']'):
                recovered = recovered.rstrip().rstrip(',') + ']'
            return json.loads(recovered)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Truncated JSON recovery failed: {e}")

    return []


def _build_signals(raw_signals: list, artifacts: CollectedArtifacts) -> List[Signal]:
    """Convert raw Gemini output into Signal objects, enforcing max 5 and quality."""
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


async def synthesize(
    artifacts: CollectedArtifacts,
    request: ResearchRequest,
) -> ResearchResponse:
    """
    Send collected artifacts to Gemini Flash 2.5 for signal extraction.
    Parse the JSON response into a ResearchResponse.
    """
    system_prompt = _build_system_prompt(request)
    content_prompt = _build_content_prompt(artifacts)

    try:
        loop = asyncio.get_event_loop()
        raw_text = await loop.run_in_executor(
            None, _call_gemini_sync, system_prompt, content_prompt
        )
        raw_text = raw_text.strip()
        parsed = _parse_json_safe(raw_text)

        # Gemini may return a dict with a "signals" key or a raw array
        if isinstance(parsed, dict):
            raw_signals = parsed.get("signals", [])
            if not raw_signals and isinstance(parsed, list):
                raw_signals = parsed
        elif isinstance(parsed, list):
            raw_signals = parsed
        else:
            raw_signals = []

    except json.JSONDecodeError as e:
        logger.error(f"Gemini returned invalid JSON: {e}\nRaw: {raw_text[:500]}")
        raw_signals = []
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raw_signals = []

    signals = _build_signals(raw_signals, artifacts)

    # Re-number signals sequentially
    for i, sig in enumerate(signals, 1):
        sig.id = i

    # Sources summary
    video_count = len(artifacts.videos)
    article_count = len(artifacts.articles)
    sources_analyzed = {
        "videos": video_count,
        "articles": article_count,
    }

    return ResearchResponse(
        person=PersonInfo(
            name=request.target_name,
            title=request.target_title,
            company=request.target_company,
        ),
        signals=signals,
        sources_analyzed=sources_analyzed,
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
