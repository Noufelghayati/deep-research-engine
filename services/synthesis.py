from google import genai
from google.genai import types
from models.internal import CollectedArtifacts
from models.requests import ResearchRequest
from models.responses import (
    ResearchResponse,
    SelectedSource,
    SourceType,
    MatchType,
    TalkingPoint,
)
from config import settings
import json
import re
import asyncio
import logging

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.gemini_api_key)

ROLE_ANGLES = {
    "account_executive": (
        "You are preparing a senior Account Executive for a meeting. "
        "Focus on business outcomes, strategic priorities, competitive "
        "positioning, and deal-relevant talking points. The tone should "
        "be confident and peer-level."
    ),
    "sales_dev_rep": (
        "You are preparing an SDR for initial outreach. Focus on "
        "attention-grabbing insights, recent news the prospect cares "
        "about, and personalized opening angles. The tone should be "
        "concise and curiosity-driven."
    ),
    "customer_success_manager": (
        "You are preparing a CSM for a quarterly review or renewal "
        "conversation. Focus on the customer's stated goals, industry "
        "challenges, and how to deepen the relationship. The tone "
        "should be consultative and empathetic."
    ),
    "executive": (
        "You are preparing a VP/C-level for an executive briefing. "
        "Focus on strategic vision, market trends, and high-level "
        "value alignment. The tone should be concise, strategic, "
        "and board-room appropriate."
    ),
}


def _build_system_prompt(request: ResearchRequest) -> str:
    role_angle = ROLE_ANGLES.get(
        request.user_role.value, ROLE_ANGLES["account_executive"]
    )

    # Determine if we need leadership framing
    framing_rule = ""
    if not request.target_name:
        framing_rule = (
            "\nIMPORTANT FRAMING RULE: Since no specific person was targeted, "
            "frame all insights as company-level observations. Use phrasing like "
            '"Company leadership has emphasized..." or "In recent leadership interviews..."'
        )

    lines = [
        "You are a world-class sales research analyst. Your job is to",
        "synthesize raw source material (YouTube transcripts and articles) into",
        "actionable sales preparation material.",
        "",
        role_angle,
        "",
        f"TARGET COMPANY: {request.target_company}",
    ]
    if request.target_name:
        lines.append(f"TARGET PERSON: {request.target_name}")
    if request.target_title:
        lines.append(f"PERSON TITLE: {request.target_title}")
    if request.your_name:
        lines.append(f"SENDER NAME: {request.your_name}")
    if request.your_company:
        lines.append(f"SENDER COMPANY: {request.your_company}")
    if request.context:
        lines.append(f"CONTEXT: {request.context}")

    lines.append(framing_rule)
    lines.append("")
    lines.append('You MUST return valid JSON with exactly this structure:')
    lines.append("""{
  "pre_read": ["bullet1", "bullet2", ...],
  "talking_points": [
    {"point": "...", "source_url": "...", "timestamp": "MM:SS or null"}
  ],
  "draft_email": "..."
}""")
    lines.append("")
    lines.append("""RULES:
- pre_read: max 7 bullets. Each should be a complete, actionable insight.
  Lead with what the person/company cares about, not generic facts.
  These should be skimmable in under 60 seconds.
- talking_points: 3-6 points. Each MUST cite the source_url it came from.
  If from a YouTube video, include the approximate timestamp (MM:SS format).
  Each point should be a natural conversation opener the seller can actually say.
- draft_email: A personalized first-touch outreach email (3-5 sentences).
  It MUST reference a specific insight from the research.
  It should feel hand-written and subtle, not templated or salesy.
  Do NOT use "[Your Name]" placeholders — use the sender name if provided.""")

    # Framing rule for leadership fallback
    lines.append("""
- CRITICAL: NEVER fabricate information not found in the provided sources.
- If a source is from a company leader (CEO, CRO, etc.) and NOT the target
  person, you MUST frame insights as "Company leadership has emphasized..."
  or "In recent leadership interviews..." — NEVER attribute their statements
  to the target person.
- If sources are thin, be honest about it. Conservative output beats fluff.
- Role-aware framing: CEO-level insights differ from VP-level or RM-level.""")

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
            section += f"Match type: PERSON-LEVEL (features the target person)\n"
        else:
            section += f"Match type: COMPANY-LEVEL (features company leadership)\n"
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
            "No sources were found. Generate a response acknowledging "
            "the lack of sources. Still produce the JSON structure, but "
            "be honest that limited public information was available. "
            "Suggest the seller do manual research on LinkedIn and the "
            "company website."
        )

    return (
        "Analyze the following sources and produce the JSON output:\n\n"
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


def _parse_json_safe(raw: str) -> dict:
    """
    Parse JSON from Gemini, handling common issues:
    - Control characters inside string values (tabs, newlines)
    - Trailing commas
    """
    # First try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Clean control characters inside JSON string values
    # Replace literal newlines/tabs inside strings with spaces
    cleaned = re.sub(r'[\x00-\x1f\x7f]', lambda m: ' ' if m.group() not in ('\n', '\r') else m.group(), raw)

    # More aggressive: replace all control chars except structural newlines
    # by re-encoding with ensure_ascii
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Last resort: strip all control characters except \n and \t,
    # then replace unescaped newlines inside strings
    cleaned = re.sub(r'(?<!\\)\n', ' ', raw)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
    return json.loads(cleaned)


async def synthesize(
    artifacts: CollectedArtifacts,
    request: ResearchRequest,
) -> ResearchResponse:
    """
    Send collected artifacts to Gemini Flash 2.5 for synthesis.
    Parse the JSON response into a ResearchResponse.
    """
    system_prompt = _build_system_prompt(request)
    content_prompt = _build_content_prompt(artifacts)

    # Run the sync Gemini call in a thread to not block the event loop
    try:
        loop = asyncio.get_event_loop()
        raw_text = await loop.run_in_executor(
            None, _call_gemini_sync, system_prompt, content_prompt
        )
        raw_text = raw_text.strip()

        # Handle potential markdown code blocks
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

        parsed = _parse_json_safe(raw_text)

    except json.JSONDecodeError as e:
        logger.error(f"Gemini returned invalid JSON: {e}\nRaw: {raw_text[:500]}")
        parsed = {
            "pre_read": [
                "Research completed but synthesis formatting failed. "
                "Raw sources are available in selected_sources."
            ],
            "talking_points": [],
            "draft_email": "Unable to generate email due to synthesis error.",
        }
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        parsed = {
            "pre_read": [f"Synthesis unavailable: {str(e)}"],
            "talking_points": [],
            "draft_email": "Unable to generate email due to API error.",
        }

    # Build selected_sources from artifacts
    selected_sources = []
    for v in artifacts.videos:
        match_type = MatchType.PERSON if v.is_person_match else MatchType.COMPANY_LEADERSHIP
        selected_sources.append(
            SelectedSource(
                type=SourceType.YOUTUBE,
                match_type=match_type,
                title=v.title,
                url=v.url,
                why_selected=(
                    f"Score {v.match_score}, signals: {', '.join(v.match_signals)}"
                ),
                company_match_signals=v.match_signals,
            )
        )
    for a in artifacts.articles:
        selected_sources.append(
            SelectedSource(
                type=SourceType.ARTICLE,
                match_type=MatchType.COMPANY_CONTEXT,
                title=a.title,
                url=a.url,
                why_selected="Article supplement for company context",
                company_match_signals=["article_search"],
            )
        )

    # Build talking points — handle None values from Gemini
    talking_points = []
    for tp in parsed.get("talking_points", []):
        if not isinstance(tp, dict):
            continue
        talking_points.append(
            TalkingPoint(
                point=tp.get("point") or "",
                source_url=tp.get("source_url") or "",
                timestamp=tp.get("timestamp"),
            )
        )

    return ResearchResponse(
        selected_sources=selected_sources,
        pre_read=parsed.get("pre_read", [])[:7],
        talking_points=talking_points,
        draft_email=parsed.get("draft_email", ""),
        metadata={
            "steps_attempted": artifacts.steps_attempted,
            "total_videos": len(artifacts.videos),
            "total_articles": len(artifacts.articles),
            "videos_with_transcripts": sum(
                1 for v in artifacts.videos if v.transcript_available
            ),
            "article_search_log": [
                entry.model_dump() for entry in artifacts.article_search_log
            ],
        },
    )
