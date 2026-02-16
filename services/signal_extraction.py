"""
Per-source signal extraction pipeline.

Each source independently extracts structured signals before synthesis.
All extractions run in parallel via asyncio.gather.

Architecture: fetch -> filter -> extract signals (per source) -> synthesis
"""
from google import genai
from google.genai import types
from models.internal import CollectedArtifacts, ExtractedSignal
from config import settings
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.gemini_api_key)

_SYSTEM_PROMPT = """You extract structured intelligence signals from a single source about an executive or company.

Return a JSON array of 2-4 signal objects. Each signal:
{
  "signal_type": "quote" | "theme" | "fact" | "opinion",
  "content": "max 30 words - the key insight",
  "quote": "verbatim quote if available, 15-40 words, or null",
  "timestamp": "MM:SS if from audio/video, or null",
  "confidence": 0.0-1.0
}

Signal types:
- "quote": A direct verbatim statement from the person
- "theme": A recurring theme or priority they express
- "fact": A verifiable data point (number, date, milestone)
- "opinion": A stated belief, stance, or perspective

RULES:
- Extract what's ACTUALLY in the source - don't speculate
- Quotes must be verbatim from the text
- confidence = 1.0 for direct statements, 0.7 for inferred, 0.5 for tangential
- Focus on what reveals executive thinking, pressures, and priorities"""


def _extract_sync(source_text: str, source_context: str) -> list:
    """Synchronous Gemini call for signal extraction from one source."""
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=f"{source_context}\n\nSOURCE CONTENT:\n{source_text[:5000]}",
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=2048,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        raw = json.loads(response.text.strip())
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ("signals", "items", "data"):
                if key in raw and isinstance(raw[key], list):
                    return raw[key]
        return []
    except Exception as e:
        logger.warning(f"Signal extraction Gemini call failed: {e}")
        return []


def _parse_signals(raw_list: list) -> list:
    """Parse raw Gemini output into ExtractedSignal objects."""
    signals = []
    for raw in (raw_list or [])[:4]:
        if not isinstance(raw, dict):
            continue
        sig_type = (raw.get("signal_type") or "theme").lower()
        if sig_type not in ("quote", "theme", "fact", "opinion"):
            sig_type = "theme"
        content = (raw.get("content") or "").strip()
        if not content:
            continue
        signals.append(ExtractedSignal(
            signal_type=sig_type,
            content=content,
            quote=(raw.get("quote") or "").strip() or None,
            timestamp=raw.get("timestamp"),
            confidence=min(max(float(raw.get("confidence") or 0.8), 0.0), 1.0),
        ))
    return signals


async def extract_all_signals(artifacts: CollectedArtifacts, emit=None) -> None:
    """
    Extract structured signals from all collected sources in parallel.
    Modifies source objects in-place, adding extracted_signals.
    """
    loop = asyncio.get_event_loop()
    tasks = []
    source_refs = []

    person = artifacts.person_name or "N/A"
    company = artifacts.company_name

    # Build extraction tasks for each source
    for p in artifacts.podcasts:
        text = p.transcript_text or p.description
        if not text:
            continue
        ctx = (
            f"TARGET: {person} at {company}\n"
            f"SOURCE TYPE: Podcast Episode\n"
            f"TITLE: {p.title}\n"
            f"SHOW: {p.podcast_title}\n"
            f"DATE: {p.published_at or 'Unknown'}"
        )
        tasks.append(loop.run_in_executor(None, _extract_sync, text, ctx))
        source_refs.append(p)

    for v in artifacts.videos:
        text = v.transcript_text or v.description
        if not text:
            continue
        ctx = (
            f"TARGET: {person} at {company}\n"
            f"SOURCE TYPE: YouTube Video\n"
            f"TITLE: {v.title}\n"
            f"CHANNEL: {v.channel_title}\n"
            f"DATE: {v.published_at or 'Unknown'}"
        )
        tasks.append(loop.run_in_executor(None, _extract_sync, text, ctx))
        source_refs.append(v)

    for a in artifacts.articles:
        if not a.text:
            continue
        ctx = (
            f"TARGET: {person} at {company}\n"
            f"SOURCE TYPE: Article\n"
            f"TITLE: {a.title}\n"
            f"DATE: {a.published_date or 'Unknown'}"
        )
        tasks.append(loop.run_in_executor(None, _extract_sync, a.text, ctx))
        source_refs.append(a)

    if not tasks:
        logger.info("No sources to extract signals from")
        return

    if emit:
        await emit("log", step="extraction", message=f"Extracting signals from {len(tasks)} source(s)...")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    extracted_count = 0
    for source_obj, result in zip(source_refs, results):
        if isinstance(result, Exception):
            title = getattr(source_obj, 'title', '?')[:50]
            logger.warning(f"Signal extraction failed for '{title}': {result}")
            continue
        signals = _parse_signals(result)
        source_obj.extracted_signals = signals
        extracted_count += len(signals)

    logger.info(f"Signal extraction complete: {extracted_count} signals from {len(tasks)} sources")
    if emit:
        await emit("log", step="extraction", message=f"Extracted {extracted_count} signal(s) from {len(tasks)} source(s)")
