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


async def extract_signals_for_sources(
    sources: list,
    person_name: str,
    company_name: str,
) -> int:
    """
    Extract structured signals from a flat list of source objects.
    Skips sources that already have non-empty extracted_signals (dedup).
    Detects source type from object attributes.
    Returns the count of newly extracted signals.
    """
    loop = asyncio.get_event_loop()
    tasks = []
    source_refs = []

    for source in sources:
        # Skip if already extracted
        if getattr(source, 'extracted_signals', None):
            continue

        # Detect source type by attribute and build context
        if hasattr(source, 'podcast_title'):
            text = source.transcript_text or source.description
            if not text:
                continue
            ctx = (
                f"TARGET: {person_name} at {company_name}\n"
                f"SOURCE TYPE: Podcast Episode\n"
                f"TITLE: {source.title}\n"
                f"SHOW: {source.podcast_title}\n"
                f"DATE: {source.published_at or 'Unknown'}"
            )
        elif hasattr(source, 'channel_title'):
            text = source.transcript_text or source.description
            if not text:
                continue
            ctx = (
                f"TARGET: {person_name} at {company_name}\n"
                f"SOURCE TYPE: YouTube Video\n"
                f"TITLE: {source.title}\n"
                f"CHANNEL: {source.channel_title}\n"
                f"DATE: {source.published_at or 'Unknown'}"
            )
        else:
            if not getattr(source, 'text', None):
                continue
            text = source.text
            ctx = (
                f"TARGET: {person_name} at {company_name}\n"
                f"SOURCE TYPE: Article\n"
                f"TITLE: {source.title}\n"
                f"DATE: {getattr(source, 'published_date', None) or 'Unknown'}"
            )

        tasks.append(loop.run_in_executor(None, _extract_sync, text, ctx))
        source_refs.append(source)

    if not tasks:
        return 0

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

    logger.info(f"Signal extraction: {extracted_count} signals from {len(tasks)} new sources")
    return extracted_count


async def extract_all_signals(artifacts: CollectedArtifacts, emit=None) -> None:
    """
    Extract structured signals from all collected sources in parallel.
    Modifies source objects in-place, adding extracted_signals.
    Skips sources that already have extracted signals.
    """
    all_sources = list(artifacts.podcasts) + list(artifacts.videos) + list(artifacts.articles)
    if not all_sources:
        logger.info("No sources to extract signals from")
        return

    new_count = sum(1 for s in all_sources if not getattr(s, 'extracted_signals', None))
    if new_count == 0:
        logger.info("All sources already have extracted signals — skipping")
        return

    if emit:
        await emit("log", step="extraction", message=f"Extracting signals from {new_count} source(s)...")

    extracted_count = await extract_signals_for_sources(
        all_sources,
        artifacts.person_name or "N/A",
        artifacts.company_name,
    )

    logger.info(f"Signal extraction complete: {extracted_count} signals from {new_count} sources")
    if emit:
        await emit("log", step="extraction", message=f"Extracted {extracted_count} signal(s) from {new_count} source(s)")
