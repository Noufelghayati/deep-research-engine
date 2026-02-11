from typing import List
from models.requests import ResearchRequest
from models.internal import CollectedArtifacts, ScoredVideo
from models.responses import ResearchResponse
from services.youtube_search import search_youtube
from services.youtube_transcript import fetch_transcript
from services.disambiguation import score_and_filter
from services.article_fetcher import search_and_fetch_article
from services.synthesis import synthesize
from config import settings
import asyncio
import logging

logger = logging.getLogger(__name__)


def _compute_result_strength(videos: List[ScoredVideo]) -> float:
    """
    Normalized quality score for collected YouTube videos.
    Videos with transcripts count full score; without count at 60%.
    """
    if not videos:
        return 0.0

    total = 0.0
    for v in videos:
        weight = 1.0 if v.transcript_available else 0.6
        total += v.match_score * weight

    max_possible = settings.max_youtube_artifacts * 1.0
    return min(total / max_possible, 1.0)


async def _step1_person_youtube(
    request: ResearchRequest,
    artifacts: CollectedArtifacts,
) -> float:
    """
    Step 1: Person-level YouTube search.
    Returns the result strength.
    """
    if not request.target_name:
        logger.info("Step 1 skipped: no target_name provided")
        return 0.0

    logger.info(f"Step 1: Person-level YouTube for '{request.target_name}'")
    artifacts.steps_attempted.append("step1_person_youtube")

    queries = [
        f'{request.target_name} {request.target_company} company interview',
        f'{request.target_name} {request.target_company} company podcast',
        f'{request.target_name} {request.target_company} company keynote',
    ]

    all_candidates = []
    seen_ids = set()

    for i, query in enumerate(queries):
        if i > 0:
            await asyncio.sleep(2.0)
        candidates = await search_youtube(
            query, max_results=settings.youtube_search_max_results
        )
        for c in candidates:
            if c.video_id not in seen_ids:
                seen_ids.add(c.video_id)
                all_candidates.append(c)

    step1_videos = await score_and_filter(
        all_candidates,
        request.target_name,
        request.target_company,
        max_keep=2,
    )

    # Ensure transcripts are fetched for accepted videos
    for video in step1_videos:
        if not video.transcript_text and not video.transcript_available:
            text, available = await fetch_transcript(video.video_id)
            video.transcript_text = text
            video.transcript_available = available

    artifacts.videos.extend(step1_videos)
    strength = _compute_result_strength(step1_videos)
    logger.info(f"Step 1 result: {len(step1_videos)} videos, strength={strength:.2f}")
    return strength


async def _step2_company_leadership(
    request: ResearchRequest,
    artifacts: CollectedArtifacts,
) -> None:
    """
    Step 2: Company leadership YouTube fallback.
    Searches for CEO, CRO, CFO, CTO, COO, CMO, CIO, CSO, CHRO interviews.
    """
    remaining_slots = settings.max_youtube_artifacts - len(artifacts.videos)
    if remaining_slots <= 0:
        return

    logger.info(f"Step 2: Company leadership fallback for '{request.target_company}'")
    artifacts.steps_attempted.append("step2_company_leadership")

    all_candidates = []
    seen_ids = {v.video_id for v in artifacts.videos}

    company = request.target_company
    queries = [
        f'{company} company CEO interview',
        f'{company} company founder interview',
        f'{company} company CFO interview',
        f'{company} company CRO interview',
        f'{company} company CTO interview',
        f'{company} company COO interview',
        f'{company} company CMO interview',
        f'{company} company leadership panel',
        f'{company} company conference talk',
    ]

    for i, query in enumerate(queries):
        if i > 0:
            await asyncio.sleep(2.0)  # DDG rate-limit avoidance
        candidates = await search_youtube(query, max_results=5)
        for c in candidates:
            if c.video_id not in seen_ids:
                seen_ids.add(c.video_id)
                all_candidates.append(c)

    step2_videos = await score_and_filter(
        all_candidates,
        request.target_name,
        request.target_company,
        max_keep=remaining_slots,
    )

    for video in step2_videos:
        if not video.transcript_text:
            text, available = await fetch_transcript(video.video_id)
            video.transcript_text = text
            video.transcript_available = available

    artifacts.videos.extend(step2_videos)
    logger.info(f"Step 2 result: {len(step2_videos)} additional videos")


async def _step3_article_supplement(
    request: ResearchRequest,
    artifacts: CollectedArtifacts,
) -> None:
    """
    Step 3: Article fallback/supplement.
    Used when YouTube content is thin or absent.
    """
    combined_strength = _compute_result_strength(artifacts.videos)
    total_artifacts = len(artifacts.videos) + len(artifacts.articles)
    max_total = settings.max_youtube_artifacts + settings.max_article_artifacts

    needs_article = (
        total_artifacts < max_total
        and (combined_strength < 0.7 or len(artifacts.videos) == 0)
    )

    if not needs_article:
        return

    logger.info("Step 3: Article supplement")
    artifacts.steps_attempted.append("step3_article_fallback")

    article, article_log = await search_and_fetch_article(
        request.target_company,
        request.target_name,
    )
    artifacts.article_search_log = article_log
    if article:
        artifacts.articles.append(article)
        logger.info(f"Step 3 result: article found ({article.content_length_chars} chars)")
    else:
        logger.info("Step 3 result: no article found")


async def run_research(request: ResearchRequest) -> ResearchResponse:
    """
    Execute the full research decision tree:
      Step 1 -> Step 2 (if weak) -> Step 3 (if thin) -> Synthesis
    """
    artifacts = CollectedArtifacts(
        person_name=request.target_name,
        company_name=request.target_company,
        person_title=request.target_title,
    )

    # Step 1: Person-level YouTube
    step1_strength = await _step1_person_youtube(request, artifacts)

    # Step 2: Company leadership fallback (if Step 1 weak)
    if step1_strength < settings.weak_result_threshold:
        await _step2_company_leadership(request, artifacts)

    # Step 3: Article supplement
    await _step3_article_supplement(request, artifacts)

    # Synthesis
    logger.info(
        f"Synthesis: {len(artifacts.videos)} videos, "
        f"{len(artifacts.articles)} articles, "
        f"steps={artifacts.steps_attempted}"
    )
    artifacts.steps_attempted.append("synthesis")

    response = await synthesize(artifacts, request)
    return response
