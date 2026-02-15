import time
from typing import List
from models.requests import ResearchRequest
from models.internal import CollectedArtifacts, ScoredVideo
from models.responses import ResearchResponse
from services.youtube_search import search_youtube
from services.youtube_transcript import fetch_transcript
from services.disambiguation import score_and_filter, score_and_filter_podcasts
from services.article_fetcher import search_and_fetch_articles
from services.podcast_search import search_podcast_episodes, scrape_episode_page, fetch_podcast_transcript
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


async def _step0_podcasts(
    request: ResearchRequest,
    artifacts: CollectedArtifacts,
    emit=None,
) -> None:
    """
    Step 0: Podcast episode discovery via Serper + ListenNotes scraping.
    Skipped if no webshare proxy or no target_name.
    """
    if not settings.webshare_proxy_url:
        logger.info("Step 0 skipped: no Webshare proxy configured")
        if emit:
            await emit("log", step="step0", message="Skipped: no proxy configured")
        return

    if not request.target_name:
        logger.info("Step 0 skipped: no target_name")
        if emit:
            await emit("log", step="step0", message="Skipped: no target name")
        return

    if not settings.serper_api_key:
        logger.info("Step 0 skipped: no Serper API key")
        if emit:
            await emit("log", step="step0", message="Skipped: no Serper API key")
        return

    logger.info(f"Step 0: Podcast search for '{request.target_name}'")
    artifacts.steps_attempted.append("step0_podcasts")

    # Search for podcast episodes (Serper first, then direct ListenNotes fallback)
    candidates, search_method = await search_podcast_episodes(
        request.target_name,
        request.target_company,
        max_results=5,
    )

    method_label = {"serper": "Serper", "direct": "ListenNotes direct search", "none": "Serper"}.get(search_method, search_method)
    if emit:
        await emit("log", step="step0", message=f"{method_label} found {len(candidates)} ListenNotes episode(s)")

    if not candidates:
        if emit:
            await emit("log", step="step0", message="No podcast episodes found")
        return

    # Score and filter candidates
    kept, all_scored = score_and_filter_podcasts(
        candidates,
        request.target_name,
        request.target_company,
        max_keep=settings.max_podcast_artifacts,
    )

    if emit:
        for sp in all_scored:
            passed = sp.match_score >= settings.disambiguation_threshold
            status = "\u2713" if passed else "\u2717"
            signals = ", ".join(sp.match_signals) if sp.match_signals else "none"
            await emit("log", step="step0", message=f"{status} \"{sp.title[:70]}\" = {sp.match_score} [{signals}]")
        await emit("log", step="step0", message=f"Kept {len(kept)} of {len(all_scored)} podcast(s) (threshold \u2265 {settings.disambiguation_threshold})")

    # For each accepted episode: scrape page for audio URL, then transcribe
    for podcast in kept:
        # Scrape ListenNotes page for audio URL + metadata
        if emit:
            await emit("log", step="step0", message=f"Scraping ListenNotes page for \"{podcast.title[:50]}\"...")
        scraped = await scrape_episode_page(podcast.url)

        if not scraped or not scraped.get("audio_url"):
            if emit:
                await emit("log", step="step0", message=f"No audio URL found for \"{podcast.title[:50]}\"")
            # Still add to artifacts (without transcript)
            artifacts.podcasts.append(podcast)
            continue

        # Update podcast with scraped metadata
        podcast.audio_url = scraped["audio_url"]
        if scraped.get("audio_length_sec"):
            podcast.audio_length_sec = scraped["audio_length_sec"]
        if scraped.get("podcast_name") and not podcast.podcast_title:
            podcast.podcast_title = scraped["podcast_name"]
        if scraped.get("published_date") and not podcast.published_at:
            podcast.published_at = scraped["published_date"]
        if scraped.get("title") and not podcast.title:
            podcast.title = scraped["title"]

        # Add to artifacts BEFORE transcript fetch so timeout won't lose it
        artifacts.podcasts.append(podcast)

        # Transcribe audio
        if emit:
            await emit("log", step="step0", message=f"Transcribing podcast audio for \"{podcast.title[:50]}\"...")

            async def _on_log_s0(msg):
                await emit("log", step="step0", message=f"  {msg}")

            text, available = await fetch_podcast_transcript(
                podcast.audio_url, podcast.episode_id, on_log=_on_log_s0,
            )
        else:
            text, available = await fetch_podcast_transcript(
                podcast.audio_url, podcast.episode_id,
            )

        podcast.transcript_text = text
        podcast.transcript_available = available

    logger.info(f"Step 0 result: {len(artifacts.podcasts)} podcast(s)")


async def _step1_person_youtube(
    request: ResearchRequest,
    artifacts: CollectedArtifacts,
    emit=None,
) -> float:
    """
    Step 1: Person-level YouTube search.
    Returns the result strength.
    """
    if not request.target_name:
        logger.info("Step 1 skipped: no target_name provided")
        if emit:
            await emit("log", step="step1", message="Skipped: no target name provided")
        return 0.0

    logger.info(f"Step 1: Person-level YouTube for '{request.target_name}'")
    artifacts.steps_attempted.append("step1_person_youtube")

    name = request.target_name
    company = request.target_company
    queries = [
        f'{name} {company} interview',
        f'{name} {company} podcast',
        f'{name} {company}',
        f'{name} keynote',
        f'{company} CEO interview',
    ]

    all_candidates = []
    seen_ids = set()

    for i, query in enumerate(queries):
        if i > 0:
            await asyncio.sleep(1.5)
        candidates = await search_youtube(
            query, max_results=settings.youtube_search_max_results
        )
        new_count = 0
        for c in candidates:
            if c.video_id not in seen_ids:
                seen_ids.add(c.video_id)
                all_candidates.append(c)
                new_count += 1
        if emit:
            await emit("log", step="step1", message=f"Query: \"{query}\" \u2192 {len(candidates)} result(s), {new_count} new")

    if emit:
        await emit("log", step="step1", message=f"Total unique candidates: {len(all_candidates)}")

    step1_videos, all_scored = await score_and_filter(
        all_candidates,
        request.target_name,
        request.target_company,
        max_keep=settings.max_youtube_artifacts,
    )

    if emit:
        for sv in all_scored:
            passed = sv.match_score >= settings.disambiguation_threshold
            status = "\u2713" if passed else "\u2717"
            signals = ", ".join(sv.match_signals) if sv.match_signals else "none"
            await emit("log", step="step1", message=f"{status} \"{sv.title[:70]}\" = {sv.match_score} [{signals}]")
        await emit("log", step="step1", message=f"Kept {len(step1_videos)} of {len(all_scored)} candidates (threshold \u2265 {settings.disambiguation_threshold})")

    # Add videos to artifacts BEFORE transcript fetch so timeout won't lose them
    artifacts.videos.extend(step1_videos)

    # Fetch transcripts for accepted videos
    for video in step1_videos:
        if not video.transcript_text and not video.transcript_available:
            if emit:
                await emit("log", step="step1", message=f"Fetching transcript for {video.video_id}...")
                async def _on_log_s1(msg):
                    await emit("log", step="step1", message=f"  {msg}")
                text, available = await fetch_transcript(video.video_id, on_log=_on_log_s1)
            else:
                text, available = await fetch_transcript(video.video_id)
            video.transcript_text = text
            video.transcript_available = available

    strength = _compute_result_strength(step1_videos)
    logger.info(f"Step 1 result: {len(step1_videos)} videos, strength={strength:.2f}")
    if emit:
        await emit("log", step="step1", message=f"Result strength: {strength:.2f}")
    return strength


async def _step2_company_leadership(
    request: ResearchRequest,
    artifacts: CollectedArtifacts,
    emit=None,
) -> None:
    """
    Step 2: Company leadership YouTube fallback.
    Fills remaining video slots with company-level leadership content.
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
        f'{company} CEO interview',
        f'{company} founder interview',
        f'{company} CFO interview',
        f'{company} CRO interview',
        f'{company} CTO interview',
        f'{company} COO interview',
        f'{company} leadership panel',
        f'{company} conference talk',
    ]

    for i, query in enumerate(queries):
        if i > 0:
            await asyncio.sleep(1.5)
        candidates = await search_youtube(query, max_results=5)
        new_count = 0
        for c in candidates:
            if c.video_id not in seen_ids:
                seen_ids.add(c.video_id)
                all_candidates.append(c)
                new_count += 1
        if emit:
            await emit("log", step="step2", message=f"Query: \"{query}\" \u2192 {len(candidates)} result(s), {new_count} new")

    if emit:
        await emit("log", step="step2", message=f"Total unique candidates: {len(all_candidates)}")

    step2_videos, all_scored = await score_and_filter(
        all_candidates,
        request.target_name,
        request.target_company,
        max_keep=remaining_slots,
    )

    if emit:
        for sv in all_scored:
            passed = sv.match_score >= settings.disambiguation_threshold
            status = "\u2713" if passed else "\u2717"
            signals = ", ".join(sv.match_signals) if sv.match_signals else "none"
            await emit("log", step="step2", message=f"{status} \"{sv.title[:70]}\" = {sv.match_score} [{signals}]")
        await emit("log", step="step2", message=f"Kept {len(step2_videos)} of {len(all_scored)} candidates (threshold \u2265 {settings.disambiguation_threshold})")

    # Add videos to artifacts BEFORE transcript fetch so timeout won't lose them
    artifacts.videos.extend(step2_videos)

    for video in step2_videos:
        if not video.transcript_text:
            if emit:
                await emit("log", step="step2", message=f"Fetching transcript for {video.video_id}...")
                async def _on_log_s2(msg):
                    await emit("log", step="step2", message=f"  {msg}")
                text, available = await fetch_transcript(video.video_id, on_log=_on_log_s2)
            else:
                text, available = await fetch_transcript(video.video_id)
            video.transcript_text = text
            video.transcript_available = available
    logger.info(f"Step 2 result: {len(step2_videos)} additional videos")


async def _step3_articles(
    request: ResearchRequest,
    artifacts: CollectedArtifacts,
    emit=None,
) -> None:
    """
    Step 3: Article search — always runs, collects up to max_article_artifacts.
    """
    logger.info("Step 3: Article search")
    artifacts.steps_attempted.append("step3_articles")

    articles, article_log = await search_and_fetch_articles(
        request.target_company,
        request.target_name,
        max_articles=settings.max_article_artifacts,
    )

    if emit:
        for entry in article_log:
            status_icon = {
                "accepted": "\u2713",
                "rejected_no_company_mention": "\u2717",
                "fetch_failed": "!",
                "skipped_domain": "\u2014",
                "no_results": "\u2014",
                "search_error": "!",
                "rejected_too_old": "\u2717",
                "rejected_duplicate_topic": "\u2717",
            }.get(entry.status, "?")
            url_part = f" {entry.url[:80]}" if entry.url else ""
            source_part = f" [{entry.source}]" if entry.source else ""
            await emit("log", step="step3", message=f"{status_icon} [{entry.status}]{source_part}{url_part} \u2014 {entry.reason}")

    artifacts.article_search_log = article_log
    artifacts.articles.extend(articles)
    logger.info(f"Step 3 result: {len(articles)} article(s) found")


async def run_research(request: ResearchRequest, on_progress=None) -> ResearchResponse:
    """
    Execute the full research pipeline:
      Step 0 (podcasts) -> Step 1 (person YouTube) -> Step 2 (company fallback) -> Step 3 (articles) -> Synthesis
    Enforces pipeline_timeout_sec. Tracks boring-VP and common-name edge cases.
    """
    pipeline_start = time.time()

    async def emit(event_type: str, **kwargs):
        if on_progress:
            await on_progress(event_type, kwargs)

    _emit = emit if on_progress else None

    artifacts = CollectedArtifacts(
        person_name=request.target_name,
        company_name=request.target_company,
        person_title=request.target_title,
    )

    warnings = []
    timed_out = False

    def _elapsed():
        return time.time() - pipeline_start

    def _time_left():
        return max(0, settings.pipeline_timeout_sec - _elapsed())

    # ── Step 0: Podcast Search ──
    await emit("step", step="step0", status="searching", message="Searching for podcast episodes...")
    try:
        await asyncio.wait_for(
            _step0_podcasts(request, artifacts, emit=_emit),
            timeout=_time_left(),
        )
    except asyncio.TimeoutError:
        logger.warning(f"Pipeline timeout during step 0 (podcasts) at {_elapsed():.1f}s")
        timed_out = True
        warnings.append("Partial research \u2014 showing available signals")

    for p in artifacts.podcasts:
        await emit("found", kind="podcast", title=p.title, score=round(p.match_score, 2))
    if artifacts.podcasts:
        await emit("step", step="step0", status="done", message=f"Found {len(artifacts.podcasts)} podcast episode(s)")
    else:
        await emit("step", step="step0", status="done", message="No podcast episodes found")

    # ── Step 1: Person-level YouTube ──
    if not timed_out:
        await emit("step", step="step1", status="searching", message="Searching for person videos...")
    try:
        step1_strength = await asyncio.wait_for(
            _step1_person_youtube(request, artifacts, emit=_emit),
            timeout=_time_left(),
        ) if not timed_out else 0.0
    except asyncio.TimeoutError:
        logger.warning(f"Pipeline timeout during step 1 at {_elapsed():.1f}s")
        step1_strength = 0.0
        timed_out = True
        warnings.append("Partial research \u2014 showing available signals")

    person_video_count = len(artifacts.videos)
    for v in artifacts.videos:
        await emit("found", kind="video", title=v.title, score=round(v.match_score, 2))
    await emit("step", step="step1", status="done", message=f"Found {person_video_count} person video(s)")

    # ── Step 2: Company leadership fallback (if Step 1 weak) ──
    if not timed_out and step1_strength < settings.weak_result_threshold:
        prev_count = len(artifacts.videos)
        await emit("step", step="step2", status="searching", message="Searching for company leadership videos...")
        try:
            await asyncio.wait_for(
                _step2_company_leadership(request, artifacts, emit=_emit),
                timeout=_time_left(),
            )
        except asyncio.TimeoutError:
            logger.warning(f"Pipeline timeout during step 2 at {_elapsed():.1f}s")
            timed_out = True
            warnings.append("Partial research \u2014 showing available signals")
        new_videos = artifacts.videos[prev_count:]
        for v in new_videos:
            await emit("found", kind="video", title=v.title, score=round(v.match_score, 2))
        await emit("step", step="step2", status="done", message=f"Found {len(new_videos)} leadership video(s)")

    # Detect "boring VP" — no person-level interviews found
    person_podcast_count = sum(1 for p in artifacts.podcasts if p.is_person_match)
    has_person_content = person_video_count > 0 or person_podcast_count > 0
    if not has_person_content and request.target_name:
        warnings.append("No direct interviews found \u2014 signals derived from company-level sources and role context")

    # ── Step 3: Articles (always runs, even after timeout — articles are fast) ──
    await emit("step", step="step3", status="searching", message="Searching for articles...")
    try:
        await asyncio.wait_for(
            _step3_articles(request, artifacts, emit=_emit),
            timeout=max(_time_left(), 15),  # give articles at least 15s even after timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"Pipeline timeout during step 3 at {_elapsed():.1f}s")
        if "Partial research" not in str(warnings):
            warnings.append("Partial research \u2014 showing available signals")

    for a in artifacts.articles:
        await emit("found", kind="article", title=a.title)
    if artifacts.articles:
        await emit("step", step="step3", status="done", message=f"Found {len(artifacts.articles)} article(s)")
    else:
        await emit("step", step="step3", status="done", message="Using video sources only")

    # ── Synthesis ──
    await emit("step", step="synthesis", status="running", message="Synthesizing intelligence...")
    podcast_count = len(artifacts.podcasts)
    video_count = len(artifacts.videos)
    article_count = len(artifacts.articles)
    transcript_count = sum(1 for v in artifacts.videos if v.transcript_available)
    podcast_transcript_count = sum(1 for p in artifacts.podcasts if p.transcript_available)
    if on_progress:
        parts = []
        if podcast_count:
            parts.append(f"{podcast_count} podcast(s) ({podcast_transcript_count} transcribed)")
        parts.append(f"{video_count} video(s) ({transcript_count} with transcripts)")
        parts.append(f"{article_count} article(s)")
        await emit("log", step="synthesis", message=f"Analyzing {' + '.join(parts)}...")

    logger.info(
        f"Synthesis: {podcast_count} podcasts, {video_count} videos, "
        f"{article_count} articles, "
        f"steps={artifacts.steps_attempted}"
    )
    artifacts.steps_attempted.append("synthesis")

    response = await synthesize(
        artifacts,
        request,
        has_person_content=has_person_content,
    )

    # Attach warnings
    response.warnings = warnings

    signal_count = len(response.signals)
    await emit("step", step="synthesis", status="done", message=f"Found {signal_count} signal{'s' if signal_count != 1 else ''}")

    elapsed = round(_elapsed(), 1)
    logger.info(f"Pipeline completed in {elapsed}s: {signal_count} signals, {podcast_count} podcasts, {video_count} videos, {article_count} articles")
    return response
