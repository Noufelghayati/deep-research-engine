"""
Podcast episode discovery via iTunes/Serper + ListenNotes scraping + Gemini transcription.

Pipeline:
  1a. iTunes/Apple Podcasts Search API (free, no key, returns direct audio URLs)
  1b. Fallback: Serper search  →  site:listennotes.com results (needs scraping)
  2.  Scrape episode page  →  audio URL + metadata  (Serper results only)
  3.  Download audio from CDN  →  trim  →  Gemini transcription
"""

import re
import json
import asyncio
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import httpx
from bs4 import BeautifulSoup
from config import settings
from models.internal import PodcastCandidate
from utils.text import clean_transcript_text
import logging

logger = logging.getLogger(__name__)

_TEMP_DIR = Path(tempfile.gettempdir()) / "josh_audio"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

# Common podcast CDN audio URL patterns
_AUDIO_URL_PATTERNS = [
    re.compile(r'https?://[^\s"\'<>]+\.(?:mp3|m4a|ogg|opus|wav)(?:\?[^\s"\'<>]*)?', re.I),
]

# Domains that host podcast audio files
_PODCAST_CDN_DOMAINS = [
    "anchor.fm", "podcasts.google.com", "traffic.libsyn.com",
    "traffic.megaphone.fm", "cdn.simplecast.com", "media.blubrry.com",
    "chrt.fm", "pdst.fm", "dts.podtrac.com", "chtbl.com",
    "www.buzzsprout.com", "media.transistor.fm", "feeds.soundcloud.com",
    "play.podtrac.com", "audio.buzzsprout.com", "episodes.buzzsprout.com",
    "media.zencastr.com", "stream.redcircle.com", "api.spreaker.com",
    "episodes.castos.com", "content.blubrry.com", "rss.art19.com",
    "tracking.swap.fm", "aphid.fireside.fm", "media.rss.com",
    "podbean.com", "podcastaddict.com", "omnycontent.com",
    "www.podtrac.com", "embed.podcasts.apple.com",
]


def _noop_log(msg):
    pass


# ---------------------------------------------------------------------------
# 1a. Serper search for ListenNotes episodes
# ---------------------------------------------------------------------------

async def _serper_search_podcasts(
    person_name: str,
    company_name: str,
    max_results: int = 5,
) -> List[PodcastCandidate]:
    """
    Use Serper to find ListenNotes episode pages mentioning the person.
    Returns raw PodcastCandidate list (no audio URLs yet — those come from scraping).
    """
    if not settings.serper_api_key:
        logger.info("Serper podcast search skipped: no Serper API key")
        return []

    queries = [
        f'"{person_name}" podcast site:listennotes.com',
        f'"{person_name}" "{company_name}" podcast site:listennotes.com',
    ]

    api_url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }

    candidates = []
    seen_urls = set()

    for query in queries:
        if len(candidates) >= max_results:
            break

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    api_url,
                    json={"q": query, "num": 10},
                    headers=headers,
                )

            if resp.status_code != 200:
                logger.warning(f"Serper podcast search HTTP {resp.status_code} for '{query[:60]}'")
                continue

            data = resp.json()

            for item in data.get("organic", []):
                link = item.get("link", "")
                if not link or link in seen_urls:
                    continue

                # Only keep ListenNotes episode pages
                if "listennotes.com" not in link:
                    continue
                # Skip non-episode pages (e.g. podcast show pages, search pages)
                # Episode URLs look like: listennotes.com/podcasts/show-name/episode-title-IDHERE/
                # or listennotes.com/episodes/...
                if "/podcasts/" not in link and "/episodes/" not in link:
                    continue

                seen_urls.add(link)

                # Extract episode ID from URL (last path segment before trailing slash)
                parts = link.rstrip("/").split("/")
                episode_id = parts[-1] if parts else ""

                title = item.get("title", "")
                snippet = item.get("snippet", "")

                # Try to split "Episode Title - Podcast Name" from Serper title
                podcast_title = ""
                if " - " in title:
                    # ListenNotes titles often: "Episode Title - Show Name | Listen Notes"
                    segments = title.split(" - ")
                    if len(segments) >= 2:
                        # Last segment often has "| Listen Notes"
                        show_part = segments[-1].split("|")[0].strip()
                        podcast_title = show_part
                        title = " - ".join(segments[:-1])
                # Clean "| Listen Notes" from title
                title = re.sub(r'\s*\|\s*Listen\s*Notes\s*$', '', title).strip()

                candidates.append(PodcastCandidate(
                    episode_id=episode_id,
                    title=title,
                    description=snippet,
                    podcast_title=podcast_title,
                    published_at="",
                    audio_url="",
                    link=link,
                ))

                if len(candidates) >= max_results:
                    break

        except Exception as e:
            logger.warning(f"Serper podcast search failed for '{query[:60]}': {e}")

    logger.info(f"Serper podcast search: {len(candidates)} ListenNotes episodes found")
    return candidates


# ---------------------------------------------------------------------------
# 1b. iTunes/Apple Podcasts Search API (fallback)
# ---------------------------------------------------------------------------

async def _itunes_search_podcasts(
    person_name: str,
    company_name: str,
    max_results: int = 5,
) -> List[PodcastCandidate]:
    """
    Search the iTunes/Apple Podcasts API for podcast episodes.
    Free, no API key needed. Returns direct audio URLs.

    Fallback when Serper returns 0 ListenNotes results.
    """
    queries = [
        person_name,
        f"{person_name} {company_name}",
    ]

    candidates = []
    seen_ids = set()

    for query in queries:
        if len(candidates) >= max_results:
            break

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://itunes.apple.com/search",
                    params={
                        "term": query,
                        "entity": "podcastEpisode",
                        "limit": 15,
                    },
                )

            if resp.status_code != 200:
                logger.warning(f"iTunes podcast search HTTP {resp.status_code} for '{query[:40]}'")
                continue

            data = resp.json()
            results = data.get("results", [])
            logger.info(f"iTunes podcast search '{query[:40]}': {len(results)} results")

            for item in results:
                track_id = str(item.get("trackId", ""))
                if not track_id or track_id in seen_ids:
                    continue
                seen_ids.add(track_id)

                audio_url = item.get("episodeUrl", "")
                title = item.get("trackName", "")
                podcast_title = item.get("collectionName", "")
                description = item.get("description", "") or item.get("shortDescription", "")
                release_date = item.get("releaseDate", "")[:10]  # "2022-06-13T09:00:00Z" → "2022-06-13"

                # Build a viewable URL (Apple Podcasts)
                collection_id = item.get("collectionId", "")
                link = f"https://podcasts.apple.com/podcast/id{collection_id}?i={track_id}" if collection_id else ""

                candidates.append(PodcastCandidate(
                    episode_id=f"itunes_{track_id}",
                    title=title,
                    description=description[:500] if description else "",
                    podcast_title=podcast_title,
                    published_at=release_date,
                    audio_url=audio_url,  # iTunes already gives us the audio URL!
                    link=link,
                ))

                if len(candidates) >= max_results:
                    break

        except Exception as e:
            logger.warning(f"iTunes podcast search failed for '{query[:40]}': {e}")

    logger.info(f"iTunes podcast search: {len(candidates)} episodes found")
    return candidates


# ---------------------------------------------------------------------------
# 1. Combined podcast search (iTunes → Serper fallback)
# ---------------------------------------------------------------------------

async def search_podcast_episodes(
    person_name: str,
    company_name: str,
    max_results: int = 5,
) -> Tuple[List[PodcastCandidate], str]:
    """
    Search for podcast episodes mentioning the person.
    Tries iTunes first (free, direct audio URLs); if 0 results, falls back to
    Serper (site:listennotes.com, requires scraping for audio URLs).

    Returns (candidates, search_method) where search_method is 'itunes' or 'serper'.
    """
    # Try iTunes first (gives direct audio URLs, no scraping needed)
    candidates = await _itunes_search_podcasts(person_name, company_name, max_results)
    if candidates:
        return candidates, "itunes"

    # Fallback: Serper → ListenNotes (needs scraping for audio URLs)
    logger.info("iTunes returned 0 podcast results, trying Serper podcast search...")
    candidates = await _serper_search_podcasts(person_name, company_name, max_results)
    if candidates:
        return candidates, "serper"

    return [], "none"


# ---------------------------------------------------------------------------
# 2. Scrape ListenNotes episode page for audio URL + metadata
# ---------------------------------------------------------------------------

async def scrape_episode_page(url: str) -> Optional[dict]:
    """
    Fetch a ListenNotes episode page via Webshare proxy and extract:
      - audio_url: direct audio file URL
      - title, podcast_name, description, published_date

    Returns dict or None on failure.
    """
    if not settings.webshare_proxy_url:
        logger.info("Podcast scrape skipped: no Webshare proxy configured")
        return None

    try:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers=_HEADERS,
            proxy=settings.webshare_proxy_url,
        ) as client:
            resp = await client.get(url)

        if resp.status_code != 200:
            logger.warning(f"ListenNotes page returned HTTP {resp.status_code}: {url}")
            return None

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        result = {
            "audio_url": "",
            "title": "",
            "podcast_name": "",
            "description": "",
            "published_date": "",
            "audio_length_sec": 0,
        }

        # --- Extract metadata ---

        # Title from og:title or <title>
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            result["title"] = og_title["content"].strip()
        elif soup.title:
            result["title"] = soup.title.get_text(strip=True)

        # Description from og:description
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            result["description"] = og_desc["content"].strip()

        # Published date from meta tags
        for meta_name in ["article:published_time", "datePublished", "publish_date"]:
            meta = soup.find("meta", property=meta_name) or soup.find("meta", attrs={"name": meta_name})
            if meta and meta.get("content"):
                result["published_date"] = meta["content"].strip()[:10]
                break

        # --- Extract audio URL (multiple strategies) ---

        audio_url = ""

        # Strategy 1: <audio> tag with src
        audio_tag = soup.find("audio")
        if audio_tag:
            src = audio_tag.get("src", "")
            if src:
                audio_url = src
            else:
                source_tag = audio_tag.find("source")
                if source_tag and source_tag.get("src"):
                    audio_url = source_tag["src"]

        # Strategy 2: og:audio meta tag
        if not audio_url:
            og_audio = soup.find("meta", property="og:audio")
            if og_audio and og_audio.get("content"):
                audio_url = og_audio["content"]

        # Strategy 3: twitter:player:stream
        if not audio_url:
            tw_stream = soup.find("meta", attrs={"name": "twitter:player:stream"})
            if tw_stream and tw_stream.get("content"):
                audio_url = tw_stream["content"]

        # Strategy 4: JSON-LD PodcastEpisode
        if not audio_url:
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    import json
                    ld = json.loads(script.string or "")
                    # Could be a list or a single object
                    items = ld if isinstance(ld, list) else [ld]
                    for item in items:
                        if item.get("@type") in ("PodcastEpisode", "AudioObject", "Episode"):
                            content_url = item.get("contentUrl") or item.get("url", "")
                            if content_url and any(ext in content_url.lower() for ext in [".mp3", ".m4a", ".ogg", ".opus"]):
                                audio_url = content_url
                                break
                            # Check associatedMedia or audio
                            media = item.get("associatedMedia") or item.get("audio")
                            if isinstance(media, dict):
                                content_url = media.get("contentUrl", "")
                                if content_url:
                                    audio_url = content_url
                                    break
                        # Also check for podcast_name in JSON-LD
                        if item.get("@type") == "PodcastEpisode":
                            part_of = item.get("partOfSeries") or {}
                            if isinstance(part_of, dict) and part_of.get("name"):
                                result["podcast_name"] = part_of["name"]
                            if item.get("datePublished") and not result["published_date"]:
                                result["published_date"] = str(item["datePublished"])[:10]
                except Exception:
                    continue

        # Strategy 5: Regex scan for known podcast CDN audio URLs in HTML
        if not audio_url:
            for pattern in _AUDIO_URL_PATTERNS:
                matches = pattern.findall(html)
                for m in matches:
                    # Prefer URLs from known podcast CDN domains
                    if any(cdn in m for cdn in _PODCAST_CDN_DOMAINS):
                        audio_url = m
                        break
                if audio_url:
                    break
            # If no CDN match, take first .mp3/.m4a URL found
            if not audio_url:
                for pattern in _AUDIO_URL_PATTERNS:
                    matches = pattern.findall(html)
                    if matches:
                        audio_url = matches[0]
                        break

        # Clean up audio URL
        if audio_url:
            # Remove HTML entities
            audio_url = audio_url.replace("&amp;", "&")
            result["audio_url"] = audio_url

        # Try to extract audio duration from page
        # ListenNotes often has duration in the page text like "1 hr 23 min" or "45 min"
        duration_match = re.search(r'(\d+)\s*hr?\s*(\d+)\s*min', html)
        if duration_match:
            hours = int(duration_match.group(1))
            minutes = int(duration_match.group(2))
            result["audio_length_sec"] = hours * 3600 + minutes * 60
        else:
            duration_match = re.search(r'(\d+)\s*min', html)
            if duration_match:
                result["audio_length_sec"] = int(duration_match.group(1)) * 60

        logger.info(
            f"Scraped ListenNotes: audio_url={'found' if audio_url else 'MISSING'}, "
            f"title='{result['title'][:50]}', duration={result['audio_length_sec']}s"
        )
        return result

    except Exception as e:
        logger.warning(f"Failed to scrape ListenNotes page {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# 3. Download podcast audio + transcribe with Gemini
# ---------------------------------------------------------------------------

def _download_podcast_audio_sync(
    audio_url: str,
    episode_id: str,
    max_duration_sec: int,
    on_log=None,
) -> Optional[Path]:
    """
    Download podcast audio from CDN. No proxy needed (public URLs).
    Returns path to downloaded file.
    """
    log = on_log or _noop_log

    _TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Determine file extension from URL
    ext = "mp3"  # default
    url_lower = audio_url.lower()
    for e in ["m4a", "ogg", "opus", "wav", "mp3"]:
        if f".{e}" in url_lower:
            ext = e
            break

    safe_id = re.sub(r'[^\w\-]', '_', episode_id)[:80]
    audio_path = _TEMP_DIR / f"podcast_{safe_id}.{ext}"

    # Clean up stale file
    if audio_path.exists():
        audio_path.unlink()

    log(f"Downloading podcast audio...")
    logger.info(f"Downloading podcast audio: {audio_url[:120]}")

    try:
        import httpx as httpx_sync
        with httpx_sync.Client(
            timeout=60.0,
            follow_redirects=True,
            headers={
                "User-Agent": _HEADERS["User-Agent"],
                "Accept": "*/*",
            },
        ) as client:
            with client.stream("GET", audio_url) as resp:
                if resp.status_code != 200:
                    log(f"Audio download failed: HTTP {resp.status_code}")
                    logger.warning(f"Podcast audio HTTP {resp.status_code}: {audio_url[:100]}")
                    return None

                # Check content-length to enforce 50MB guard
                content_length = int(resp.headers.get("content-length", 0))
                max_bytes = 50 * 1024 * 1024  # 50MB
                if content_length > max_bytes:
                    log(f"Audio too large ({content_length // (1024*1024)}MB > 50MB limit)")
                    logger.warning(f"Podcast audio too large: {content_length} bytes")
                    return None

                downloaded = 0
                with open(audio_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            log(f"Audio exceeded 50MB during download, aborting")
                            f.close()
                            audio_path.unlink(missing_ok=True)
                            return None
                        f.write(chunk)

        size_kb = audio_path.stat().st_size // 1024
        log(f"Audio downloaded: {size_kb:,}KB")
        logger.info(f"Podcast audio downloaded: {audio_path.name} ({size_kb}KB)")

        # Trim if needed
        if max_duration_sec > 0:
            from services.youtube_transcript import _trim_audio
            audio_path = _trim_audio(audio_path, max_duration_sec, on_log=log)

        return audio_path

    except Exception as e:
        log(f"Audio download failed: {str(e)[:100]}")
        logger.warning(f"Podcast audio download failed: {e}")
        audio_path.unlink(missing_ok=True)
        return None


def _fetch_podcast_transcript_sync(
    audio_url: str,
    episode_id: str,
    on_log=None,
) -> Tuple[Optional[str], bool]:
    """
    Download podcast audio + transcribe with Gemini (sync, runs in thread).
    Returns (transcript_text, is_available).
    """
    log = on_log or _noop_log

    audio_path = _download_podcast_audio_sync(
        audio_url,
        episode_id,
        max_duration_sec=settings.max_podcast_audio_duration_sec,
        on_log=log,
    )

    if not audio_path:
        return None, False

    try:
        from services.youtube_transcript import _transcribe_with_gemini_sync
        text = _transcribe_with_gemini_sync(audio_path, on_log=log)

        if text:
            text = clean_transcript_text(text)
            if len(text) > settings.max_transcript_chars:
                text = text[: settings.max_transcript_chars] + "... [truncated]"
            log(f"Podcast transcription complete: {len(text):,} chars")
            logger.info(f"Podcast transcript for {episode_id}: {len(text)} chars")
            return text, True

        log("Podcast transcription returned empty")
        return None, False

    finally:
        try:
            if audio_path and audio_path.exists():
                audio_path.unlink()
                logger.debug(f"Cleaned up temp podcast audio: {audio_path}")
        except Exception:
            pass


async def fetch_podcast_transcript(
    audio_url: str,
    episode_id: str,
    on_log=None,
) -> Tuple[Optional[str], bool]:
    """
    Async wrapper: download podcast audio + Gemini transcription.
    Returns (transcript_text, is_available).
    """
    import queue as queue_mod

    if on_log:
        log_q = queue_mod.Queue()

        def sync_log(msg):
            log_q.put(msg)

        async def run_with_log_bridge():
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                None, _fetch_podcast_transcript_sync, audio_url, episode_id, sync_log,
            )

            while not future.done():
                while True:
                    try:
                        msg = log_q.get_nowait()
                        await on_log(msg)
                    except queue_mod.Empty:
                        break
                await asyncio.sleep(0.3)

            # Drain remaining
            while True:
                try:
                    msg = log_q.get_nowait()
                    await on_log(msg)
                except queue_mod.Empty:
                    break

            return future.result()

        try:
            return await asyncio.wait_for(run_with_log_bridge(), timeout=300)
        except asyncio.TimeoutError:
            await on_log("Podcast transcript fetch timed out after 5 minutes")
            logger.warning(f"Podcast transcript timed out for {episode_id}")
            return None, False
    else:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    _fetch_podcast_transcript_sync, audio_url, episode_id,
                ),
                timeout=300,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Podcast transcript timed out for {episode_id}")
            return None, False
