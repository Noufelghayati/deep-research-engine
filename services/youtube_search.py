import re
import logging
from typing import List, Optional
from urllib.parse import unquote, quote_plus

import httpx
from bs4 import BeautifulSoup

from models.internal import YouTubeCandidate
from config import settings

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}

# Regex to extract YouTube video ID from various URL formats
_YT_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/)"
    r"([a-zA-Z0-9_-]{11})"
)


def _extract_video_id(url: str) -> Optional[str]:
    """Extract the 11-char video ID from a YouTube URL."""
    m = _YT_VIDEO_ID_RE.search(url)
    return m.group(1) if m else None


async def _search_brave(query: str, max_results: int) -> List[YouTubeCandidate]:
    """Search Brave for YouTube videos."""
    search_url = f"https://search.brave.com/search?q={quote_plus(query)}&tf=py"

    try:
        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True, headers=_HEADERS
        ) as client:
            resp = await client.get(search_url)

        if resp.status_code != 200:
            logger.warning(f"Brave returned status {resp.status_code}")
            return []
    except Exception as e:
        logger.warning(f"Brave search failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    candidates = []
    seen_ids = set()

    # Brave embeds YouTube URLs directly in result links
    for link in soup.select('a[href*="youtube.com/watch"]'):
        if len(candidates) >= max_results:
            break

        href = link.get("href", "")
        video_id = _extract_video_id(href)
        if not video_id or video_id in seen_ids:
            continue
        seen_ids.add(video_id)

        title = link.get_text(strip=True) or "Unknown Title"
        # Clean up Brave's title format ("YouTubeyoutube.com· watch..." prefix)
        title = re.sub(r"^YouTubeyoutube\.com[^\w]*watch\s*", "", title)
        # Skip duration-only links (e.g., "13:09")
        if re.match(r"^\d{1,2}:\d{2}$", title):
            continue

        # Try to get description from sibling/parent snippet
        description = ""
        parent = link.find_parent("div", class_=True)
        if parent:
            snippet = parent.select_one("div.snippet-description, p.snippet-description")
            if snippet:
                description = snippet.get_text(strip=True)

        candidates.append(
            YouTubeCandidate(
                video_id=video_id,
                title=title,
                description=description,
                channel_title="Unknown Channel",
                published_at="",
            )
        )

    return candidates


async def _search_ddg(query: str, max_results: int) -> List[YouTubeCandidate]:
    """Fallback: search DDG HTML lite for YouTube videos."""
    ddg_query = f"site:youtube.com {query}"
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(ddg_query)}&df=y"

    try:
        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True, headers=_HEADERS
        ) as client:
            resp = await client.get(search_url)

        if resp.status_code != 200:
            return []
    except Exception:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    candidates = []
    seen_ids = set()

    for result in soup.select("div.result"):
        if len(candidates) >= max_results:
            break

        link_tag = result.select_one("a.result__a")
        if not link_tag:
            continue

        href = link_tag.get("href", "")
        if not href:
            continue

        if "uddg=" in href:
            actual_url = unquote(href.split("uddg=")[1].split("&")[0])
        elif href.startswith("http"):
            actual_url = href
        else:
            continue

        video_id = _extract_video_id(actual_url)
        if not video_id or video_id in seen_ids:
            continue
        seen_ids.add(video_id)

        title = link_tag.get_text(strip=True) or "Unknown Title"
        snippet_tag = result.select_one("a.result__snippet")
        description = snippet_tag.get_text(strip=True) if snippet_tag else ""

        candidates.append(
            YouTubeCandidate(
                video_id=video_id,
                title=title,
                description=description,
                channel_title="Unknown Channel",
                published_at="",
            )
        )

    return candidates


async def _search_serper(query: str, max_results: int) -> List[YouTubeCandidate]:
    """Last fallback: Serper.dev video search API."""
    if not settings.serper_api_key:
        logger.warning("Serper API not configured, skipping video fallback")
        return []

    api_url = "https://google.serper.dev/videos"
    payload = {"q": query, "num": max_results, "tbs": "qdr:y"}
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(api_url, json=payload, headers=headers)

        if resp.status_code != 200:
            logger.warning(f"Serper videos API returned HTTP {resp.status_code}")
            return []

        data = resp.json()
        candidates = []
        seen_ids = set()

        for item in data.get("videos", []):
            link = item.get("link", "")
            video_id = _extract_video_id(link)
            if not video_id or video_id in seen_ids:
                continue
            seen_ids.add(video_id)

            candidates.append(
                YouTubeCandidate(
                    video_id=video_id,
                    title=item.get("title", "Unknown Title"),
                    description=item.get("snippet", ""),
                    channel_title=item.get("channel", "Unknown Channel"),
                    published_at=item.get("date", ""),
                )
            )

            if len(candidates) >= max_results:
                break

        return candidates

    except Exception as e:
        logger.warning(f"Serper video search failed: {e}")
        return []


async def search_youtube(query: str, max_results: int = 5) -> List[YouTubeCandidate]:
    """
    Search for YouTube videos.
    Brave (primary) -> DDG -> Serper.dev (last fallback).
    """
    # Brave/DDG use site:youtube.com prefix; Serper /videos doesn't need it
    yt_query = f"site:youtube.com {query}"

    # Try Brave first
    candidates = await _search_brave(yt_query, max_results)
    if candidates:
        logger.info(
            f"YouTube/Brave search '{query[:50]}' returned {len(candidates)} videos"
        )
        return candidates

    # Fallback to DDG
    logger.info("Brave returned no results, falling back to DDG")
    candidates = await _search_ddg(query, max_results)
    if candidates:
        logger.info(
            f"YouTube/DDG search '{query[:50]}' returned {len(candidates)} videos"
        )
        return candidates

    # Last fallback: Serper.dev video API (clean query, no site: prefix)
    logger.info("DDG returned no results, falling back to Serper")
    candidates = await _search_serper(query, max_results)  # uses original query without site:youtube.com
    logger.info(
        f"YouTube/Serper search '{query[:50]}' returned {len(candidates)} videos"
    )
    return candidates
