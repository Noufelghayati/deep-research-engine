import re
import asyncio
import datetime
import httpx
from bs4 import BeautifulSoup
from typing import Optional, List, Tuple
from models.internal import ArticleContent, ArticleSearchEntry
from config import settings
import logging
from urllib.parse import unquote, quote_plus

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

_SKIP_DOMAINS = [
    "google.com", "youtube.com", "facebook.com",
    "twitter.com", "x.com", "instagram.com",
    "reddit.com", "tiktok.com", "pinterest.com",
    "wikipedia.org",
    "duckduckgo.com", "brave.com",
]

# LinkedIn is conditionally allowed for person-specific searches
_LINKEDIN_DOMAIN = "linkedin.com"


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _article_mentions_company(text: str, company_name: str) -> bool:
    """Check if article text actually mentions the company (full name)."""
    normalized_text = _normalize(text[:5000])
    normalized_company = _normalize(company_name)

    if normalized_company in normalized_text:
        return True

    if normalized_company.replace(" ", "") in normalized_text.replace(" ", ""):
        return True

    return False


def _titles_overlap(title_a: str, title_b: str) -> bool:
    """Simple topic dedup: check if two article titles share significant words."""
    words_a = set(_normalize(title_a).split())
    words_b = set(_normalize(title_b).split())
    # Remove very common words
    stop = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "is", "are", "was", "with", "by", "from"}
    words_a -= stop
    words_b -= stop
    if not words_a or not words_b:
        return False
    overlap = words_a & words_b
    smaller = min(len(words_a), len(words_b))
    return smaller > 0 and len(overlap) / smaller >= 0.6


async def fetch_article(url: str) -> Optional[ArticleContent]:
    """Fetch and extract main text from a web article URL."""
    try:
        async with httpx.AsyncClient(
            timeout=20.0, follow_redirects=True, headers=_HEADERS
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(
            ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]
        ):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else ""

        article_tag = soup.find("article")
        if article_tag:
            text = article_tag.get_text(separator="\n", strip=True)
        else:
            main_tag = soup.find("main")
            if main_tag:
                text = main_tag.get_text(separator="\n", strip=True)
            else:
                paragraphs = soup.find_all("p")
                text = "\n".join(
                    p.get_text(strip=True)
                    for p in paragraphs
                    if len(p.get_text(strip=True)) > 40
                )

        if not text or len(text) < 200:
            logger.info(f"Article too short or empty: {url}")
            return None

        if len(text) > settings.max_transcript_chars:
            text = text[: settings.max_transcript_chars] + "... [truncated]"

        logger.info(f"Article fetched: {url} ({len(text)} chars)")
        return ArticleContent(
            url=url, title=title, text=text, content_length_chars=len(text)
        )

    except Exception as e:
        logger.warning(f"Failed to fetch article {url}: {e}")
        return None


def _is_blocked_domain(url: str, allow_linkedin: bool = False) -> Optional[str]:
    """Return the blocked domain name if URL matches, else None."""
    blocked = next((d for d in _SKIP_DOMAINS if d in url), None)
    if blocked:
        return blocked
    # LinkedIn is blocked by default, allowed for person-specific queries
    if _LINKEDIN_DOMAIN in url and not allow_linkedin:
        return _LINKEDIN_DOMAIN
    return None


def _is_article_too_old(url: str, title: str, max_age_months: int = 12) -> Optional[int]:
    """
    Check if article is older than max_age_months based on URL path or title.
    Returns the detected year if too old, else None.
    """
    today = datetime.date.today()
    cutoff_year = today.year - 1  # ~12 months

    # Check URL for year patterns like /2020/ or /2020-10/
    year_in_url = re.search(r"/(\d{4})/", url)
    if year_in_url:
        year = int(year_in_url.group(1))
        if 2000 <= year <= today.year and year < cutoff_year:
            return year

    # Check title for year mentions like "October 2020" or "| 2020"
    year_in_title = re.findall(r"\b(20\d{2})\b", title)
    if year_in_title:
        most_recent = max(int(y) for y in year_in_title)
        if most_recent < cutoff_year:
            return most_recent

    return None


async def _search_brave(
    query: str, search_log: List[ArticleSearchEntry],
    allow_linkedin: bool = False,
) -> List[str]:
    """Search Brave for article URLs (scraping fallback)."""
    search_url = f"https://search.brave.com/search?q={quote_plus(query)}"
    urls = []

    try:
        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True, headers=_HEADERS
        ) as client:
            resp = await client.get(search_url)

        if resp.status_code != 200:
            search_log.append(ArticleSearchEntry(
                query=query, url="",
                status="search_error",
                reason=f"Brave returned HTTP {resp.status_code}",
            ))
            return urls

        soup = BeautifulSoup(resp.text, "html.parser")

        for link in soup.select('a[href^="http"]'):
            href = link.get("href", "")
            if not href or "brave.com" in href or "search.brave" in href:
                continue

            blocked = _is_blocked_domain(href, allow_linkedin=allow_linkedin)
            if blocked:
                search_log.append(ArticleSearchEntry(
                    query=query, url=href,
                    status="skipped_domain",
                    reason=f"Domain '{blocked}' is in skip list",
                ))
                continue

            if href not in urls:
                urls.append(href)

    except Exception as e:
        logger.warning(f"Brave search failed for '{query}': {e}")
        search_log.append(ArticleSearchEntry(
            query=query, url="",
            status="search_error",
            reason=f"Brave search exception: {e}",
        ))

    return urls


async def _search_duckduckgo(
    query: str, search_log: List[ArticleSearchEntry],
    allow_linkedin: bool = False,
) -> List[str]:
    """Fallback: search DuckDuckGo HTML lite for URLs."""
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    urls = []

    try:
        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True, headers=_HEADERS
        ) as client:
            resp = await client.get(search_url)

        if resp.status_code != 200:
            search_log.append(ArticleSearchEntry(
                query=query, url="",
                status="search_error",
                reason=f"DDG returned HTTP {resp.status_code} (likely rate-limited)",
            ))
            return urls

        soup = BeautifulSoup(resp.text, "html.parser")

        for link in soup.select("a.result__a"):
            href = link.get("href", "")
            if not href:
                continue

            if "uddg=" in href:
                actual_url = unquote(href.split("uddg=")[1].split("&")[0])
            elif href.startswith("http"):
                actual_url = href
            else:
                continue

            blocked = _is_blocked_domain(actual_url, allow_linkedin=allow_linkedin)
            if blocked:
                search_log.append(ArticleSearchEntry(
                    query=query, url=actual_url,
                    status="skipped_domain",
                    reason=f"Domain '{blocked}' is in skip list",
                ))
                continue

            if actual_url not in urls:
                urls.append(actual_url)

    except Exception as e:
        logger.warning(f"DDG search failed for '{query}': {e}")
        search_log.append(ArticleSearchEntry(
            query=query, url="",
            status="search_error",
            reason=f"DDG search exception: {e}",
        ))

    return urls


async def _search_serper(
    query: str, search_log: List[ArticleSearchEntry],
    allow_linkedin: bool = False,
) -> List[str]:
    """Search using Serper.dev API (Google results)."""
    if not settings.serper_api_key:
        search_log.append(ArticleSearchEntry(
            query=query, url="",
            status="search_error",
            reason="Serper API not configured (missing API key)",
            source="serper",
        ))
        return []

    api_url = "https://google.serper.dev/search"
    payload = {
        "q": query,
        "tbs": "qdr:y",
    }
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    urls = []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(api_url, json=payload, headers=headers)

        if resp.status_code != 200:
            error_detail = ""
            try:
                err_data = resp.json()
                error_detail = err_data.get("message", "")
            except Exception:
                error_detail = resp.text[:200]
            logger.warning(f"Serper API HTTP {resp.status_code}: {error_detail}")
            search_log.append(ArticleSearchEntry(
                query=query, url="",
                status="search_error",
                reason=f"Serper API returned HTTP {resp.status_code}: {error_detail[:150]}",
                source="serper",
            ))
            return urls

        data = resp.json()
        for item in data.get("organic", []):
            href = item.get("link", "")
            if not href:
                continue

            blocked = _is_blocked_domain(href, allow_linkedin=allow_linkedin)
            if blocked:
                search_log.append(ArticleSearchEntry(
                    query=query, url=href,
                    status="skipped_domain",
                    reason=f"Domain '{blocked}' is in skip list",
                    source="serper",
                ))
                continue

            if href not in urls:
                urls.append(href)

    except Exception as e:
        logger.warning(f"Serper search failed for '{query}': {e}")
        search_log.append(ArticleSearchEntry(
            query=query, url="",
            status="search_error",
            reason=f"Serper search exception: {e}",
            source="serper",
        ))

    return urls


async def _web_search(
    query: str, search_log: List[ArticleSearchEntry],
    allow_linkedin: bool = False,
) -> Tuple[List[str], str]:
    """Search using Serper.dev (primary) -> Brave -> DDG (last fallback).
    Returns (urls, source_engine)."""
    urls = await _search_serper(query, search_log, allow_linkedin=allow_linkedin)
    if urls:
        return urls, "serper"

    logger.info(f"Serper returned 0 results for '{query[:40]}', trying Brave")

    urls = await _search_brave(query, search_log, allow_linkedin=allow_linkedin)
    if urls:
        return urls, "brave"

    logger.info(f"Brave returned 0 results for '{query[:40]}', trying DDG")

    urls = await _search_duckduckgo(query, search_log, allow_linkedin=allow_linkedin)
    return urls, "duckduckgo"


async def search_and_fetch_articles(
    company_name: str,
    person_name: Optional[str] = None,
    max_articles: int = 4,
) -> Tuple[List[ArticleContent], List[ArticleSearchEntry]]:
    """
    Search for relevant articles, validate they mention the company,
    and return up to max_articles with topic deduplication.
    Returns (articles, search_log).
    """
    search_log: List[ArticleSearchEntry] = []
    accepted: List[ArticleContent] = []
    seen_urls: set = set()

    current_year = datetime.date.today().year
    year_range = f"{current_year - 1} OR {current_year}"

    # Person-first query ordering: person queries first, company queries last
    # LinkedIn is allowed for person-specific queries only
    person_queries = []
    if person_name:
        person_queries = [
            f'{person_name} {company_name}',
            f'{person_name} {company_name} interview OR podcast OR panel',
            f'{person_name} {company_name} conference OR keynote OR quote',
        ]

    company_queries = [
        f'{company_name} partnership',
        f'{company_name} funding',
        f'{company_name} expansion',
        f'{company_name} {year_range}',
    ]

    # Build query list with metadata: (query, allow_linkedin)
    queries_with_meta = []
    for q in person_queries:
        queries_with_meta.append((q, True))   # LinkedIn allowed for person queries
    for q in company_queries:
        queries_with_meta.append((q, False))   # LinkedIn blocked for company queries

    max_urls_per_query = 4

    for i, (query, allow_linkedin) in enumerate(queries_with_meta):
        if len(accepted) >= max_articles:
            break

        if i > 0:
            await asyncio.sleep(1.5)

        candidate_urls, source = await _web_search(query, search_log, allow_linkedin=allow_linkedin)
        logger.info(
            f"Article search '{query[:50]}' returned {len(candidate_urls)} candidate URLs via {source}"
        )

        if not candidate_urls:
            search_log.append(ArticleSearchEntry(
                query=query, url="",
                status="no_results",
                reason="Search returned 0 candidate URLs for this query",
                source=source,
            ))
            continue

        tried = 0
        for url in candidate_urls:
            if len(accepted) >= max_articles:
                break
            if tried >= max_urls_per_query:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Pre-fetch date check
            old_year = _is_article_too_old(url, "")
            if old_year:
                search_log.append(ArticleSearchEntry(
                    query=query, url=url,
                    status="rejected_too_old",
                    reason=f"URL indicates article is from {old_year} (older than 12 months)",
                    source=source,
                ))
                continue

            article = await fetch_article(url)
            if not article:
                search_log.append(ArticleSearchEntry(
                    query=query, url=url,
                    status="fetch_failed",
                    reason="Could not fetch or extract text (too short, empty, or HTTP error)",
                    source=source,
                ))
                tried += 1
                continue

            # Post-fetch date check
            old_year = _is_article_too_old(url, article.title)
            if old_year:
                search_log.append(ArticleSearchEntry(
                    query=query, url=url,
                    status="rejected_too_old",
                    reason=f"Article '{article.title[:60]}' is from {old_year} (older than 12 months)",
                    source=source,
                ))
                tried += 1
                continue

            # Accept article if it mentions the company OR the person (for LinkedIn etc.)
            mentions_company = _article_mentions_company(article.text, company_name)
            mentions_person = (
                person_name and person_name.lower() in article.text.lower()
            )
            if not mentions_company and not mentions_person:
                search_log.append(ArticleSearchEntry(
                    query=query, url=url,
                    status="rejected_no_company_mention",
                    reason=f"Article text mentions neither '{company_name}' nor '{person_name or ''}'",
                    source=source,
                ))
                tried += 1
                continue

            # Topic dedup: skip if title overlaps significantly with an already-accepted article
            is_dup = False
            for existing in accepted:
                if _titles_overlap(article.title, existing.title):
                    search_log.append(ArticleSearchEntry(
                        query=query, url=url,
                        status="rejected_duplicate_topic",
                        reason=f"Topic overlaps with already-accepted '{existing.title[:50]}'",
                        source=source,
                    ))
                    is_dup = True
                    break
            if is_dup:
                tried += 1
                continue

            search_log.append(ArticleSearchEntry(
                query=query, url=url,
                status="accepted",
                reason=f"Article mentions '{company_name}' — {article.content_length_chars} chars extracted",
                source=source,
            ))
            logger.info(f"Article accepted ({len(accepted)+1}/{max_articles}): {url} (via {source})")
            accepted.append(article)

    logger.info(f"Article search complete: {len(accepted)} articles accepted for {company_name}")
    return accepted, search_log
