from pydantic import BaseModel
from typing import Optional, List


class YouTubeCandidate(BaseModel):
    """Raw search result before disambiguation."""
    video_id: str
    title: str
    description: str
    channel_title: str
    published_at: str


class ScoredVideo(BaseModel):
    """After disambiguation scoring."""
    video_id: str
    title: str
    description: str
    channel_title: str
    published_at: str
    match_score: float
    match_signals: List[str] = []
    is_person_match: bool = False
    transcript_text: Optional[str] = None
    transcript_available: bool = False
    url: str = ""


class PodcastCandidate(BaseModel):
    """Raw podcast episode from Serper search before disambiguation."""
    episode_id: str          # Slug from ListenNotes URL
    title: str
    description: str
    podcast_title: str       # Show name
    published_at: str
    audio_url: str = ""      # Direct audio file URL (filled after scrape)
    audio_length_sec: int = 0
    link: str                # ListenNotes episode page URL


class ScoredPodcast(BaseModel):
    """After disambiguation scoring."""
    episode_id: str
    title: str
    description: str
    podcast_title: str
    published_at: str
    audio_url: str = ""
    audio_length_sec: int = 0
    match_score: float
    match_signals: List[str] = []
    is_person_match: bool = False
    transcript_text: Optional[str] = None
    transcript_available: bool = False
    url: str = ""            # ListenNotes page URL


class ArticleContent(BaseModel):
    """Fetched article result."""
    url: str
    title: str
    text: str
    content_length_chars: int
    published_date: Optional[str] = None


class ArticleSearchEntry(BaseModel):
    """Log entry for each URL considered during article search."""
    query: str
    url: str
    status: str  # "accepted", "rejected_no_company_mention", "rejected_too_old", "fetch_failed", "skipped_domain"
    reason: str
    source: str = ""  # "brave", "duckduckgo", "google"


class CollectedArtifacts(BaseModel):
    """Everything gathered before synthesis."""
    podcasts: List[ScoredPodcast] = []
    videos: List[ScoredVideo] = []
    articles: List[ArticleContent] = []
    article_search_log: List[ArticleSearchEntry] = []
    steps_attempted: List[str] = []
    person_name: Optional[str] = None
    company_name: str
    person_title: Optional[str] = None
