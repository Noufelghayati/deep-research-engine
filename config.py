from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    youtube_api_key: str = "PLACEHOLDER_YT_KEY"
    gemini_api_key: str = "PLACEHOLDER_GEMINI_KEY"
    serper_api_key: str = ""
    webshare_proxy_url: str = ""  # e.g. http://user:pass@p.webshare.io:80
    ffmpeg_location: str = ""  # path to ffmpeg dir, e.g. C:/xampp/htdocs/Josh/marouane/ffmpeg
    cors_origins: List[str] = [
        "chrome-extension://*",
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    # Decision tree thresholds
    disambiguation_threshold: float = 0.3
    weak_result_threshold: float = 0.5
    max_youtube_artifacts: int = 4
    max_article_artifacts: int = 4
    max_transcript_chars: int = 30000
    max_transcription_duration_sec: int = 1500  # 25 min cap per video
    pipeline_timeout_sec: int = 100  # generous budget; articles always run even after timeout
    youtube_search_max_results: int = 5
    max_podcast_artifacts: int = 2  # keep low; podcast transcription is slow
    max_podcast_audio_duration_sec: int = 1800  # 30 min cap per episode

    # Gemini settings (kept for reference)
    gemini_model: str = "gemini-3-flash-preview"
    gemini_max_output_tokens: int = 65536
    gemini_temperature: float = 0.3

    # Claude settings
    anthropic_api_key: str = "PLACEHOLDER_ANTHROPIC_KEY"
    claude_model: str = "claude-sonnet-4-6"
    claude_max_output_tokens: int = 64000
    claude_temperature: float = 0.3

    # Auth
    jwt_secret_key: str = "CHANGE_ME_IN_PRODUCTION"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
