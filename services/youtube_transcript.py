import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from config import settings
from utils.text import clean_transcript_text

logger = logging.getLogger(__name__)

# Add ffmpeg to PATH so yt-dlp can find it (for partial downloads / audio cuts)
if settings.ffmpeg_location and os.path.isdir(settings.ffmpeg_location):
    os.environ["PATH"] += os.pathsep + settings.ffmpeg_location
    logger.info(f"Added ffmpeg to PATH: {settings.ffmpeg_location}")

# Circuit breaker: if YouTube blocks transcripts (429), skip further attempts
_transcript_blocked = False

# Temp directory for audio downloads
_TEMP_DIR = Path(tempfile.gettempdir()) / "josh_audio"


def _build_proxy_config():
    """Build Webshare proxy config if credentials are set."""
    if not settings.webshare_proxy_url:
        return None

    # Parse http://user:pass@host:port
    url = settings.webshare_proxy_url
    # Strip scheme
    if "://" in url:
        url = url.split("://", 1)[1]

    # Split user:pass@host:port
    if "@" not in url:
        return None

    auth, _ = url.rsplit("@", 1)
    if ":" not in auth:
        return None

    username, password = auth.split(":", 1)

    return WebshareProxyConfig(
        proxy_username=username,
        proxy_password=password,
    )


# Build proxy config once at module load
_proxy_config = _build_proxy_config()

# Create API instance with or without proxy
if _proxy_config:
    _ytt_api = YouTubeTranscriptApi(proxy_config=_proxy_config)
    logger.info("YouTube transcript API configured with Webshare proxy")
else:
    _ytt_api = YouTubeTranscriptApi()
    logger.info("YouTube transcript API configured without proxy")


# ---------------------------------------------------------------------------
# Primary: youtube-transcript-api (existing YouTube transcripts)
# ---------------------------------------------------------------------------

def _fetch_existing_transcript_sync(video_id: str) -> Tuple[Optional[str], bool]:
    """Try to fetch an existing YouTube transcript (auto-generated or manual)."""
    global _transcript_blocked

    try:
        transcript = _ytt_api.fetch(video_id, languages=["en"])
        snippets = transcript.snippets
        if not snippets:
            return None, False

        full_text = " ".join(s.text for s in snippets)
        full_text = clean_transcript_text(full_text)

        if len(full_text) > settings.max_transcript_chars:
            full_text = full_text[: settings.max_transcript_chars] + "... [truncated]"

        logger.info(f"Transcript fetched for {video_id}: {len(full_text)} chars (existing)")
        return full_text, True

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "Too Many Requests" in error_str:
            _transcript_blocked = True
            logger.warning(f"YouTube blocking transcripts (429) for {video_id}")
        else:
            logger.info(f"No existing transcript for {video_id}: {e}")
        return None, False


# ---------------------------------------------------------------------------
# Fallback: yt-dlp download + Gemini audio transcription
# ---------------------------------------------------------------------------

def _trim_audio(audio_path: Path, max_sec: int) -> Path:
    """Trim audio to first max_sec seconds using ffmpeg. Returns trimmed path."""
    import subprocess

    trimmed = audio_path.with_name(f"{audio_path.stem}_trimmed{audio_path.suffix}")
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-t", str(max_sec), "-c", "copy", str(trimmed),
    ]
    logger.info(f"Trimming audio to first {max_sec}s: {audio_path.name}")
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode == 0 and trimmed.exists():
        # Replace original with trimmed version
        audio_path.unlink()
        trimmed.rename(audio_path)
        logger.info(f"Trimmed audio: {audio_path.name} ({audio_path.stat().st_size // 1024}KB)")
        return audio_path
    else:
        logger.warning(f"ffmpeg trim failed: {result.stderr.decode(errors='ignore')[:200]}")
        # If trim fails, just use the original file
        if trimmed.exists():
            trimmed.unlink()
        return audio_path


def _download_audio_sync(video_id: str) -> Optional[Path]:
    """Download YouTube audio using yt-dlp. Returns path to audio file."""
    try:
        import yt_dlp
    except ImportError:
        logger.warning("yt-dlp not installed, skipping audio transcription fallback")
        return None

    _TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up any previously cached audio for this video
    for ext in ["m4a", "webm", "opus", "mp3", "wav", "ogg"]:
        old = _TEMP_DIR / f"{video_id}.{ext}"
        if old.exists():
            old.unlink()
            logger.debug(f"Removed stale audio cache: {old.name}")

    output_template = str(_TEMP_DIR / f"{video_id}.%(ext)s")

    url = f"https://www.youtube.com/watch?v={video_id}"
    max_sec = settings.max_transcription_duration_sec

    # Download full audio, then trim afterward (download_ranges is unreliable for audio)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": False,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 5,
        "fragment_retries": 5,
        "file_access_retries": 3,
        "http_chunk_size": 1048576,  # 1MB chunks for more reliable downloads
    }
    if settings.webshare_proxy_url:
        ydl_opts["proxy"] = settings.webshare_proxy_url

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get("duration", 0) or 0
            logger.info(f"Audio downloaded for {video_id}: {duration}s total")

        # Find the downloaded file
        audio_path = None
        for ext in ["m4a", "webm", "opus", "mp3", "wav", "ogg"]:
            candidate = _TEMP_DIR / f"{video_id}.{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if not audio_path:
            logger.warning(f"Download completed but no audio file found for {video_id}")
            return None

        size_kb = audio_path.stat().st_size // 1024
        logger.info(f"Audio file: {audio_path.name} ({size_kb}KB)")

        # Trim to first N seconds if video is longer than cap
        if duration > max_sec:
            audio_path = _trim_audio(audio_path, max_sec)

        return audio_path

    except Exception as e:
        logger.warning(f"Audio download failed for {video_id}: {e}")
        return None


def _transcribe_with_gemini_sync(audio_path: Path) -> Optional[str]:
    """Send audio file to Gemini for transcription."""
    from google import genai
    from google.genai import types

    try:
        client = genai.Client(api_key=settings.gemini_api_key)

        # Upload the audio file
        logger.info(f"Uploading audio to Gemini: {audio_path.name}")
        uploaded_file = client.files.upload(file=audio_path)

        # Transcribe
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=[
                uploaded_file,
                "Transcribe this audio into English text. "
                "Return ONLY the raw transcript — no timestamps, "
                "no speaker labels, no formatting, no commentary.",
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=65536,
            ),
        )

        text = response.text.strip() if response.text else None

        # Cleanup uploaded file from Gemini
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

        if text and len(text) > 50:
            logger.info(f"Gemini transcription complete: {len(text)} chars")
            return text
        else:
            logger.warning("Gemini transcription returned too short or empty result")
            return None

    except Exception as e:
        logger.error(f"Gemini transcription failed: {e}")
        return None


def _fallback_transcribe_sync(video_id: str) -> Optional[str]:
    """Fallback pipeline: download audio with yt-dlp, transcribe with Gemini."""
    audio_path = _download_audio_sync(video_id)
    if not audio_path:
        return None

    try:
        return _transcribe_with_gemini_sync(audio_path)
    finally:
        # Always cleanup the temp audio file
        try:
            if audio_path and audio_path.exists():
                audio_path.unlink()
                logger.debug(f"Cleaned up temp audio: {audio_path}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API: existing transcript first, then audio fallback
# ---------------------------------------------------------------------------

def _fetch_transcript_sync(video_id: str) -> Tuple[Optional[str], bool]:
    """
    Fetch transcript for a video.
    1. Try existing YouTube transcript (fast, free)
    2. If unavailable, download audio + transcribe with Gemini (slower, small cost)
    """
    # Step 1: Try existing YouTube transcript
    text, available = _fetch_existing_transcript_sync(video_id)
    if available:
        return text, True

    # Step 2: Fallback — download audio + Gemini transcription
    logger.info(f"No existing transcript for {video_id}, trying audio fallback")
    text = _fallback_transcribe_sync(video_id)
    if text:
        text = clean_transcript_text(text)
        if len(text) > settings.max_transcript_chars:
            text = text[: settings.max_transcript_chars] + "... [truncated]"
        logger.info(f"Transcript for {video_id}: {len(text)} chars (via Gemini audio)")
        return text, True

    return None, False


def _fetch_transcript_with_timestamps_sync(video_id: str) -> Optional[List[Dict]]:
    """Synchronous timestamped transcript fetch (existing transcripts only)."""
    try:
        transcript = _ytt_api.fetch(video_id, languages=["en"])
        snippets = transcript.snippets
        if not snippets:
            return None

        segments = []
        for s in snippets:
            segments.append({
                "text": s.text,
                "start": s.start,
                "duration": s.duration,
            })

        return segments if segments else None

    except Exception as e:
        logger.info(f"No timestamped transcript for {video_id}: {e}")
        return None


async def fetch_transcript(video_id: str) -> Tuple[Optional[str], bool]:
    """
    Fetch English transcript for a video.
    1. Existing YouTube transcript (youtube-transcript-api)
    2. Audio download + Gemini transcription (yt-dlp + Gemini)
    Returns (transcript_text, is_available).
    """
    # Circuit breaker only blocks existing transcript fetch;
    # audio fallback uses a different path so still worth trying
    return await asyncio.to_thread(_fetch_transcript_sync, video_id)


async def fetch_transcript_with_timestamps(
    video_id: str,
) -> Optional[List[Dict]]:
    """
    Returns list of {text, start, duration} dicts for timestamp references.
    Only available for videos with existing YouTube transcripts.
    """
    if _transcript_blocked and not settings.webshare_proxy_url:
        return None

    return await asyncio.to_thread(_fetch_transcript_with_timestamps_sync, video_id)


async def fallback_transcribe(video_id: str) -> Tuple[Optional[str], bool]:
    """
    Force audio fallback only: yt-dlp download + Gemini transcription.
    Skips youtube-transcript-api entirely. For testing purposes.
    """
    text = await asyncio.to_thread(_fallback_transcribe_sync, video_id)
    if text:
        text = clean_transcript_text(text)
        if len(text) > settings.max_transcript_chars:
            text = text[: settings.max_transcript_chars] + "... [truncated]"
        return text, True
    return None, False


# ---------------------------------------------------------------------------
# Progress-aware fallback (for test UI with real-time progress)
# ---------------------------------------------------------------------------

def fallback_transcribe_with_progress_sync(
    video_id: str,
    on_progress,  # callable(stage: str, percent: int, detail: str)
) -> Optional[str]:
    """Fallback pipeline with real-time progress reporting."""
    try:
        import yt_dlp
    except ImportError:
        on_progress("error", 0, "yt-dlp not installed")
        return None

    # --- Phase 1: Download audio (0–50%) ---
    on_progress("download", 0, "Starting audio download...")

    _TEMP_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["m4a", "webm", "opus", "mp3", "wav", "ogg"]:
        old = _TEMP_DIR / f"{video_id}.{ext}"
        if old.exists():
            old.unlink()

    output_template = str(_TEMP_DIR / f"{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"
    max_sec = settings.max_transcription_duration_sec

    def progress_hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                dl_pct = int((downloaded / total) * 50)
                dl_mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                on_progress("download", dl_pct, f"Downloading audio... {dl_mb:.1f}MB / {total_mb:.1f}MB")
        elif d["status"] == "finished":
            on_progress("download", 50, "Download complete")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": False,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 5,
        "fragment_retries": 5,
        "file_access_retries": 3,
        "http_chunk_size": 1048576,
        "progress_hooks": [progress_hook],
    }
    if settings.webshare_proxy_url:
        ydl_opts["proxy"] = settings.webshare_proxy_url

    audio_path = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get("duration", 0) or 0

        for ext in ["m4a", "webm", "opus", "mp3", "wav", "ogg"]:
            candidate = _TEMP_DIR / f"{video_id}.{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if not audio_path:
            on_progress("error", 0, "Download completed but no audio file found")
            return None

        # Trim if needed (50–55%)
        if duration > max_sec:
            on_progress("trim", 52, f"Trimming to first {max_sec // 60} minutes...")
            audio_path = _trim_audio(audio_path, max_sec)
            on_progress("trim", 55, "Trim complete")

    except Exception as e:
        on_progress("error", 0, f"Download failed: {e}")
        return None

    # --- Phase 2: Upload to Gemini (55–70%) ---
    try:
        from google import genai
        from google.genai import types

        on_progress("upload", 58, "Uploading audio to Gemini...")
        client = genai.Client(api_key=settings.gemini_api_key)
        uploaded_file = client.files.upload(file=audio_path)
        on_progress("upload", 70, "Upload complete")

        # --- Phase 3: Transcribe (70–95%) ---
        on_progress("transcribe", 72, "Transcribing with Gemini...")

        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=[
                uploaded_file,
                "Transcribe this audio into English text. "
                "Return ONLY the raw transcript — no timestamps, "
                "no speaker labels, no formatting, no commentary.",
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=65536,
            ),
        )

        text = response.text.strip() if response.text else None

        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

        if text and len(text) > 50:
            on_progress("done", 100, f"Complete — {len(text):,} chars transcribed")
            return text
        else:
            on_progress("error", 0, "Transcription returned empty result")
            return None

    except Exception as e:
        on_progress("error", 0, f"Transcription failed: {e}")
        return None
    finally:
        try:
            if audio_path and audio_path.exists():
                audio_path.unlink()
        except Exception:
            pass
