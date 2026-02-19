import asyncio
import os
import tempfile
import queue as queue_mod
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


def _noop_log(msg):
    pass


# ---------------------------------------------------------------------------
# Primary: youtube-transcript-api (existing YouTube transcripts)
# ---------------------------------------------------------------------------

def _fetch_existing_transcript_sync(video_id: str, on_log=None) -> Tuple[Optional[str], bool]:
    """Try to fetch an existing YouTube transcript (auto-generated or manual)."""
    global _transcript_blocked
    log = on_log or _noop_log

    try:
        # Try English first
        log("Trying YouTube captions (English)...")
        try:
            transcript = _ytt_api.fetch(video_id, languages=["en"])
            log("English captions found")
        except Exception:
            # No English transcript — try any available language
            transcript = None
            log("No English captions, checking other languages...")
            try:
                transcript_list = list(_ytt_api.list(video_id))
                if transcript_list:
                    t = transcript_list[0]
                    lang_info = f"{t.language} ({t.language_code})"
                    generated = " [auto-generated]" if t.is_generated else ""
                    log(f"Found: {lang_info}{generated}")
                    logger.info(f"No English transcript for {video_id}, using {lang_info}")
                    # Try translating to English if possible
                    if t.is_translatable:
                        log(f"Translating {t.language_code} \u2192 English...")
                        try:
                            transcript = t.translate("en").fetch()
                            log(f"Translation successful")
                            logger.info(f"Translated {t.language_code} -> en for {video_id}")
                        except Exception as te:
                            log(f"Translation failed: {str(te)[:80]}, using original {t.language_code}")
                            logger.info(f"Translation failed for {video_id}: {te}, using original language")
                            transcript = t.fetch()
                    else:
                        log(f"Not translatable, using original {t.language_code}")
                        transcript = t.fetch()
                else:
                    log("No captions available for this video")
            except Exception as le:
                log(f"Could not list transcripts: {str(le)[:80]}")
                logger.info(f"Could not list transcripts for {video_id}: {le}")
                return None, False

        if not transcript:
            return None, False

        snippets = transcript.snippets
        if not snippets:
            log("Captions returned empty snippets")
            return None, False

        max_sec = settings.max_transcription_duration_sec
        total_snippets = len(snippets)
        snippets = [s for s in snippets if s.start < max_sec]
        full_text = " ".join(s.text for s in snippets)
        full_text = clean_transcript_text(full_text)

        if len(full_text) > settings.max_transcript_chars:
            full_text = full_text[: settings.max_transcript_chars] + "... [truncated]"

        trimmed_note = f", capped at {max_sec // 60}min ({len(snippets)}/{total_snippets} snippets)" if len(snippets) < total_snippets else ""
        log(f"Captions fetched: {len(full_text):,} chars{trimmed_note}")
        logger.info(f"Transcript fetched for {video_id}: {len(full_text)} chars (existing)")
        return full_text, True

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "Too Many Requests" in error_str:
            _transcript_blocked = True
            log("YouTube is rate-limiting (429), captions blocked")
            logger.warning(f"YouTube blocking transcripts (429) for {video_id}")
        else:
            log(f"Captions failed: {error_str[:80]}")
            logger.info(f"No existing transcript for {video_id}: {e}")
        return None, False


# ---------------------------------------------------------------------------
# Fallback: yt-dlp download + Gemini audio transcription
# ---------------------------------------------------------------------------

def _trim_audio(audio_path: Path, max_sec: int, on_log=None) -> Path:
    """Trim audio to first max_sec seconds using ffmpeg. Returns trimmed path."""
    import subprocess
    log = on_log or _noop_log

    trimmed = audio_path.with_name(f"{audio_path.stem}_trimmed{audio_path.suffix}")
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-t", str(max_sec), "-c", "copy", str(trimmed),
    ]
    log(f"Trimming audio to first {max_sec // 60} minutes...")
    logger.info(f"Trimming audio to first {max_sec}s: {audio_path.name}")
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode == 0 and trimmed.exists():
        # Replace original with trimmed version
        audio_path.unlink()
        trimmed.rename(audio_path)
        size_kb = audio_path.stat().st_size // 1024
        log(f"Trimmed: {size_kb:,}KB")
        logger.info(f"Trimmed audio: {audio_path.name} ({size_kb}KB)")
        return audio_path
    else:
        log("Trim failed, using full audio")
        logger.warning(f"ffmpeg trim failed: {result.stderr.decode(errors='ignore')[:200]}")
        # If trim fails, just use the original file
        if trimmed.exists():
            trimmed.unlink()
        return audio_path


def _download_audio_sync(video_id: str, on_log=None) -> Optional[Path]:
    """Download YouTube audio using yt-dlp. Returns path to audio file."""
    log = on_log or _noop_log
    try:
        import yt_dlp
    except ImportError:
        log("yt-dlp not installed, cannot download audio")
        logger.warning("yt-dlp not installed, skipping audio transcription fallback")
        return None

    _TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up any previously cached audio for this video
    for ext in ["m4a", "webm", "opus", "mp3", "wav", "ogg", "mp4"]:
        old = _TEMP_DIR / f"{video_id}.{ext}"
        if old.exists():
            old.unlink()
            logger.debug(f"Removed stale audio cache: {old.name}")

    output_template = str(_TEMP_DIR / f"{video_id}.%(ext)s")

    url = f"https://www.youtube.com/watch?v={video_id}"
    max_sec = settings.max_transcription_duration_sec

    log("Downloading audio with yt-dlp...")

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
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }
    if settings.webshare_proxy_url:
        ydl_opts["proxy"] = settings.webshare_proxy_url

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get("duration", 0) or 0
            log(f"Audio downloaded: {duration}s duration")
            logger.info(f"Audio downloaded for {video_id}: {duration}s total")

        # Find the downloaded file
        audio_path = None
        for ext in ["m4a", "webm", "opus", "mp3", "wav", "ogg", "mp4"]:
            candidate = _TEMP_DIR / f"{video_id}.{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if not audio_path:
            log("Download completed but no audio file found")
            logger.warning(f"Download completed but no audio file found for {video_id}")
            return None

        size_kb = audio_path.stat().st_size // 1024
        log(f"Audio file: {audio_path.suffix} ({size_kb:,}KB)")
        logger.info(f"Audio file: {audio_path.name} ({size_kb}KB)")

        # Trim to first N seconds if video is longer than cap
        if duration > max_sec:
            audio_path = _trim_audio(audio_path, max_sec, on_log=log)

        return audio_path

    except Exception as e:
        log(f"Download failed: {str(e)[:100]}")
        logger.warning(f"Audio download failed for {video_id}: {e}")
        return None


def _transcribe_with_gemini_sync(audio_path: Path, on_log=None) -> Optional[str]:
    """Send audio file to Gemini for transcription."""
    from google import genai
    from google.genai import types
    log = on_log or _noop_log

    try:
        client = genai.Client(api_key=settings.gemini_api_key)

        # Upload the audio file
        log("Uploading audio to Gemini...")
        logger.info(f"Uploading audio to Gemini: {audio_path.name}")
        uploaded_file = client.files.upload(file=audio_path)
        log("Upload complete, waiting for processing...")

        # Wait for file to become ACTIVE (processing can take time for large files)
        import time as _time
        for i in range(30):  # up to 60 seconds
            status = client.files.get(name=uploaded_file.name)
            if status.state.name == "ACTIVE":
                log("Gemini file ready")
                break
            log(f"Gemini processing... ({i * 2}s)")
            logger.info(f"Waiting for Gemini file processing... state={status.state.name}")
            _time.sleep(2)

        # Transcribe (with retry on 503)
        log("Transcribing audio with Gemini...")
        response = None
        for attempt in range(3):
            try:
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
                break  # success
            except Exception as retry_err:
                err_str = str(retry_err)
                if ("503" in err_str or "UNAVAILABLE" in err_str) and attempt < 2:
                    wait = (attempt + 1) * 3  # 3s, 6s
                    log(f"Gemini 503 — retrying in {wait}s (attempt {attempt + 2}/3)...")
                    logger.warning(f"Gemini transcription 503, retry {attempt + 1}/2 in {wait}s")
                    _time.sleep(wait)
                else:
                    raise  # non-503 or final attempt — let outer handler catch it
        if response is None:
            log("Gemini transcription failed after retries")
            return None

        text = response.text.strip() if response.text else None

        # Cleanup uploaded file from Gemini
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

        if text and len(text) > 50:
            log(f"Gemini transcription complete: {len(text):,} chars")
            logger.info(f"Gemini transcription complete: {len(text)} chars")
            return text
        else:
            log("Gemini returned empty/too short result")
            logger.warning("Gemini transcription returned too short or empty result")
            return None

    except Exception as e:
        log(f"Gemini transcription failed: {str(e)[:100]}")
        logger.error(f"Gemini transcription failed: {e}")
        return None


def _fallback_transcribe_sync(video_id: str, on_log=None) -> Optional[str]:
    """Fallback pipeline: download audio with yt-dlp, transcribe with Gemini."""
    log = on_log or _noop_log
    audio_path = _download_audio_sync(video_id, on_log=log)
    if not audio_path:
        return None

    try:
        return _transcribe_with_gemini_sync(audio_path, on_log=log)
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

def _fetch_transcript_sync(video_id: str, on_log=None) -> Tuple[Optional[str], bool]:
    """
    Fetch transcript for a video.
    1. Try existing YouTube transcript (fast, free)
    2. If unavailable, download audio + transcribe with Gemini (slower, small cost)
    """
    log = on_log or _noop_log

    # Step 1: Try existing YouTube transcript
    text, available = _fetch_existing_transcript_sync(video_id, on_log=log)
    if available:
        return text, True

    # Step 2: Fallback — download audio + Gemini transcription
    log("No captions available \u2014 falling back to audio download + Gemini transcription")
    logger.info(f"No existing transcript for {video_id}, trying audio fallback")
    text = _fallback_transcribe_sync(video_id, on_log=log)
    if text:
        text = clean_transcript_text(text)
        if len(text) > settings.max_transcript_chars:
            text = text[: settings.max_transcript_chars] + "... [truncated]"
        log(f"Audio fallback complete: {len(text):,} chars")
        logger.info(f"Transcript for {video_id}: {len(text)} chars (via Gemini audio)")
        return text, True

    log("All transcript methods failed")
    return None, False


def _fetch_transcript_with_timestamps_sync(video_id: str) -> Optional[List[Dict]]:
    """Synchronous timestamped transcript fetch (existing transcripts only)."""
    try:
        transcript = _ytt_api.fetch(video_id, languages=["en"])
        snippets = transcript.snippets
        if not snippets:
            return None

        max_sec = settings.max_transcription_duration_sec
        segments = []
        for s in snippets:
            if s.start >= max_sec:
                break
            segments.append({
                "text": s.text,
                "start": s.start,
                "duration": s.duration,
            })

        return segments if segments else None

    except Exception as e:
        logger.info(f"No timestamped transcript for {video_id}: {e}")
        return None


async def fetch_transcript(video_id: str, on_log=None) -> Tuple[Optional[str], bool]:
    """
    Fetch English transcript for a video.
    1. Existing YouTube transcript (youtube-transcript-api)
    2. Audio download + Gemini transcription (yt-dlp + Gemini)
    Returns (transcript_text, is_available).
    on_log: optional async callable(message: str) for real-time progress.
    Timeout: 5 minutes max to prevent blocking the pipeline.
    """
    if on_log:
        # Bridge sync on_log calls from the thread to async on_log
        log_q = queue_mod.Queue()

        def sync_log(msg):
            log_q.put(msg)

        async def run_with_log_bridge():
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, _fetch_transcript_sync, video_id, sync_log)

            while not future.done():
                # Drain log messages
                while True:
                    try:
                        msg = log_q.get_nowait()
                        await on_log(msg)
                    except queue_mod.Empty:
                        break
                await asyncio.sleep(0.3)

            # Drain remaining log messages
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
            await on_log("Transcript fetch timed out after 5 minutes")
            logger.warning(f"Transcript fetch timed out after 5 min for {video_id}")
            return None, False
    else:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_fetch_transcript_sync, video_id),
                timeout=300,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Transcript fetch timed out after 5 min for {video_id}")
            return None, False


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

    # --- Phase 1: Download audio (0-50%) ---
    on_progress("download", 0, "Starting audio download...")

    _TEMP_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["m4a", "webm", "opus", "mp3", "wav", "ogg", "mp4"]:
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
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }
    if settings.webshare_proxy_url:
        ydl_opts["proxy"] = settings.webshare_proxy_url

    audio_path = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get("duration", 0) or 0

        for ext in ["m4a", "webm", "opus", "mp3", "wav", "ogg", "mp4"]:
            candidate = _TEMP_DIR / f"{video_id}.{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if not audio_path:
            on_progress("error", 0, "Download completed but no audio file found")
            return None

        # Trim if needed (50-55%)
        if duration > max_sec:
            on_progress("trim", 52, f"Trimming to first {max_sec // 60} minutes...")
            audio_path = _trim_audio(audio_path, max_sec)
            on_progress("trim", 55, "Trim complete")

    except Exception as e:
        on_progress("error", 0, f"Download failed: {e}")
        return None

    # --- Phase 2: Upload to Gemini (55-70%) ---
    try:
        from google import genai
        from google.genai import types

        on_progress("upload", 58, "Uploading audio to Gemini...")
        client = genai.Client(api_key=settings.gemini_api_key)
        uploaded_file = client.files.upload(file=audio_path)
        on_progress("upload", 68, "Upload complete, processing...")

        # Wait for file to become ACTIVE
        import time as _time
        for i in range(30):
            status = client.files.get(name=uploaded_file.name)
            if status.state.name == "ACTIVE":
                break
            on_progress("upload", 68 + min(i, 5), f"Gemini processing file... ({i*2}s)")
            _time.sleep(2)
        on_progress("upload", 72, "File ready")

        # --- Phase 3: Transcribe (72-95%) ---
        on_progress("transcribe", 74, "Transcribing with Gemini...")

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
