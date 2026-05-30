import whisper
import logging

logger = logging.getLogger(__name__)


def transcribe_video(video_path: str) -> list:
    """Transcribe video/audio using Whisper and return timestamped segments."""
    logger.info(f"Loading Whisper model for: {video_path}")
    model = whisper.load_model("base")
    # Keep word-level timing so downstream clip boundaries can align to sentence ends.
    result = model.transcribe(video_path, fp16=False, word_timestamps=True)
    segments = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            ws = w.get("start")
            we = w.get("end")
            if ws is None or we is None:
                continue
            words.append({
                "word": str(w.get("word", "")).strip(),
                "start": float(ws),
                "end": float(we),
            })

        segments.append({
            "text": seg["text"].strip(),
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "words": words,
        })
    logger.info(f"Transcription complete: {len(segments)} segments")
    return segments
