import tempfile
import os
from pathlib import Path
from faster_whisper import WhisperModel

# Global model instance for efficiency (loads only once)
_model_instance = None

def get_whisper_model():
    global _model_instance
    if _model_instance is None:
        print("[WHISPER] Loading base model...")
        # Use 'base' for balance. 'cpu' for reliability, 'int8' for speed/memory efficiency
        _model_instance = WhisperModel("base", device="cpu", compute_type="int8")
    return _model_instance

def transcribe_audio(content: bytes, filename: str, return_timestamps: bool = True) -> dict:
    """
    Transcribes audio bytes into text using faster-whisper.
    
    Args:
        content: Audio file content as bytes
        filename: Original filename (used to determine file extension)
        return_timestamps: If True, returns transcription with timestamps
    
    Returns:
        Dictionary with:
        - text: Plain transcribed text as a string
        - segments: List of segments with timestamps (if return_timestamps=True)
        - language: Detected language
        - duration: Audio duration in seconds
    """
    suffix = Path(filename).suffix if filename else ".mp3"
    
    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        print(f"[WHISPER] Transcribing: {filename}")
        model = get_whisper_model()
        segments, info = model.transcribe(tmp_path, beam_size=5)
        
        text_content = ""
        segments_with_timestamps = []
        
        for segment in segments:
            text_content += segment.text + " "
            if return_timestamps:
                segments_with_timestamps.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip()
                })
            
        print(f"[WHISPER] Finished. Language: {info.language}, Duration: {info.duration:.2f}s")
        
        result = {
            "text": text_content.strip(),
            "language": info.language,
            "duration": round(info.duration, 2)
        }
        
        if return_timestamps:
            result["segments"] = segments_with_timestamps
        
        return result
        
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

