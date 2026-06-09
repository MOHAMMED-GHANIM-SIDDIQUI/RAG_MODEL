import io

from gtts import gTTS


def text_to_audio(response: str) -> io.BytesIO:
    audio_bytes = io.BytesIO()
    tts = gTTS(text=response, lang="en")
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes
