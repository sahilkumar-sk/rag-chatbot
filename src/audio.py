# import os
# import tempfile
# import subprocess
# from config import WHISPER_MODEL


# def transcribe_audio(audio_bytes: bytes, groq_client) -> str:
#     """Transcribe audio using Groq Whisper."""
#     if len(audio_bytes) < 100:
#         return ""

#     tmp_input = None
#     tmp_output = None

#     try:
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#             tmp.write(audio_bytes)
#             tmp_input = tmp.name

#         tmp_output = tmp_input.replace(".wav", "_boosted.wav")
#         boost_success = False

#         try:
#             result = subprocess.run([
#                 "ffmpeg", "-y", "-i", tmp_input,
#                 "-af", "volume=3.0",
#                 "-ar", "16000",
#                 "-ac", "1",
#                 tmp_output
#             ], capture_output=True, timeout=10)
#             boost_success = result.returncode == 0
#         except Exception:
#             pass  # ffmpeg not available, use original

#         audio_to_send = tmp_output if boost_success else tmp_input

#         with open(audio_to_send, "rb") as audio_file:
#             transcription = groq_client.audio.transcriptions.create(
#                 model=WHISPER_MODEL,
#                 file=audio_file,
#                 response_format="text",
#                 language="en",
#                 prompt="The user is asking a question about a research document."
#             )

#         return str(transcription).strip()

#     except Exception as e:
#         print(f"❌ Transcription error: {e}")
#         return ""
#     finally:
#         for path in [tmp_input, tmp_output]:
#             if path and os.path.exists(path):
#                 os.unlink(path)