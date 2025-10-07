from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer
import io, wave, json, subprocess
import numpy as np
import soundfile as sf
from app.utils import compute_acoustic_score, compute_phoneme_score

app = FastAPI()

# -------------------------
# CORS Middleware
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load Vosk model once at startup
# -------------------------
MODEL_PATH = "models/vosk-model-small-en-us-0.15"
model = Model(MODEL_PATH)

@app.post("/analyze/")
async def analyze(audio: UploadFile = File(...), text: str = Form(...)):
    # ----------------------
    # Read audio into memory
    # ----------------------
    raw_audio = await audio.read()
    in_bytes = io.BytesIO(raw_audio)
    
    # ----------------------
    # Convert to mono 16kHz WAV in memory
    # ----------------------
    out_bytes = io.BytesIO()
    subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
        input=in_bytes.read(),
        stdout=out_bytes,
        stderr=subprocess.DEVNULL
    )
    out_bytes.seek(0)

    # ----------------------
    # Load for scoring
    # ----------------------
    y, sr = sf.read(out_bytes, dtype="float32")

    # ----------------------
    # Load for Vosk recognition
    # ----------------------
    out_bytes.seek(0)
    wf = wave.open(out_bytes, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    
    # Use larger chunks to speed up recognition
    while data := wf.readframes(16000):  # 1 second chunks
        rec.AcceptWaveform(data)

    predicted_text = json.loads(rec.FinalResult()).get("text", "").strip().lower()

    # ----------------------
    # Compute scores
    # ----------------------
    acoustic_score = compute_acoustic_score(y, sr)
    phoneme_score = compute_phoneme_score(text.lower(), predicted_text)
    final_score = 0.6 * phoneme_score + 0.4 * acoustic_score

    return {
        "predicted_text": predicted_text,
        "phoneme_score": round(float(phoneme_score), 2),
        "acoustic_score": round(float(acoustic_score), 2),
        "final_score": round(float(final_score), 2),
    }
