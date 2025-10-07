from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile, wave, json, subprocess
from vosk import Model, KaldiRecognizer
import numpy as np
import soundfile as sf
from app.utils import compute_acoustic_score, compute_phoneme_score

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/vosk-model-small-en-us-0.15"
model = Model(MODEL_PATH)

@app.post("/analyze/")
async def analyze(audio: UploadFile = File(...), text: str = Form(...)):
    # ----------------------
    # Save uploaded file temporarily
    # ----------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
        tmp_input.write(await audio.read())
        tmp_input_path = tmp_input.name

    # ----------------------
    # Convert to mono 16kHz WAV using ffmpeg
    # ----------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_fixed:
        tmp_fixed_path = tmp_fixed.name

    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_input_path, "-ac", "1", "-ar", "16000", tmp_fixed_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # ----------------------
    # Acoustic score
    # ----------------------
    y, sr = sf.read(tmp_fixed_path, dtype="float32")

    # ----------------------
    # Vosk recognition
    # ----------------------
    wf = wave.open(tmp_fixed_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    predicted_text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            predicted_text += res.get("text", "") + " "
    res = json.loads(rec.FinalResult())
    predicted_text += res.get("text", "")
    predicted_text = predicted_text.strip().lower()

    # ----------------------
    # Compute final score
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
