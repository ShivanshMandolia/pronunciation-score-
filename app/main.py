from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import torch, tempfile, librosa
from app.utils import extract_features, compute_acoustic_score, compute_phoneme_score

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load small Whisper model
device = 0 if torch.cuda.is_available() else -1
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)

@app.post("/analyze/")
async def analyze(audio: UploadFile = File(...), text: str = Form(...)):
    # Save uploaded file temporarily
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(await audio.read())
    tmp.close()

    # Transcribe
    result = asr_pipeline(tmp.name)
    predicted_text = result["text"].lower()

    # Load for scoring
    y, sr = librosa.load(tmp.name, sr=16000)
    acoustic_score = compute_acoustic_score(y, sr)
    phoneme_score = compute_phoneme_score(text, predicted_text)
    final_score = (0.6 * phoneme_score) + (0.4 * acoustic_score)

    return {
    "predicted_text": predicted_text,
    "phoneme_score": round(float(phoneme_score), 2),
    "acoustic_score": round(float(acoustic_score), 2),
    "final_score": round(float(final_score), 2),
    }

