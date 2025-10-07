from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import io, wave, json, numpy as np
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
    # Load audio into memory
    # ----------------------
    raw_audio = await audio.read()
    audio_segment = AudioSegment.from_file(io.BytesIO(raw_audio))
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)

    # ----------------------
    # Feed Vosk in larger chunks
    # ----------------------
    wav_bytes = io.BytesIO()
    audio_segment.export(wav_bytes, format="wav")
    wav_bytes.seek(0)

    wf = wave.open(wav_bytes, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    while data := wf.readframes(16000):  # 1 second chunks
        rec.AcceptWaveform(data)
    predicted_text = json.loads(rec.FinalResult()).get("text", "").strip().lower()

    # ----------------------
    # Acoustic score
    # ----------------------
    y = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    sr = 16000
    acoustic_score = compute_acoustic_score(y, sr)
    phoneme_score = compute_phoneme_score(text.lower(), predicted_text)
    final_score = 0.6 * phoneme_score + 0.4 * acoustic_score

    return {
        "predicted_text": predicted_text,
        "phoneme_score": round(float(phoneme_score), 2),
        "acoustic_score": round(float(acoustic_score), 2),
        "final_score": round(float(final_score), 2),
    }
