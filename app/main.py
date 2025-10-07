from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile, librosa, requests, os
from app.utils import compute_acoustic_score, compute_phoneme_score

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# AssemblyAI Inference API
# -------------------------
AAI_TOKEN = os.getenv("AAI_TOKEN")  # Set your AssemblyAI API key in Render or .env
HEADERS = {"authorization": AAI_TOKEN}
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

@app.post("/analyze/")
async def analyze(audio: UploadFile = File(...), text: str = Form(...)):
    # Save uploaded audio temporarily
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(await audio.read())
    tmp.close()

    # Upload audio to AssemblyAI
    with open(tmp.name, "rb") as f:
        upload_resp = requests.post("https://api.assemblyai.com/v2/upload", 
                                    headers=HEADERS, data=f)
    if upload_resp.status_code != 200:
        return {"error": f"Audio upload failed: {upload_resp.text}"}
    audio_url = upload_resp.json()["upload_url"]

    # Request transcription
    transcript_req = {
        "audio_url": audio_url,
        "auto_chapters": False,
        "speaker_labels": False
    }
    transcript_resp = requests.post(TRANSCRIPT_URL, headers=HEADERS, json=transcript_req)
    if transcript_resp.status_code != 200:
        return {"error": f"Transcription request failed: {transcript_resp.text}"}
    transcript_id = transcript_resp.json()["id"]

    # Poll until transcription is complete
    while True:
        check_resp = requests.get(f"{TRANSCRIPT_URL}/{transcript_id}", headers=HEADERS)
        if check_resp.status_code != 200:
            return {"error": f"Transcription check failed: {check_resp.text}"}
        status = check_resp.json()["status"]
        if status == "completed":
            predicted_text = check_resp.json()["text"].lower()
            break
        elif status == "failed":
            return {"error": "Transcription failed"}
    
    # Load audio locally for scoring
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
