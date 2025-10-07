from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile, librosa, torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from app.utils import extract_features, compute_acoustic_score, compute_phoneme_score

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load local Speech2Text model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "facebook/s2t-small-librispeech-asr"
processor = Speech2TextProcessor.from_pretrained(MODEL_NAME)
model = Speech2TextForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)


@app.post("/analyze/")
async def analyze(audio: UploadFile = File(...), text: str = Form(...)):
    # Save uploaded file temporarily
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(await audio.read())
    tmp.close()

    # Load audio for processing
    y, sr = librosa.load(tmp.name, sr=16000)
    
    # Convert waveform to model input
    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # Generate predicted text
    with torch.no_grad():
        generated_ids = model.generate(input_features)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()

    # Compute pronunciation scores
    acoustic_score = compute_acoustic_score(y, sr)
    phoneme_score = compute_phoneme_score(text, predicted_text)
    final_score = (0.6 * phoneme_score) + (0.4 * acoustic_score)

    return {
        "predicted_text": predicted_text,
        "phoneme_score": round(float(phoneme_score), 2),
        "acoustic_score": round(float(acoustic_score), 2),
        "final_score": round(float(final_score), 2),
    }
