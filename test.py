import requests

HF_TOKEN = "hf_your_token_here"
API_URL = "https://api-inference.huggingface.co/models/Systran/faster-whisper-small"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

with open("fruits.wav", "rb") as f:
    data = f.read()

response = requests.post(API_URL, headers=headers, data=data)
print(response.status_code)
print(response.text)
