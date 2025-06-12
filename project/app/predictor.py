import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../indobert_model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Ganti dengan label sesuai model Anda
EMOTION_LABELS = ['Marah', 'Senang', 'Sedih', 'Takut', 'Cinta', 'Netral']

def classify_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().tolist()
    result = list(zip(EMOTION_LABELS, probs))
    result.sort(key=lambda x: x[1], reverse=True)
    return result[0][0]  # hanya label dengan probabilitas tertinggi

# def classify_emotion(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().tolist()
#     result = list(zip(EMOTION_LABELS, probs))
#     result.sort(key=lambda x: x[1], reverse=True)
#     top_label, top_score = result[0]
#     return f"{top_label} ({top_score*100:.1f}%)"
