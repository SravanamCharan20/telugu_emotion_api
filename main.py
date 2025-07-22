from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from indic_transliteration.sanscript import transliterate, ITRANS, TELUGU

# Load all model components
MODEL_PATH = "telugu_story_emotion_lstm_model.h5"
TOKENIZER_PATH = "tokenizer_lstm.pkl"
MLB_PATH = "label_binarizer_lstm.pkl"

app = FastAPI(title="Telugu Story Emotion Detection API")

# Load model and assets once
try:
    model = load_model(MODEL_PATH, compile=False)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(MLB_PATH, "rb") as f:
        mlb = pickle.load(f)
    MAX_LEN = model.input_shape[1]
except Exception as e:
    raise RuntimeError(f"Error loading model assets: {e}")

# Request format
class StoryInput(BaseModel):
    story: str
    top_n: int = 3

# Utility functions
def is_telugu(text):
    return bool(re.search(r'[\u0C00-\u0C7F]', text))

def preprocess_story(text):
    if not is_telugu(text):
        text = transliterate(text, ITRANS, TELUGU)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded

# API Endpoint
@app.post("/predict")
def predict_emotions(data: StoryInput):
    try:
        input_seq = preprocess_story(data.story)
        preds = model.predict(input_seq)[0]
        top_indices = preds.argsort()[-data.top_n:][::-1]

        result = []
        for i in top_indices:
            result.append({
                "emotion": mlb.classes_[i],
                "confidence": round(float(preds[i]), 4)
            })

        return {"story": data.story, "predicted_emotions": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))