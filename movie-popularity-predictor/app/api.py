import joblib
import re
import os
import sys
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import hstack

try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
except ImportError:
    print(" NLTK not found. Please ensure it's in requirements-app.txt")
    sys.exit(1)

MODEL_PATH = os.getenv('MODEL_PATH', 'random_forest_model.pkl')
TFIDF_PATH = os.getenv('TFIDF_PATH', 'tfidf_vectorizer.pkl')

app = FastAPI(
    title="Movie Popularity Predictor API",
    description="Predicts movie popularity based on overview and vote metrics.",
    version="1.0.0"
)

try:
    model = joblib.load(MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)
    print(" Model and TF-IDF vectorizer loaded successfully.")
except FileNotFoundError:
    print(f" FATAL ERROR: Model or TF-IDF file not found.")
    print(f"Ensure '{MODEL_PATH}' and '{TFIDF_PATH}' exist after running the training script.")
    sys.exit(1)
except Exception as e:
    print(f" An error occurred while loading the model files: {e}")
    sys.exit(1)


try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"Could not load NLTK data. Ensure it is downloaded. Error: {e}")
    sys.exit(1)

def clean_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(cleaned_tokens)


class PredictionRequest(BaseModel):
    overview: str
    vote_average: float
    vote_count: int

class PredictionResponse(BaseModel):
    predicted_popularity: float



@app.get("/", tags=["General"])
def read_root():
    return {"status": "API is running successfully"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_popularity(request: PredictionRequest):
    try:
 
        cleaned_overview = clean_text(request.overview)
        

        text_features = tfidf_vectorizer.transform([cleaned_overview])
        numeric_features = pd.DataFrame([[request.vote_average, request.vote_count]])
        

        combined_features = hstack([text_features, numeric_features.values])
        

        prediction = model.predict(combined_features)[0]
        
        return PredictionResponse(predicted_popularity=prediction)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

