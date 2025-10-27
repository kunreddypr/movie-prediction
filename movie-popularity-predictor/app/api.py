from pathlib import Path
import joblib
import re
import os
from typing import Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import hstack
import psycopg2
from psycopg2 import OperationalError

try:
    from app.nlp_utils import get_language_tools, ensure_nltk_data
except ImportError:  # pragma: no cover - allows running as a script
    from nlp_utils import get_language_tools, ensure_nltk_data

try:
    from app.train_model import train_model, MODEL_PATH, TFIDF_PATH, TrainingError
except ImportError:  # pragma: no cover - allows running via python -m
    from train_model import train_model, MODEL_PATH, TFIDF_PATH, TrainingError

DEFAULT_MODEL_PATH = Path(MODEL_PATH)
DEFAULT_TFIDF_PATH = Path(TFIDF_PATH)

MODEL_PATH = Path(os.getenv('MODEL_PATH', str(DEFAULT_MODEL_PATH)))
TFIDF_PATH = Path(os.getenv('TFIDF_PATH', str(DEFAULT_TFIDF_PATH)))

app = FastAPI(
    title="Movie Popularity Predictor API",
    description="Predicts movie popularity based on overview and vote metrics.",
    version="1.0.0"
)

def _load_model_artifacts() -> Tuple[object, object]:
    """Load the model artefacts, training them if they do not exist."""

    missing = [path for path in (MODEL_PATH, TFIDF_PATH) if not path.exists()]
    if missing:
        print(" Model artefacts missing; triggering training run...")
        try:
            train_model()
        except TrainingError as exc:
            print(f" Training failed while creating artefacts: {exc}")
            raise

    try:
        model_obj = joblib.load(MODEL_PATH)
        vectorizer_obj = joblib.load(TFIDF_PATH)
        print(" Model and TF-IDF vectorizer loaded successfully.")
        return model_obj, vectorizer_obj
    except FileNotFoundError as exc:
        print(f" FATAL ERROR: Required artefact not found after training: {exc}")
        raise
    except Exception as exc:
        print(f" An error occurred while loading the model files: {exc}")
        raise


ensure_nltk_data()
lemmatizer, stop_words = get_language_tools()
model, tfidf_vectorizer = _load_model_artifacts()

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

        _store_prediction(DB_CONFIG, request, prediction)

        return PredictionResponse(predicted_popularity=prediction)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

