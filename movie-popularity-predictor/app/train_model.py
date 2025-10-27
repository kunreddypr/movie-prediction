from pathlib import Path
import pandas as pd
import joblib
import re
import os
from typing import Optional, Tuple

try:
    from app.nlp_utils import get_language_tools
except ImportError:  # pragma: no cover - allows running as a script
    from nlp_utils import get_language_tools

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack


class TrainingError(RuntimeError):
    """Raised when the training pipeline cannot complete successfully."""

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DATA_PATH = os.getenv('SOURCE_DATA_PATH', str(BASE_DIR / 'movies.csv'))
MODEL_PATH = os.getenv('MODEL_PATH', str(BASE_DIR / 'random_forest_model.pkl'))
TFIDF_PATH = os.getenv('TFIDF_PATH', str(BASE_DIR / 'tfidf_vectorizer.pkl'))
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
MAX_FEATURES = int(os.getenv('MAX_FEATURES', 800))
N_ESTIMATORS = int(os.getenv('N_ESTIMATORS', 8))


try:
    lemmatizer, stop_words = get_language_tools()
except Exception as exc:  # pragma: no cover - defensive, we re-raise below
    raise TrainingError(f"Could not prepare NLTK tooling: {exc}") from exc

def clean_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(cleaned_tokens)

def train_model() -> Tuple[str, str]:
    print("--- Starting Model Training Process ---")
    try:
        df = pd.read_csv(SOURCE_DATA_PATH)
    except FileNotFoundError as exc:
        raise TrainingError(f"Source data file not found at '{SOURCE_DATA_PATH}'.") from exc

    print(f"Data loaded successfully from '{SOURCE_DATA_PATH}'. Shape: {df.shape}")

    df['cleaned_overview'] = df['overview'].apply(clean_text)
    df.dropna(subset=['popularity', 'vote_average', 'vote_count', 'cleaned_overview'], inplace=True)
    print(f"Data cleaned. Shape after dropping NAs: {df.shape}")

    if df.empty:
        raise TrainingError("No data remaining after cleaning. Cannot train model.")

    X_text = df['cleaned_overview']
    X_numeric = df[['vote_average', 'vote_count']]
    y = df['popularity']

    X_text_train, X_text_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
        X_text, X_numeric, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(" Data split into training and testing sets.")

    tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
    X_text_train_tfidf = tfidf.fit_transform(X_text_train)
    X_text_test_tfidf = tfidf.transform(X_text_test)
    print(" TF-IDF vectorizer fitted on training data.")

    X_train_combined = hstack([X_text_train_tfidf, X_numeric_train.values])
    X_test_combined = hstack([X_text_test_tfidf, X_numeric_test.values])
    print("Text and numeric features combined.")

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_combined, y_train)
    print("Random Forest model trained.")

    predictions = model.predict(X_test_combined)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model evaluation - Mean Squared Error: {mse:.4f}")

    joblib.dump(tfidf, TFIDF_PATH)
    print(f"TF-IDF vectorizer saved to {TFIDF_PATH}")
    joblib.dump(model, MODEL_PATH)
    print(f" Random Forest model saved to {MODEL_PATH}")

    print("\n Model training and saving process completed successfully!")
    return MODEL_PATH, TFIDF_PATH

if __name__ == '__main__':
    try:
        train_model()
    except TrainingError as exc:
        print(f" A FATAL ERROR occurred during the training process: {exc}")
        raise SystemExit(1) from exc

