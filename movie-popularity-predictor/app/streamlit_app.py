import streamlit as st
import requests
import pandas as pd
import psycopg2
from io import StringIO
import os


API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
FASTAPI_URL = f"{API_BASE_URL}/predict"
# Allow the API request timeout to be configured via env var
API_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "20"))

st.set_page_config(
    page_title="Movie Popularity Predictor",
    layout="wide"
)

@st.cache_resource
def init_connection():
    try:
        host = os.getenv("PREDICTIONS_HOST", os.getenv("POSTGRES_HOST"))
        dbname = os.getenv("PREDICTIONS_DB", os.getenv("POSTGRES_DB"))
        user = os.getenv("PREDICTIONS_USER", os.getenv("POSTGRES_USER"))
        password = os.getenv("PREDICTIONS_PASSWORD", os.getenv("POSTGRES_PASSWORD"))
        port = os.getenv("PREDICTIONS_PORT", os.getenv("POSTGRES_PORT", 5432))
        conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password, port=port)
        return conn
    except (psycopg2.OperationalError, Exception) as e:
        st.error(f"Error connecting to PostgreSQL database: {e}")
        st.info("Please check the POSTGRES environment variables in your docker-compose.yml file.")
        return None


@st.cache_data(ttl=600)
def fetch_prediction_history(_conn):
    if _conn is None:
        return pd.DataFrame()
    with _conn.cursor() as cur:
        cur.execute("SELECT prediction_time, overview, vote_average, vote_count, predicted_popularity FROM prediction_history ORDER BY prediction_time DESC LIMIT 100;")
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=columns)


conn = init_connection()


st.title("Movie Popularity Predictor")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Prediction History"])


with tab1:
    st.header("Predict a Single Movie's Popularity")
    with st.form(key='movie_form'):
        overview_text = st.text_area(
            "Movie Overview/Synopsis",
            "A small town sheriff battles a ruthless outlaw who has returned for a final bloody confrontation.",
            height=200
        )
        col1, col2 = st.columns(2)
        with col1:
            vote_average = st.slider("Average Vote Score (1.0 to 10.0)", 1.0, 10.0, 7.5, 0.1)
        with col2:
            vote_count = st.number_input("Vote Count (Minimum 100)", 100, value=2500, step=100)
        submit_button = st.form_submit_button(label='Get Popularity Prediction')

    if submit_button:
        if not overview_text:
            st.error("Please provide a movie overview.")
        else:
            payload = {"overview": overview_text, "vote_average": vote_average, "vote_count": vote_count}
            try:
                response = requests.post(FASTAPI_URL, json=payload, timeout=API_TIMEOUT)
                if response.status_code == 200:
                    result = response.json()
                    st.metric("Predicted Popularity Score", f"{result.get('predicted_popularity'):.4f} ")
                else:
                    st.error(f"Error calling API. Status: {response.status_code}")
                    st.json(response.json())
            except requests.exceptions.ConnectionError:
                st.error(f"Connection Error: Could not connect to the API service at {API_BASE_URL}.")
            except requests.exceptions.ReadTimeout:
                st.error(f"API timed out after {API_TIMEOUT}s. Try again or increase API_TIMEOUT_SECONDS.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


with tab2:
    st.header("Predict Popularity for a Batch of Movies")
    uploaded_file = st.file_uploader("Upload a CSV file with `overview`, `vote_average`, `vote_count` columns.", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = {'overview', 'vote_average', 'vote_count'}
            if not required_cols.issubset(df.columns):
                st.error(f"CSV must contain: {', '.join(required_cols)}")
            else:
                st.success("CSV file validated!")
                if st.button("Run Batch Prediction"):
                    predictions = []
                    progress_bar = st.progress(0, text="Starting...")
                    for index, row in df.iterrows():
                        payload = row.to_dict()
                        try:
                            response = requests.post(FASTAPI_URL, json=payload, timeout=API_TIMEOUT)
                            predictions.append(response.json().get('predicted_popularity') if response.status_code == 200 else None)
                        except requests.RequestException:
                            predictions.append(None)
                        progress_bar.progress((index + 1) / len(df), text=f"Processing movie {index + 1}/{len(df)}")
                    df['predicted_popularity'] = predictions
                    st.dataframe(df)
                    st.download_button("Download results", df.to_csv(index=False).encode('utf-8'), 'prediction_results.csv', 'text/csv')
        except Exception as e:
            st.error(f"Error processing file: {e}")


with tab3:
    st.header("View Past Predictions")
    if st.button("Refresh History"):
        st.cache_data.clear()
        st.rerun()

    if conn:
        history_df = fetch_prediction_history(conn)
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No prediction history found. Make a prediction in the first tab to see it here!")
    else:
        st.warning("Could not display history because the database connection failed. Check service logs.")

st.markdown("---")
st.caption("Backend: FastAPI | Frontend: Streamlit | Database: PostgreSQL")
