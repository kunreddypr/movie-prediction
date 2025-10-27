import pandas as pd
import os
import uuid


RAW_DATA_PATH = 'data/raw-data'
SOURCE_FILE = 'movies.csv' 

def generate_ingestion_file(n_samples=50):

    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    
    try:
        df = pd.read_csv(SOURCE_FILE)
    except FileNotFoundError:
        print(f"Error: Source file '{SOURCE_FILE}' not found. Ensure it's in the root directory.")
        return


    sample_df = df.sample(n=n_samples, replace=False)


    file_id = uuid.uuid4().hex[:8]
    file_name = f"movies_sample_{file_id}.csv"
    output_path = os.path.join(RAW_DATA_PATH, file_name)


    sample_df.to_csv(output_path, index=False)
    
    print(f"Successfully generated new file: {output_path} with {n_samples} records.")

if __name__ == '__main__':
    generate_ingestion_file(n_samples=100)
    generate_ingestion_file(n_samples=50)