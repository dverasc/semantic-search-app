import streamlit as st
from functions import calculate_embeddings, clean_textfiled
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from qdrant_client.http import models as rest
import pandas as pd
from qdrant_client import QdrantClient, models
import json

import numpy as np

st.set_page_config(
    page_title="Data Transformed"
)



st.markdown(
"""
Start by uploading a CSV file of data. Your uploaded data will be transformed and vectorized and the resulting file of embeddings will be saved to your local folder.
You will need this file for the next phase.** 
"""
)

TEXT_FIELD_NAME = st.text_input("Enter the field name that you will use for the embeddings")
data_file = st.file_uploader("Please upload a CSV file", type="csv")
if data_file is not None:
    df = pd.read_csv(data_file)
    df = clean_textfiled(df, TEXT_FIELD_NAME)
    # vectors file will save to your local folder
    npy_file_path = data_file.name

    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')


    # # Split the data into chunks to save RAM
    batch_size = 1000
    num_chunks = len(df) // batch_size + 1

    embeddings_list = []

    # Iterate over chunks and calculate embeddings
    for i in tqdm(range(num_chunks), desc="Calculating Embeddings"):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_texts = df[TEXT_FIELD_NAME].iloc[start_idx:end_idx].tolist()
        batch_embeddings = calculate_embeddings(batch_texts, model)
        embeddings_list.extend(batch_embeddings)
    
    # Convert embeddings list to a numpy array
    embeddings_array = np.array(embeddings_list)

    # Save the embeddings to an NPY file
    np.save(npy_file_path, embeddings_array)

    print(f"Embeddings saved to {npy_file_path}")


else:
    st.warning("you need to upload a csv file")


