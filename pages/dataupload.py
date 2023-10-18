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
    page_title="Data Upload to DB"
)

VECTOR_FIELD_NAME = st.text_input("Enter the field name that you will use for the embeddings field in the db")


st.markdown(
    """
   **Upload your embeddings file (should be the .npy file in your local folder). This file's vector data will be uploaded to your local Qdrant db instance.**
   **You will also need to upload the csv file that was used to create the embeddings file. This is so we can upload the appropriate payload along with the vectors**
    """
    )
embed_data_file = st.file_uploader("Please upload the corresponding vectors file", type="npy")
data_file = st.file_uploader("Please upload the appropriate CSV file", type="csv")
TEXT_FIELD_NAME = st.text_input("Enter the field name that you will use for the embeddings")
if embed_data_file is not None and data_file is not None:
    client = QdrantClient('http://localhost:6333')
    df = pd.read_csv(data_file)
    df = clean_textfiled(df, TEXT_FIELD_NAME)
    payload = df.to_json(orient='records')
    payload = json.loads(payload)
    vectors = np.load(embed_data_file)
    client.recreate_collection(
    collection_name="amazon-products",
    vectors_config={
        VECTOR_FIELD_NAME: models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
            on_disk=True,
        )
    },
    # Quantization is optional, but it can significantly reduce the memory usage
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True
        )
    )
)
    client.upload_collection(
    collection_name="amazon-products",
    vectors={
        VECTOR_FIELD_NAME: vectors
    },
    payload=payload,
    ids=None,  # Vector ids will be assigned automatically
    batch_size=256  # How many vectors will be uploaded in a single request?
)



else:
    st.warning("you need to upload an npy file")