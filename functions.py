import streamlit as st
from typing import List
# Define a function to calculate embeddings
def calculate_embeddings(texts, model):
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings

#define a function to clean up data
def clean_textfiled(df, TEXT_FIELD_NAME):
    # Handle missing or non-string values in the TEXT_FIELD_NAME column
    df[TEXT_FIELD_NAME] = df[TEXT_FIELD_NAME].fillna('')  # Replace NaN with empty string
    df[TEXT_FIELD_NAME] = df[TEXT_FIELD_NAME].astype(str)  # Ensure all values are strings

    df[TEXT_FIELD_NAME] =  df[TEXT_FIELD_NAME].map(lambda x: x.lstrip('Make sure this fits by entering your model number. |').rstrip('aAbBcC'))
    return df


