import streamlit as st



st.set_page_config(
    page_title="Home Page & Data Loading"
)

st.sidebar.success("Each page is another stage in the demo, starting with the data loading phase.")

st.markdown(
    """
**This application is a semantic search demo complete with data uploading and querying.**


**You can start on the db upload page, where you will be uploading a CSV file of data. This uploaded data will be vectorized and the resulting file of embeddings will be saved to your local folder. From there, your data will be uploaded to the vector db.**

**Once you have uploaded your data to the db, you can go on the search page and look up results semantically similar to your query**
"""
)

