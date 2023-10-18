
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import streamlit as st


st.set_page_config(
    page_title="Data Search"
)


st.markdown(
"""
**At this point, you should have your data vectorized and uploaded to the db. This means you can now (semantically) search it! Go ahead and try it below** 
"""
)
# search qdrant
collection_name = "amazon-products"

client = QdrantClient('http://localhost:6333')
# Initialize encoder model
model = SentenceTransformer('all-MiniLM-L6-v2')
text  = st.text_input("Search your data here")

vector = model.encode(text).tolist()

        # Use `vector` for search for closest vectors in the collection
hits = client.search(
    collection_name=collection_name,
    query_vector=vector,
    limit=3
)

# st.json(hits.payload)
# df = pd.DataFrame(hits)

# st.dataframe(df)
count = 0
for hit in hits:
    st.image(hit.payload["Image"])
    st.table(hit.payload)



    