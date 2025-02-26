import faiss
import numpy as np

# Load FAISS index
index = faiss.read_index("city_embeddings.index")

# Get the number of stored vectors
print(f"🔍 Total Cities in Index: {index.ntotal}")
print(f"📏 Vector Dimension: {index.d}")

