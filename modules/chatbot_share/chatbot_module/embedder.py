from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts):
        arr = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.array(arr, dtype="float32")