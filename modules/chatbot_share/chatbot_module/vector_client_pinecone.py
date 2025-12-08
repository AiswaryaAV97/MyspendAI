import os
from dotenv import load_dotenv
# ensure project .env is loaded when this module is imported
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
env_path = os.path.join(root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
import numpy as np
from typing import List, Dict, Any

# new Pinecone client (v2+)
try:
    from pinecone import Pinecone
except Exception as e:
    raise RuntimeError("pinecone client import failed; ensure 'pinecone' package is installed") from e

PINE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "smartspend-rag")
EMBED_DIM = int(os.getenv("EMBED_DIM", 384))

if not PINE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set in environment")

# Initialize client
pc = Pinecone(api_key=PINE_API_KEY)

# list_indexes may return list or dict depending on client version; normalize
_existing = pc.list_indexes()
if isinstance(_existing, dict) and "names" in _existing:
    existing_indexes = _existing["names"]
else:
    # list of index objects, extract names
    existing_indexes = [idx.get("name") if isinstance(idx, dict) else idx.name for idx in _existing]

# Skip auto-create — index created manually in Pinecone console. Verify it exists:
if INDEX_NAME not in existing_indexes:
    raise RuntimeError(
        f"Pinecone index '{INDEX_NAME}' not found. Existing indexes: {existing_indexes}. "
        f"Create the index in the Pinecone console with dimension={EMBED_DIM} and metric=cosine."
    )

# connect to the existing index
index = pc.Index(INDEX_NAME)

def upsert_vectors(items: List[Dict[str, Any]], batch_size: int = 100):
    """
    items: list of {"id": str, "vector": np.ndarray, "metadata": dict}
    """
    to_upsert = [
        {"id": str(it["id"]), "values": (it["vector"].tolist() if hasattr(it["vector"], "tolist") else list(map(float, it["vector"]))), "metadata": it.get("metadata", {})}
        for it in items
    ]
    for i in range(0, len(to_upsert), batch_size):
        index.upsert(vectors=to_upsert[i:i+batch_size])

def query_vector(query_vector: np.ndarray, top_k: int = 5, filter: Dict = None):
    q = (query_vector.tolist() if hasattr(query_vector, "tolist") else list(map(float, query_vector)))
    res = index.query(vector=q, top_k=top_k, filter=filter, include_metadata=True)
    # normalize return structure
    matches = res.get("matches") or res.get("results") or []
    return matches

def delete_vectors(ids: List[str]):
    index.delete(ids=ids)
