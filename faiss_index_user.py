import faiss
 
import pickle
 
import os
 
import numpy as np
 
 
def create_or_update_user_index(embeddings, user_ids):
 
    """
 
    Create or update FAISS index with user embeddings and ID mapping.
 
    """
 
    index_file = "faiss_user.index"
 
    mapping_file = "user_mapping.pkl"
 
    dim = embeddings.shape[1]
 
    # --- Load or create FAISS index ---
 
    if os.path.exists(index_file):
 
        index = faiss.read_index(index_file)
 
        if index.d != dim:
 
            raise ValueError(
 
                f"Embedding dimension mismatch: index={index.d}, new={dim}"
 
            )
 
    else:
 
        index = faiss.IndexFlatL2(dim)
 
    # --- Load or create ID mapping ---
 
    if os.path.exists(mapping_file):
 
        with open(mapping_file, "rb") as f:
 
            id_mapping = pickle.load(f)
 
    else:
 
        id_mapping = []
 
    # --- Add only new user ---
 
    new_embeddings = []
 
    new_ids = []
 
    existing_ids = set(id_mapping)
 
    for user_id, emb in zip(user_ids , embeddings):
 
        if user_id not in existing_ids:  # avoid duplicates
 
            new_ids.append(user_id)
 
            new_embeddings.append(emb)
 
    if new_embeddings:
 
        index.add(np.array(new_embeddings).astype("float32"))
 
        id_mapping.extend(new_ids)
 
        # Save updated files
 
        faiss.write_index(index, index_file)
 
        with open(mapping_file, "wb") as f:
 
            pickle.dump(id_mapping, f)
 
        print(f"✅ Added {len(new_ids)} new user to FAISS index")
 
    else:
 
        print("ℹ️ No new user to add")
 
    print(f"📊 Total indexed jobs: {len(id_mapping)}")