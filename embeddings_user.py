from sentence_transformers import SentenceTransformer
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
 
 
# Load model globally (only once)
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
 
 
def generate_user_embeddings(df):
    """
    Generate embeddings for user using user_id , user_profile , user_experience, user_state ,user_skills .
    user_id is kept separately for mapping back to the DB.
    """
    texts = []
    ids = []   # keep job_id for mapping
 
    for _, row in df.iterrows():
        text = (
            f"{row['user_profile']} at {row['user_experience']} "
            f"in {row['user_city']}. "
            f"Required skills: {row['user_skills']}. "
        )
 
        texts.append(text)
        ids.append(row["user_id"])
 
 
    embeddings = model.encode(texts, show_progress_bar=True)
    return ids, np.array(embeddings).astype("float32")
 
 