from db_user import fetch_all_user
from embeddings_user import generate_user_embeddings
from faiss_index_user import create_or_update_user_index
import pickle
import pandas as pd
import time
 
# -----------------------------
# Set refresh interval (seconds)
# -----------------------------
refresh_interval = 6000

while True:
    print("⏳ Checking for new user...")
 
    # 1. Fetch all jobs from MySQL
    df_all = fetch_all_user()
 
    # 2. Load existing job data
    try:
        with open("user_data.pkl", "rb") as f:
            df_old = pickle.load(f)
        old_ids = set(df_old["user_id"].tolist())
    except FileNotFoundError:
        df_old = None
        old_ids = set()
 
    # 3. Filter only new jobs
    df_new = df_all[~df_all["user_id"].isin(old_ids)]
 
    if df_new.empty:
        print("No new users found. Waiting...")
        time.sleep(refresh_interval)
        continue
 
    print(f"🔎 Found {len(df_new)} new users . Generating embeddings...")
 
    # 4. Generate embeddings (now returns job_ids, embeddings)
    user_ids, embeddings = generate_user_embeddings(df_new)
 
    # 5. Update FAISS index (send all three)
    create_or_update_user_index(embeddings, user_ids)
 
    # 6. Update job_data.pkl (cache all jobs)
    if df_old is not None:
        df_updated = pd.concat([df_old, df_new]).drop_duplicates(subset="user_id")
    else:
        df_updated = df_new
 
    with open("user_data.pkl", "wb") as f:
        pickle.dump(df_updated, f)
 
    print("✅ FAISS index, and user data updated successfully! Waiting for next check...")
    time.sleep(refresh_interval)