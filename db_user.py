import mysql.connector
import pandas as pd
 
def fetch_all_user():
    """Fetch all jobs from the MySQL database"""
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Ram@1234",   # replace with your actual password
        database="user_db"
    )
    cursor = conn.cursor()
 
    # Fetch all records from jobs table
    cursor.execute("""
    SELECT
        user_id,
        user_profile,
        user_skills,
        user_experience,
        user_city,
        user_status
    FROM users
    WHERE user_status = 1
""")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
 
    # ✅ Proper column names as a list
    df = pd.DataFrame(
    rows,
    columns=[
        "user_id", "user_profile",
        "user_skills", "user_experience", "user_city"
    ]
)

    # 1. Remove rows where job_id is NULL
    df = df[df["user_id"].notna()]
 
    df = df.drop_duplicates(subset=["user_id"], keep="first")
 
    # 2. Replace NULL values in non-numeric columns → "Unknown"
    string_columns = ["user_id", "user_profile","user_skills", "user_experience", "user_city"]
 
# Replace both NaN and "" (empty strings)
    df[string_columns] = df[string_columns].replace("", "Not mentioned").fillna("Not mentioned")
 
    return df
df = fetch_all_user()
print(df.info())