import streamlit as st
import pandas as pd
import numpy as np
import gdown
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Skill Recommender Bot", layout="wide")

st.title("🔍 AI Skill Recommender Bot")
st.markdown("Enter your skills to find the most relevant job roles from the AI market.")

# 🔽 DOWNLOAD .pkl FILE FROM GOOGLE DRIVE
file_id = "1FgEnP1iwKTsozqnRhPYoSbbq5oD53JzS"  # Your actual .pkl file
url = f"https://drive.google.com/uc?id={file_id}"
output = "One-Hot-Encoded.pkl"
gdown.download(url, output, quiet=False)

# 🔽 LOAD THE DATAFRAME
df = pd.read_pickle(output)

# ✅ USE ONLY BINARY SKILL COLUMNS (same as in Colab)
skill_columns = df.select_dtypes(include=['int', 'float', 'bool']).columns.tolist()
job_vectors = df[skill_columns].values

# 🔽 USER INPUT
user_input = st.text_input("🧠 Enter your known skills (comma-separated):", "")

if user_input:
    user_skills = [skill.strip().lower() for skill in user_input.split(",")]

    # Build vector
    user_vector = np.zeros(len(skill_columns))
    matched_skills = []

    for i, skill in enumerate(skill_columns):
        if skill.strip().lower() in user_skills:
            user_vector[i] = 1
            matched_skills.append(skill)

    st.info(f"🧩 Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}")

    # 🔽 COSINE SIMILARITY
    similarity_scores = cosine_similarity([user_vector], job_vectors)[0]
    df['Similarity'] = similarity_scores

    # 🔽 TOP JOB MATCHES
    if df['Similarity'].max() > 0:
        top_matches = df.sort_values(by='Similarity', ascending=False).head(10)
        st.subheader("✅ Top Matching Job Titles")
        st.dataframe(top_matches[['job_title', 'Similarity']].reset_index(drop=True))
        st.success(f"Best Match: {top_matches.iloc[0]['job_title']}")
    else:
        st.warning("⚠️ No close job matches found for your input. Try different or more skills.")
