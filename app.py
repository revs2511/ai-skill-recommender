import streamlit as st
import pandas as pd
import numpy as np
import gdown
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Skill Recommender Bot", layout="wide")

st.title("🔍 AI Skill Recommender Bot")
st.markdown("Enter your skills to find the most relevant job roles from the AI market.")

# 🔽 DOWNLOAD .pkl FILE FROM GOOGLE DRIVE
file_id = "1ysLMY4BasEQ68zYo-qtoPaqGA_W8T9b_"
url = f"https://drive.google.com/uc?id={file_id}"
output = "One-Hot-Encoded.pkl"
gdown.download(url, output, quiet=False)

# 🔽 LOAD THE DATAFRAME
df = pd.read_pickle(output)

# 🔽 EXTRACT SKILL COLUMNS
skill_columns = df.columns.difference(['job_title', 'job_skills'])
job_vectors = df[skill_columns].values

# 🔽 USER INPUT
user_input = st.text_input("🧠 Enter your known skills (comma-separated):", "")

if user_input:
    user_skills = [skill.strip().lower() for skill in user_input.split(",")]
    user_vector = np.zeros(len(skill_columns))

    # Normalize dataset column names
    normalized_columns = [col.strip().lower() for col in skill_columns]
    skill_index_map = {skill: i for i, skill in enumerate(normalized_columns)}

    # Build user vector based on matching skills
    matched_skills = []
    for skill in user_skills:
        if skill in skill_index_map:
            user_vector[skill_index_map[skill]] = 1
            matched_skills.append(skill)

    # Show matched skills
    st.info(f"🧩 Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}")

    # 🔽 COSINE SIMILARITY
    similarities = cosine_similarity([user_vector], job_vectors)[0]
    df['Similarity'] = similarities

    # 🔽 TOP JOB MATCHES
    if df['Similarity'].max() > 0:
        top_matches = df.sort_values(by='Similarity', ascending=False).head(10)
        st.subheader("✅ Top Matching Job Titles")
        st.dataframe(top_matches[['job_title', 'Similarity']].reset_index(drop=True))
        st.success(f"Best Match: {top_matches.iloc[0]['job_title']}")
    else:
        st.warning("⚠️ No close job matches found for your input. Try different or more skills.")
