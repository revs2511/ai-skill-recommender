import streamlit as st
import pandas as pd
import numpy as np
import gdown
from sklearn.metrics.pairwise import cosine_similarity

st.title("🔍 AI Skill Recommender Bot")
st.markdown("Enter your skills to find the most relevant job roles from the AI market.")

# Load file from Google Drive
file_id = "1ysLMY4BasEQ68zYo-qtoPaqGA_W8T9b"  # <--- Replace this
url = f"https://drive.google.com/uc?id={file_id}"
output = "One-Hot-Encoded.pkl"
gdown.download(url, output, quiet=False)

# Load the dataframe
df = pd.read_pickle(output)

# Extract the skill columns only (excluding job_title, job_skills)
skill_columns = df.columns.difference(['job_title', 'job_skills'])

# Input from user
user_input = st.text_input(" Enter your known skills (comma-separated):", "")

if user_input:
    user_skills = [skill.strip().lower() for skill in user_input.split(",")]

    # Create a vector of same shape as skill_columns
    user_vector = np.zeros(len(skill_columns))

    for i, skill in enumerate(skill_columns):
        if skill.lower() in user_skills:
            user_vector[i] = 1

    # Compute similarity
    job_vectors = df[skill_columns].values
    similarities = cosine_similarity([user_vector], job_vectors)[0]
    df['Similarity'] = similarities

    top_matches = df.sort_values(by='Similarity', ascending=False).head(5)

    st.subheader(" Top Job Matches")
    st.dataframe(top_matches[['job_title', 'Similarity']].reset_index(drop=True))

    st.success(f"Top Match: {top_matches.iloc[0]['job_title']}")

