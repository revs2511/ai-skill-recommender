import streamlit as st
import pandas as pd
import numpy as np
import gdown
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Skill Recommender Bot", layout="wide")

st.title("🤖 AI Skill Recommender Bot")
st.markdown("Enter your known skills below to find the most suitable job roles in the AI market.")

# 🔽 DOWNLOAD .pkl FROM GOOGLE DRIVE
file_id = "1XFAm79H1XqWBM61CwZ4dyABFxG2L8Mox"  # 🔁 Replace with your 120MB file's ID
url = f"https://drive.google.com/uc?id={file_id}"
output = "One-Hot-Encoded-Streamlit.pkl"
gdown.download(url, output, quiet=False)

# 🔽 LOAD THE DATAFRAME
try:
    df = pd.read_pickle(output)
    st.success("✅ Dataset loaded successfully!")
    st.write("📊 Data shape:", df.shape)
except Exception as e:
    st.error(f"❌ Failed to load file: {e}")
    st.stop()

# 🔽 Extract skill columns
skill_columns = df.select_dtypes(include=['int', 'float', 'bool']).columns.tolist()
job_vectors = df[skill_columns].values

# 🔽 USER INPUT
user_input = st.text_input("✍️ Enter your skills (comma-separated):", "")

if user_input:
    user_skills = [skill.strip().lower() for skill in user_input.split(",")]
    
    user_vector = np.zeros(len(skill_columns))
    matched_skills = []

    for i, skill in enumerate(skill_columns):
        if skill.lower() in user_skills:
            user_vector[i] = 1
            matched_skills.append(skill)

    st.info(f"🧩 Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}")

    try:
        similarity_scores = cosine_similarity([user_vector], job_vectors)[0]
        df['Similarity'] = similarity_scores

        if df['Similarity'].max() > 0:
            top_matches = df.sort_values(by='Similarity', ascending=False).head(10)
            st.subheader("🔍 Top Job Matches")
            st.dataframe(top_matches[['job_title', 'Similarity']].reset_index(drop=True))
            st.success(f"🎯 Best Match: {top_matches.iloc[0]['job_title']}")
        else:
            st.warning("⚠️ No strong matches found. Try more or broader skills.")

    except Exception as e:
        st.error(f"❌ Error computing similarity: {e}")
