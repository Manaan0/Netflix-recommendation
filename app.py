import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="Netflix Recommender 🎬",
    page_icon="🍿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df['description'] = df['description'].fillna("No Description")
    return df

df = load_data()

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index map
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

# Recommendation function
def recommend(title):
    title = title.strip().lower()
    if title not in indices:
        return pd.DataFrame()  # Always return a DataFrame
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'description']]

# Page styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #000000, #1e1e1e);
            color: white;
        }
        .main {
            background-color: #1e1e1e;
        }
        h1, h4, p {
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# UI Title
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #e50914;">🎬 Netflix Recommender</h1>
        <p style="font-size:18px; color: #f5f5f5;">Type your favorite Netflix title and get 5 amazing suggestions 🍿</p>
    </div>
""", unsafe_allow_html=True)

# Input box
movie_title = st.text_input("🔎 Enter a Netflix title", placeholder="e.g., Stranger Things")

# Display recommendations
if movie_title:
    results = recommend(movie_title)
    if not results.empty:
        st.subheader("✨ Top 5 Recommendations:")
        for _, row in results.iterrows():
            st.markdown(f"""
                <div style="background-color:#262626; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h4 style="color:#e50914;">🎥 {row['title']}</h4>
                    <p style="color:#dcdcdc;">{row['description']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("🚫 Title not found. Please try another one.")
