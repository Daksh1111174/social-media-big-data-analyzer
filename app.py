# =====================================
# SOCIAL MEDIA BIG DATA ANALYZER
# TF-IDF TOP 2000 WORDS
# STREAMLIT + GITHUB READY
# =====================================

import streamlit as st
import feedparser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------
# APP CONFIG
# -------------------------------------
st.set_page_config(
    page_title="Social Media Big Data Analyzer",
    layout="centered"
)

st.title("📊 Social Media Big Data Analyzer")
st.caption("Trending Topic Analysis using Reddit + TF-IDF")

# -------------------------------------
# STEP 1: FETCH REDDIT TRENDING DATA
# -------------------------------------
@st.cache_data
def fetch_reddit_data():
    url = "https://www.reddit.com/r/popular/.rss"
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries]

titles = fetch_reddit_data()

df = pd.DataFrame(titles, columns=["title"])

st.subheader("Trending Reddit Titles (Sample)")
st.dataframe(df.head())

# -------------------------------------
# STEP 2: TF-IDF (YOUR EXACT LOGIC)
# -------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["title"])

scores = tfidf_matrix.sum(axis=0).A1_
