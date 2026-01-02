# =====================================
# SOCIAL MEDIA BIG DATA ANALYZER
# TF-IDF TOP 2000 WORDS + WORDCLOUD
# STREAMLIT + GITHUB READY
# =====================================

import streamlit as st
import feedparser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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

scores = tfidf_matrix.sum(axis=0).A1
words = vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame({
    "word": words,
    "tfidf": scores
})

tfidf_df = tfidf_df.sort_values(by="tfidf", ascending=False)

# -------------------------------------
# STEP 3: DISPLAY TOP 2000 WORDS
# -------------------------------------
st.subheader("Top 2000 Words using TF-IDF")
st.dataframe(tfidf_df.head(2000))

# -------------------------------------
# STEP 4: SEARCH TF-IDF VALUE FOR A WORD
# -------------------------------------
search_word = st.text_input("Enter a word to get its TF-IDF value:")

if search_word:
    w = search_word.lower()
    if w in tfidf_df["word"].values:
        value = tfidf_df.loc[
            tfidf_df["word"] == w, "tfidf"
        ].values[0]
        st.success(value)
    else:
        st.error("Word not found")

# -------------------------------------
# STEP 5: WORDCLOUD VISUALIZATION
# -------------------------------------
st.subheader("WordCloud of Trending Topics")

combined_text = " ".join(df["title"])

wordcloud = WordCloud(
    width=900,
    height=450,
    background_color="white"
).generate(combined_text)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
