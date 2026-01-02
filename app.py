# =====================================
# SOCIAL MEDIA BIG DATA ANALYZER
# STREAMLIT APP (GITHUB READY)
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
st.set_page_config(page_title="Social Media Big Data Analyzer", layout="centered")
st.title("📊 Social Media Big Data Analyzer")

# -------------------------------------
# STEP 1: FETCH TRENDING DATA (REDDIT RSS)
# -------------------------------------
@st.cache_data
def fetch_reddit_data():
    url = "https://www.reddit.com/r/popular/.rss"
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries]

texts = fetch_reddit_data()
df = pd.DataFrame(texts, columns=["text"])

st.subheader("Trending Data (Sample)")
st.dataframe(df.head())

# -------------------------------------
# STEP 2: TF-IDF VECTORIZATION
# -------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=30
)

tfidf_matrix = vectorizer.fit_transform(df["text"])

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

# -------------------------------------
# STEP 3: TF-IDF FREQUENCY (ONLY SCORES)
# -------------------------------------
scores_only = tfidf_df.sum(axis=0).sort_values(ascending=False)

st.subheader("TF-IDF Scores (Numeric Only)")
st.write(scores_only.values)

# -------------------------------------
# STEP 4: USER INPUT FOR TOPIC NAME
# -------------------------------------
user_topic = st.text_input("Enter the suggested topic name:")

final_topic = user_topic.strip().title() if user_topic else "Trending Topic"
st.success(f"Final Topic Name: {final_topic}")

# -------------------------------------
# STEP 5: RULE-BASED TREND CATEGORY
# -------------------------------------
def suggest_category(words):
    ai = {"ai", "chatgpt", "openai", "machine", "artificial"}
    finance = {"bitcoin", "crypto", "stock", "market", "economy"}
    sports = {"cricket", "football", "match", "olympics"}
    politics = {"election", "government", "minister", "policy"}
    tech = {"apple", "google", "microsoft", "tesla"}

    for w in words:
        if w in ai:
            return "Artificial Intelligence"
        if w in finance:
            return "Finance & Crypto"
        if w in sports:
            return "Sports"
        if w in politics:
            return "Politics"
        if w in tech:
            return "Technology"

    return "General Trending Topic"

trend_category = suggest_category(scores_only.index[:10])
st.info(f"Detected Trend Category: {trend_category}")

# -------------------------------------
# STEP 6: WORDCLOUD VISUALIZATION
# -------------------------------------
combined_text = " ".join(df["text"])

wordcloud = WordCloud(
    width=900,
    height=450,
    background_color="white"
).generate(combined_text)

st.subheader("WordCloud Visualization")
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
ax.set_title(f"Topic: {final_topic} ({trend_category})")
st.pyplot(fig)

# -------------------------------------
# STEP 7: TF-IDF SCORE OF A PARTICULAR WORD
# -------------------------------------
search_word = st.text_input("Enter a word to get its TF-IDF score:")

if search_word:
    word = search_word.lower()
    if word in scores_only.index:
        st.success(scores_only[word])
    else:
        st.error("Word not found in trending data")
