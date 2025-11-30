# sentiment_and_theme.py

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline

# Optional: spaCy for advanced NLP
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False


# -------------------------------------------------------------------
# 1. TEXT CLEANING
# -------------------------------------------------------------------
def clean_text(text):
    """Clean review text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9,.!? ]", "", text)
    return text.strip().lower()


# -------------------------------------------------------------------
# 2. SENTIMENT ANALYSIS
# -------------------------------------------------------------------
def get_sentiment_model(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """Load HuggingFace sentiment pipeline."""
    return pipeline("sentiment-analysis", model=model_name)


def analyze_sentiment(text, sentiment_model=None):
    """Return sentiment label and score using DistilBERT."""
    text = clean_text(text)
    if not text:
        return {"label": "neutral", "score": 0.0}

    if sentiment_model is None:
        sentiment_model = get_sentiment_model()

    # DistilBERT only allows 512 tokens
    result = sentiment_model(text[:512])[0]
    label = result["label"].lower()
    score = result["score"]

    # Map to neutral if score is low
    if label in ["positive", "negative"] and score < 0.6:
        label = "neutral"

    return {"label": label, "score": score}


# -------------------------------------------------------------------
# 3. THEME EXTRACTION
# -------------------------------------------------------------------
def extract_keywords_tfidf(corpus, top_n=10):
    """Extract top TF-IDF keywords from a list of texts."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(corpus)
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(X.toarray().sum(axis=0))[::-1]
    top_keywords = feature_array[tfidf_sorting][:top_n]
    return top_keywords.tolist()


def cluster_themes(corpus, n_clusters=5):
    """Cluster reviews into n_clusters using TF-IDF + KMeans."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(corpus)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.labels_
    return clusters


def assign_themes(df, review_column="review", n_clusters=5):
    """Assign themes to reviews and return df with theme label."""
    df = df.copy()
    df["clean_review"] = df[review_column].apply(clean_text)

    # Extract clusters as themes
    df["theme_cluster"] = cluster_themes(df["clean_review"], n_clusters=n_clusters)

    # Optional: map clusters to human-readable names (manual)
    cluster_map = {
        0: "Account Access Issues",
        1: "Transaction Performance",
        2: "User Interface & Experience",
        3: "Customer Support",
        4: "Feature Requests"
    }

    df["identified_theme"] = df["theme_cluster"].map(lambda x: cluster_map.get(x, f"Theme_{x}"))
    return df


# -------------------------------------------------------------------
# 4. PIPELINE
# -------------------------------------------------------------------
def run_pipeline(df, review_column="review", output_file="task2_results.csv"):
    """
    Full pipeline:
    1. Clean text
    2. Compute sentiment
    3. Extract themes
    4. Aggregate by bank & rating
    5. Save CSV
    """
    df = df.copy()

    # Step 1: Sentiment
    sentiment_model = get_sentiment_model()
    sentiment_results = df[review_column].apply(lambda x: analyze_sentiment(x, sentiment_model))
    df["sentiment_label"] = sentiment_results.apply(lambda x: x["label"])
    df["sentiment_score"] = sentiment_results.apply(lambda x: x["score"])

    # Step 2: Thematic Analysis
    df = assign_themes(df, review_column)

    # Step 3: Aggregation by bank and rating
    agg = df.groupby(["bank", "rating"])["sentiment_score"].mean().reset_index()
    agg = agg.rename(columns={"sentiment_score": "mean_sentiment_score"})

    # Step 4: Save results
    df.to_csv(output_file, index=False)
    print(f"Pipeline finished. Results saved to {output_file}")
    return df, agg


# -------------------------------------------------------------------
# 5. EXAMPLE USAGE (uncomment for script run)
# -------------------------------------------------------------------
# if __name__ == "__main__":
#     df = pd.read_csv("../Data/bank_reviews_clean.csv")
#     enriched_df, aggregation = run_pipeline(df)
#     print(aggregation.head())
