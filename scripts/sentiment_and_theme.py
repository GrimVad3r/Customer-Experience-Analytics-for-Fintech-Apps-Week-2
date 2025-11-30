"""
sentiment_and_theme.py
Run from project root: python task-2/sentiment_and_theme.py --input Data/cleaned_reviews.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import joblib

# ---- optional models ----
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# VADER fallback
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    VADER = SentimentIntensityAnalyzer()
except Exception:
    VADER = None

# ------ Utility / preprocessing ------
import re
def simple_preprocess(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"http\S+", "", text)                  # remove urls
    text = re.sub(r"@\w+", "", text)                     # remove @
    text = re.sub(r"[^A-Za-z0-9\s\-\']", " ", text)      # keep basic chars
    text = re.sub(r"\s+", " ", text)
    return text.lower()

# ----- Sentiment using HF pipeline (batched) -----
def hf_sentiment_batch(texts, model_name="distilbert-base-uncased-finetuned-sst-2-english", batch_size=32):
    if not HF_AVAILABLE:
        raise RuntimeError("transformers not available")
    classifier = pipeline('sentiment-analysis', model=model_name, device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else -1)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        preds = classifier(batch, truncation=True)
        # preds are like {'label':'POSITIVE','score':0.999}
        for p in preds:
            label = p['label'].lower()
            score = float(p['score'])
            # map to 3-class: positive, negative, neutral (we'll treat low-confidence as neutral)
            if score < 0.6:
                lab = 'neutral'
            else:
                lab = 'positive' if 'pos' in label else 'negative'
            results.append({'label': lab, 'score': score})
    return results

# ----- VADER fallback -----
def vader_sentiment(texts):
    if VADER is None:
        raise RuntimeError("VADER not available")
    out = []
    for t in texts:
        s = VADER.polarity_scores(t)
        # compound in [-1,1]
        comp = s['compound']
        if comp >= 0.05:
            lab = 'positive'
        elif comp <= -0.05:
            lab = 'negative'
        else:
            lab = 'neutral'
        out.append({'label': lab, 'score': comp})
    return out

# ----- TF-IDF keywords per bank -----
def top_tfidf_by_bank(df, text_col='clean_text', bank_col='bank', top_n=30, ngram_range=(1,2)):
    banks = {}
    for bank, sub in df.groupby(bank_col):
        docs = sub[text_col].astype(str).tolist()
        if len(docs) == 0:
            banks[bank] = []
            continue
        vec = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english', ngram_range=ngram_range, max_features=2000)
        X = vec.fit_transform(docs)
        # average tfidf score per term
        mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        top_idx = mean_tfidf.argsort()[::-1][:top_n]
        banks[bank] = list(zip(terms[top_idx], mean_tfidf[top_idx]))
    return banks

# ----- Cluster reviews into themes per bank -----
def cluster_themes_per_bank(df, text_col='clean_text', bank_col='bank', n_themes=4, ngram_range=(1,2)):
    theme_assignments = {}
    vec = TfidfVectorizer(max_df=0.95, min_df=3, stop_words='english', ngram_range=ngram_range, max_features=4000)
    for bank, sub in df.groupby(bank_col):
        docs = sub[text_col].astype(str).tolist()
        idx = sub.index.to_list()
        if len(docs) < max(10, n_themes):
            # not enough docs, mark as 'other'
            for i in idx:
                theme_assignments[i] = ['other']
            continue
        X = vec.fit_transform(docs)
        # normalize then kmeans
        Xn = normalize(X)
        n_clusters = min(n_themes, Xn.shape[0]//2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(Xn)
        labels = kmeans.labels_
        # find top terms per cluster
        terms = np.array(vec.get_feature_names_out())
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        cluster_top_terms = {}
        for i in range(n_clusters):
            top_terms = terms[order_centroids[i, :20]]
            cluster_top_terms[i] = top_terms.tolist()[:10]
        # assign cluster labels back to original indices
        for loc, lbl in zip(idx, labels):
            theme_assignments[loc] = cluster_top_terms[lbl]
    return theme_assignments

# ---- Main pipeline ----
def main(input_file, output_file, use_hf=True):
    df = pd.read_csv(input_file)
    required_cols = ['review_id', 'bank', 'rating', 'review_text']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    # preprocess
    df['clean_text'] = df['review_text'].fillna('').apply(simple_preprocess)

    # sentiment
    texts = df['clean_text'].tolist()
    sentiment_results = None
    if use_hf and HF_AVAILABLE:
        try:
            print("Running HuggingFace distilbert sentiment (batched)...")
            sentiment_results = hf_sentiment_batch(texts)
        except Exception as e:
            print("HF pipeline failed, falling back to VADER:", e)
            sentiment_results = vader_sentiment(texts) if VADER else [{'label':'neutral','score':0.0}]*len(texts)
    else:
        if VADER:
            print("Running VADER sentiment...")
            sentiment_results = vader_sentiment(texts)
        else:
            print("No sentiment model available, filling neutral")
            sentiment_results = [{'label':'neutral','score':0.0}]*len(texts)

    df['sentiment_label'] = [r['label'] for r in sentiment_results]
    df['sentiment_score'] = [r['score'] for r in sentiment_results]

    # aggregate checks
    agg = df.groupby(['bank','rating'])['sentiment_score'].mean().reset_index().rename(columns={'sentiment_score':'mean_sentiment_score'})
    agg.to_csv('outputs/sentiment_by_bank_rating.csv', index=False)

    # keywords by bank
    print("Extracting TF-IDF keywords per bank...")
    banks_keywords = top_tfidf_by_bank(df, top_n=30)
    # save a CSV friendly version
    rows = []
    for b,klist in banks_keywords.items():
        for term,score in klist:
            rows.append({'bank':b,'term':term,'tfidf_score':score})
    pd.DataFrame(rows).to_csv('outputs/top_keywords_per_bank.csv', index=False)

    # cluster into themes
    print("Clustering reviews per bank to extract candidate themes...")
    theme_assign = cluster_themes_per_bank(df, n_themes=4)
    # convert to human-friendly single theme string: join top 3 terms
    df['identified_theme_terms'] = df.index.map(lambda i: "; ".join(theme_assign.get(i, ['other'])[:3]))
    # For automated theme label, you may map groups like:
    # e.g. if 'login' or 'password' in theme terms => 'Account Access Issues'
    def map_terms_to_theme(termstr):
        t = termstr.lower()
        if any(x in t for x in ['login','password','sign in','otp','2fa','unable to login']):
            return 'Account Access'
        if any(x in t for x in ['slow','lag','timeout','performance','speed','delay','slow transfer']):
            return 'Performance & Speed'
        if any(x in t for x in ['ui','ux','interface','design','layout','easy to use','navigation']):
            return 'UI / UX'
        if any(x in t for x in ['support','customer service','agent','response','helpdesk','call center']):
            return 'Customer Support'
        if any(x in t for x in ['crash','bug','error','exception','failed']):
            return 'Reliability / Crashes'
        return 'Other'
    df['identified_theme'] = df['identified_theme_terms'].apply(map_terms_to_theme)

    # Save final results
    os.makedirs('outputs', exist_ok=True)
    df_out_cols = ['review_id','bank','rating','review_text','clean_text','sentiment_label','sentiment_score','identified_theme','identified_theme_terms']
    df[df_out_cols].to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    print("Saved top keywords to outputs/top_keywords_per_bank.csv and aggregated sentiment to outputs/sentiment_by_bank_rating.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='Data/cleaned_reviews.csv')
    parser.add_argument('--output', default='outputs/task2_results.csv')
    parser.add_argument('--no-hf', action='store_true', help='disable HF model, use VADER if available')
    args = parser.parse_args()
    use_hf = not args.no_hf
    main(args.input, args.output, use_hf=use_hf)
