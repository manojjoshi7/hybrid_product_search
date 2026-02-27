import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("data/cleaned_data/product_info_data_cleaned.csv")

search_columns = [
    "productId","product_name","product_brand",
    "min_quantity","min_price","vendor_name",
    "descriptions","keywords","sizes",
    "colors","categories"
]

for col in search_columns:
    df[col] = df[col].fillna("").astype(str)

# -----------------------------
# Weighted Text
# -----------------------------
df["weighted_text"] = (
    df["product_name"] * 3 + " " +
    df["product_brand"] * 2 + " " +
    df["keywords"] * 2 + " " +
    df["categories"] * 2 + " " +
    df["descriptions"] + " " +
    df["sizes"] + " " +
    df["colors"] + " " +
    df["vendor_name"] + " " +
    df["productId"]
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["weighted_text"] = df["weighted_text"].apply(clean_text)

# -----------------------------
# TF-IDF (with N-grams)
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    max_features=150000
)

tfidf_matrix = vectorizer.fit_transform(df["weighted_text"])

# -----------------------------
# Semantic Embeddings
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["weighted_text"], show_progress_bar=True)

# -----------------------------
# Save Everything
# -----------------------------
joblib.dump(df, "models/products.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(tfidf_matrix, "models/tfidf.pkl")
joblib.dump(embeddings, "models/embeddings.pkl")

print("✅ Training Complete. Models Saved.")