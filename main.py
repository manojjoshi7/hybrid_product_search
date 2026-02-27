import joblib
import numpy as np
import re
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Load Models
# -----------------------------
df = joblib.load("models/products.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
tfidf_matrix = joblib.load("models/tfidf.pkl")
embeddings = joblib.load("models/embeddings.pkl")

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Extract Vendors from Query
# -----------------------------
def extract_vendors(query):
    vendor_list = df["vendor_name"].unique()
    found_vendors = []

    for vendor in vendor_list:
        if vendor and vendor.lower() in query.lower():
            found_vendors.append(vendor)

    return found_vendors

# -----------------------------
# Extract number like "five"
# -----------------------------
def extract_number(query):
    number_map = {
        "one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9,"ten":10
    }

    for word, num in number_map.items():
        if word in query.lower():
            return num

    numbers = re.findall(r'\d+', query)
    if numbers:
        return int(numbers[0])

    return 5  # default

@app.get("/", response_class=HTMLResponse)
def search_page(request: Request, q: str = ""):

    results = []
    vendor_groups = {}
    
    if q:

        # 🔥 Check if vendor-based query
        #vendors = extract_vendors(q)
        #limit = extract_number(q)
        vendors=[]
        limit=2
        if vendors:
            # Vendor specific result
            for vendor in vendors:
                vendor_df = df[df["vendor_name"].str.lower() == vendor.lower()]
                vendor_groups[vendor] = vendor_df.head(limit).to_dict(orient="records")

        else:
            
            # Normal hybrid semantic search
            query = clean_text(q)

            query_vec = vectorizer.transform([query])
            tfidf_score = cosine_similarity(query_vec, tfidf_matrix).flatten()

            query_embed = semantic_model.encode([query])
            semantic_score = cosine_similarity(query_embed, embeddings).flatten()

            final_score = (0.6 * tfidf_score) + (0.4 * semantic_score)

            df["score"] = final_score
            filtered = df.sort_values("score", ascending=False)

            results = filtered.head(20).to_dict(orient="records")
           
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results,
        "vendor_groups": vendor_groups,
        "query": q
    })