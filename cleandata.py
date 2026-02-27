import pandas as pd
import os
import numpy as np

# -------------------------------
# Create folders if missing
# -------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("data/cleaned_data", exist_ok=True)

# -------------------------------
# Pandas display options (optional)
# -------------------------------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# -------------------------------
# Load Data
# Treat '\N' as NaN
# -------------------------------
df = pd.read_csv(
    "data/p1.csv",
    low_memory=False,
    na_values=["\\N"]
)

print("Original shape:", df.shape)

# -------------------------------
# Clean product_name & productId
# -------------------------------
cols_to_clean = ["product_name", "productId"]

for col in cols_to_clean:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

df = df.dropna(subset=cols_to_clean).reset_index(drop=True)

# -------------------------------
# Clean Numeric Columns
# -------------------------------
df["min_quantity"] = pd.to_numeric(df["min_quantity"], errors="coerce")
df["min_price"] = pd.to_numeric(df["min_price"], errors="coerce")

# Optional: remove invalid price/qty
df = df[(df["min_price"] > 0) & (df["min_quantity"] > 0)]

# -------------------------------
# Clean Text Columns (UPDATED)
# -------------------------------
TEXT_COLUMNS = [
    "productId",      # ✅ Added here
    "product_name",
    "vendor_name",
    "product_brand",
    "descriptions",
    "keywords",
    "sizes",
    "colors",
    "categories"
]

for col in TEXT_COLUMNS:
    if col not in df.columns:
        df[col] = ""
    df[col] = (
        df[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

# -------------------------------
# Create Combined Text (For Search / TF-IDF)
# -------------------------------
df["combined_text"] = df[TEXT_COLUMNS].agg(" ".join, axis=1)

# Remove extra spaces
df["combined_text"] = df["combined_text"].str.replace(r"\s+", " ", regex=True)

print("Cleaned shape:", df.shape)

# -------------------------------
# Save Cleaned Data
# -------------------------------
output_path = "data/cleaned_data/product_info_data_cleaned.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"\n✅ Cleaned data saved to: {output_path}")