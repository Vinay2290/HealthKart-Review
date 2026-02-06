import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Simple text cleaning so ML model works better."""

    text = str(text).lower()                 # convert to lowercase
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(f"[{string.punctuation}]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)      # remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces

    return text


def load_and_prepare(path: str):
    """Loads CSV and prepares training data."""

    # Load CSV file
    df = pd.read_csv(path)

    # Keep only useful columns and drop missing rows
    df = df[["reviews.text", "reviews.rating", "reviews.doRecommend"]].dropna()

    # Create sentiment label (industry standard rule)
    df["sentiment"] = df["reviews.rating"].apply(lambda x: 1 if x >= 4 else 0)

    # Recommendation label already yes/no → convert to int
    df["recommend"] = df["reviews.doRecommend"].astype(int)

    # Clean the review text
    df["clean_text"] = df["reviews.text"].apply(clean_text)

    # Features and targets
    X = df["clean_text"]
    y_sent = df["sentiment"]
    y_rec = df["recommend"]

    # Train‑test split
    return train_test_split(X, y_sent, y_rec, test_size=0.2, random_state=42)