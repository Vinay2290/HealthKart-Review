import joblib


# Load model once when file is imported
model = joblib.load("model/sentiment_model.pkl")


def predict(text: str):
    """Return sentiment and recommendation."""

    sent = model["sentiment"].predict([text])[0]
    rec = model["recommend"].predict([text])[0]

    return {
        "sentiment": "Positive" if sent == 1 else "Negative",
        "recommendation": "Yes" if rec == 1 else "No",
    }
