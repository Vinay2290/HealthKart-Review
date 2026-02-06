import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocess import load_and_prepare


# Load and split the dataset
X_train, X_test, y_sent_train, y_sent_test, y_rec_train, y_rec_test = load_and_prepare(
    "data/reviews.csv"
)


# Pipeline for sentiment prediction
sentiment_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=200))
])


# Pipeline for recommendation prediction
recommend_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=200))
])


# Train models
sentiment_pipeline.fit(X_train, y_sent_train)
recommend_pipeline.fit(X_train, y_rec_train)


# Save both models together
joblib.dump(
    {
        "sentiment": sentiment_pipeline,
        "recommend": recommend_pipeline,
    },
    "model/sentiment_model.pkl",
)


print("✅ Model training complete and saved in model/sentiment_model.pkl")

