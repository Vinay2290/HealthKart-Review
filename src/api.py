from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict



app = FastAPI(title="HealthKart Review NLP API")


class Review(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "API is running successfully"}


@app.post("/predict")
def get_prediction(review: Review):
    return predict(review.text)
