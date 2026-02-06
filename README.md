# HealthKart NLP System Review

This project is an end-to-end pipeline for natural language processing that forecasts:

The tone of a product review
- If the user suggests the product ##Tech Stack
Python
Scikit-learn
FastAPI
Docker ## Use Docker to run

Build: docker build -t healthkart-nlp.

Run: docker run -p 8000:8000 healthkart-nlp

Open the API documentation at http://localhost:8000/docs## API Endpoint

POST/predict

Enter: { "text": "Amazing product" }

Output: { "sentiment": "Positive", "recommendation": "Yes" }
