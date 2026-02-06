# HealthKart NLP System Review

Review of the HealthKart NLP System

A complete, Dockerized pipeline for Natural Language Processing (NLP) that makes the following predictions:

Product review sentiment (either positive or negative)

Probability of recommendation (Yes/No)

This project, which was created as a HealthKart Data Science Intern assignment, showcases the full machine learning lifecycle, from model training and preprocessing to containerisation and API deployment.

🚀 Features
Preparing and cleaning text for data review
Feature extraction using TF-IDF

Models of logistic regression for:
Classification of sentiment
Prediction of recommendations
Real-time inference using the FastAPI REST service
Completely replicable Dockerized operation
Project structure that is clear and production-style

Tech Stack:
Python
Scikit-learn and Pandas
FastAPI
The Uvicorn
Docker

Build: docker build -t healthkart-nlp.

Run: docker run -p 8000:8000 healthkart-nlp

Open the API documentation at http://localhost:8000/docs## API Endpoint

POST/predict

Enter: { "text": "Amazing product" }

Output: { "sentiment": "Positive", "recommendation": "Yes" }
