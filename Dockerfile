FROM python:3.10-slim


WORKDIR /app


COPY . .


RUN pip install --no-cache-dir -r requirements.txt


# Train model during Docker build
RUN python src/train.py


CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
