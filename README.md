# Twitter Sentiment Analysis Platform

A complete, end-to-end sentiment analysis system utilizing a fine-tuned BERT model. This project demonstrates a full MLOps pipeline including data preprocessing, transfer learning with Hugging Face, performance evaluation, and containerized deployment via Docker and FastAPI.

---

## System Architecture

The project is structured as a scalable microservice architecture:

1. **Data & Training Pipeline:** Local Python scripts process the IMDB dataset and fine-tune a Transformer model.
2. **Inference API:** A FastAPI backend serves the fine-tuned model. It is completely stateless, allowing for easy horizontal scaling under high load.
3. **User Interface:** A Streamlit frontend allows users to interactively test the model.
4. **Batch Processing:** A standalone CLI tool processes large datasets using efficient matrix batching.

---

## Prerequisites

- **Python 3.9+** (For local training and batch prediction)
- **Docker & Docker Compose** (For deployment)
- **Git** (For version control)

---

## Step 1: Local Setup & Model Training

> **Important:** The Docker containers do **not** train the model.  
> You must generate the model artifacts locally before building the Docker images.  
> The `model_output/` directory is git-ignored due to file size constraints.

### 1. Install Local Dependencies

```bash
pip install -r requirements.api.txt
pip install pandas datasets scikit-learn tqdm
```

### 2. Preprocess the Data

Downloads the dataset and cleans the text (removes HTML tags, URLs, and normalizes whitespace).

```bash
python scripts/preprocess.py
```

**Output:**

```
data/processed/train.csv
data/processed/test.csv
```

---

### 3. Fine-Tune the Model

Trains the BERT model and outputs evaluation metrics.

```bash
python scripts/train.py
```

**Outputs:**

- Trained model artifacts in `model_output/`
- Evaluation metrics in `results/metrics.json`
- Hyperparameter logs in `results/run_summary.json`

---

## Step 2: Containerized Deployment

Once `model_output/` exists, you can deploy the services.

Create a `.env` file:

```bash
cp .env.example .env
```

Then start the system:

```bash
docker-compose up --build -d
```

This launches two services:

- **API Backend:** http://localhost:8000
- **Streamlit UI:** http://localhost:8501

Check container health:

```bash
docker ps
```

Both services should report **healthy**.

---

## Step 3: API Usage Guide

### 1. Health Check

Verifies if the API is running and the model has finished loading.

**Endpoint:**

```
GET /health
```

**Example:**

```bash
curl http://localhost:8000/health
```

**Response (ready):**

```json
{ "status": "ok" }
```

**Response (loading):**

HTTP 503

```json
{ "status": "loading" }
```

---

### 2. Predict Sentiment

Analyzes the emotional tone of a single string of text.

**Endpoint:**

```
POST /predict
```

**Headers:**

```
Content-Type: application/json
```

**Payload:**

```json
{
  "text": "I absolutely loved the cinematography in this film!"
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely loved the cinematography in this film!"}'
```

**Response:**

```json
{
  "sentiment": "positive",
  "confidence": 0.985
}
```

---

## Step 4: Batch Prediction (CLI Utility)

For processing large CSV files efficiently without making thousands of HTTP requests, use the batch prediction script.  
This utilizes tensor batching to maximize GPU/CPU throughput.

**Usage:**

```bash
python scripts/batch_predict.py \
  --input_file data/unseen_data.csv \
  --output_file results/predictions.csv \
  --batch_size 32
```

**Requirement:**  
The input CSV must contain a `text` column.

**Output:**  
The script appends:

- `predicted_sentiment`
- `confidence`

to the CSV.

---

## Model Choice & Rationale

**Base Model:** `bert-base-uncased`

**Rationale:**  
BERT’s bidirectional attention captures deep contextual nuances (e.g., sarcasm, negation) better than traditional models.  
`bert-base-uncased` was selected to balance:

- High accuracy
- Reasonable memory usage
- Fast inference inside Docker

---

## Project Structure

```
├── data/
│   ├── processed/
│   │   ├── test.csv
│   │   └── train.csv
├── model_output/        # Generated locally (not committed)
├── results/
├── scripts/
│   ├── batch_predict.py
│   ├── preprocess.py
│   └── train.py
├── src/
│   ├── api.py
│   └── ui.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── README.md
├── requirements.api.txt
└── requirements.ui.txt
```

---

## Scalability Note

The FastAPI backend is **stateless**.  
This allows horizontal scaling by running multiple API containers behind a load balancer without modifying application code.

---
