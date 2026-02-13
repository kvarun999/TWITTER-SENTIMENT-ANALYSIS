import os
import torch
import torch.nn.functional as F
import re
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#loading .env
load_dotenv()

# Input Schema
class SentimentRequest(BaseModel):
    text: str

def clean_text(text):
    text = text.lower()
    
    # Replace HTML breaks specifically with a space to avoid mashing words
    text = re.sub(r"<br\s*/?>", " ", text)
    
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # Remove special characters (keep letters & numbers)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

# 1. Startup Event: Load the model here
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device

    # REQUIREMENT: Use environment variables for configuration
    model_path = os.getenv("MODEL_PATH", "./model_output")
    print(f"Loading model from: {model_path} on {device}")

    try:
        # Load artifacts
        app.state.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        app.state.tokenizer = AutoTokenizer.from_pretrained(model_path)
        app.state.model.to(device)
        app.state.model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        app.state.model = None
        app.state.tokenizer = None
        print(f"❌ Failed to load model: {e}")
        # We don't raise here so the container stays alive, 
        # but the health check will reflect the status.
    
    yield
    
    # Cleanup
    app.state.model = None
    app.state.tokenizer = None

# Initialize FastAPI with the lifespan manager
app = FastAPI(lifespan=lifespan)

# 2. Health Endpoint
@app.get("/health")
def health_check():
    if app.state.model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "loading"}
        )
    
    return {"status": "ok"}

# 3. Predict Endpoint
@app.post("/predict")
def predict(request: SentimentRequest):
    if app.state.model is None or app.state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    cleaned_text = clean_text(request.text)
    tokenizer = app.state.tokenizer
    model = app.state.model
    device = app.state.device

    try:
        # Tokenize
        inputs = tokenizer(
            cleaned_text, 
            padding="max_length", 
            truncation=True, 
            max_length=128,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
            # Get the predicted class (0 or 1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()

        label_map = {0: "negative", 1: "positive"}

        print(next(model.parameters()).device)

        return {
            "sentiment": label_map[predicted_class],
            "confidence": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))