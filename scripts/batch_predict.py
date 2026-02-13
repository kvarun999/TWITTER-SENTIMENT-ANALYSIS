import argparse
import os
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm  # This gives us a nice progress bar

def parse_args():
    parser = argparse.ArgumentParser(description="Run batch sentiment analysis on a CSV file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file (must have a 'text' column).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the results CSV.")
    parser.add_argument("--model_path", type=str, default="./model_output", help="Path to the trained model directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of examples to process at once.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Validation
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}. Did you train the model?")

    # 2. Load Resources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Loading model from {args.model_path} to {device}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        model.to(device)
        model.eval() # Set to evaluation mode (turns off dropout, etc.)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Load Data
    print(f"📂 Reading data from {args.input_file}...")
    try:
        df = pd.read_csv(args.input_file)
        if "text" not in df.columns:
            raise ValueError("Input CSV must contain a 'text' column.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 4. Batch Prediction Loop
    print(f"🔮 Predicting on {len(df)} examples (Batch size: {args.batch_size})...")
    
    predictions = []
    confidences = []
    
    # Iterate through the data in chunks
    for i in tqdm(range(0, len(df), args.batch_size), desc="Processing"):
        batch_texts = df["text"].iloc[i : i + args.batch_size].tolist()
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        # Move inputs to the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
            # Get class ID (0 or 1) and confidence score
            batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
            batch_conf = torch.max(probs, dim=1).values.cpu().numpy()
        
        # Map IDs to Labels
        label_map = {0: "negative", 1: "positive"}
        predictions.extend([label_map[p] for p in batch_preds])
        confidences.extend(batch_conf)

    # 5. Save Results
    df["predicted_sentiment"] = predictions
    df["confidence"] = confidences
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"✅ Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()