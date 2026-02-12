import os
import re
import pandas as pd
from datasets import load_dataset

# Ensure the required output directory exists
os.makedirs("data/processed", exist_ok=True)

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

def main():
    print("📥 Downloading and loading IMDB dataset from Hugging Face...")
    # This automatically handles downloading and caching the raw data
    dataset = load_dataset("imdb")

    # Convert directly to Pandas DataFrames for vectorized operations
    print("🔄 Converting to Pandas DataFrames...")
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()

    print("🧹 Applying text cleaning function...")
    # Apply the cleaning function to the entire text column
    df_train["text"] = df_train["text"].apply(clean_text)
    df_test["text"] = df_test["text"].apply(clean_text)

    print("💾 Saving processed data strictly to CSV format...")
    # Output strictly as CSV with only 'text' and 'label' columns to meet requirements
    df_train[["text", "label"]].to_csv("data/processed/train.csv", index=False)
    df_test[["text", "label"]].to_csv("data/processed/test.csv", index=False)

    print("✅ Preprocessing complete! Files saved as train.csv and test.csv in data/processed/")

if __name__ == "__main__":
    main()