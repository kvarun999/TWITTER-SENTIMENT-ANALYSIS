import os
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("model_output", exist_ok=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Take the argmax to get the predicted class IDs
    predictions = np.argmax(logits, axis=1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    
    # Return keys exactly as the Trainer expects them to be logged
    # We will format them for the final JSON requirement later
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def main():
    checkpoint = "bert-base-uncased"
    print(f"🚀 Initializing model: {checkpoint}")

    # 1. Load Data directly using Hugging Face datasets (Cleaner than Pandas -> Dataset)
    data_files = {
        "train": "data/processed/train.csv", 
        "test": "data/processed/test.csv"
    }
    dataset = load_dataset("csv", data_files=data_files)

    # 2. Tokenization
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )

    print("🔄 Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 3. Rename 'label' to 'labels' for the model and set format
    # Note: load_dataset("csv") keeps the columns as they are in the CSV
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 4. Model Initialization
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",      # Updated from evaluation_strategy (deprecated in newer versions)
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=2,          # Good practice: don't fill disk with checkpoints
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    # 7. Train
    print("🏋️ Starting training...")
    trainer.train()

    # 8. Save Model Artifacts
    print("💾 Saving model artifacts...")
    save_directory = "model_output"
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)

    # 9. Evaluate & Save Metrics (STRICT SCHEMA COMPLIANCE)
    print("📊 Evaluating and generating reports...")
    eval_result = trainer.evaluate()

    # The Trainer returns keys like 'eval_accuracy'. We must strip 'eval_' 
    # and strictly match the requested JSON schema.
    final_metrics = {
        "accuracy": eval_result["eval_accuracy"],
        "precision": eval_result["eval_precision"],
        "recall": eval_result["eval_recall"],
        "f1_score": eval_result["eval_f1_score"]
    }

    with open("results/metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    # 10. Save Run Summary (STRICT SCHEMA COMPLIANCE)
    run_summary = {
        "hyperparameters": {
            "model_name": checkpoint,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "num_epochs": training_args.num_train_epochs
        },
        "final_metrics": final_metrics
    }

    with open("results/run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=4)
    
    print("✅ Training complete. Artifacts and metrics saved.")

if __name__ == "__main__":
    main()