import os
import random
import pandas as pd
import torch

from datasets import Dataset, DatasetDict 
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# 1) HYPERPARAMETERS & PATHS
# ──────────────────────────────────────────────────────────────────────────────

# 1a) original data lives:
ORIG_TRAIN_CSV = "train.csv"       
ORIG_DEV_CSV   = "dev.csv"         


NEW_PROMPTS_CSV = "greece_edited.csv"

PREV_CHECKPOINT = "./past_ref_classifier/updated_model_2"

# Where to save the newly trained model:
OUTPUT_DIR = "./past_ref_classifier/updated_model_3"

# 1e) Model / Tokenizer name (same as before)
MODEL_NAME = "distilbert-base-uncased"

# 1f) Training hyperparameters
NUM_NEW_EPOCHS        = 3
BATCH_SIZE            = 16
EVAL_BATCH_SIZE       = 32
LEARNING_RATE         = 2e-5
EVAL_STRATEGY         = "epoch"
SAVE_STRATEGY         = "epoch"
LOAD_BEST_AT_END      = True
METRIC_FOR_BEST_MODEL = "accuracy"
SAVE_TOTAL_LIMIT      = 2


def load_csv_dataset(csv_path):
    """
    Reads a CSV with columns [text, label] into a Hugging Face Dataset.
    """
    df = pd.read_csv(csv_path)
    # Drop any rows with missing text or label
    df = df.dropna(subset=["text", "label"])
    return Dataset.from_pandas(df)

def prepare_merged_datasets():
    """
    1. Loads your original train/dev (if available) and new prompts.
    2. Merges them.
    3. Re-splits into new train/validation/test (80/10/10).
    """
    # Load original data if it exists ---
    datasets = []
    if os.path.exists(ORIG_TRAIN_CSV):
        ds_train_orig = load_csv_dataset(ORIG_TRAIN_CSV)
        datasets.append(ds_train_orig)
    if os.path.exists(ORIG_DEV_CSV):
        ds_dev_orig = load_csv_dataset(ORIG_DEV_CSV)
        datasets.append(ds_dev_orig)

    # Load new prompts ---
    if not os.path.exists(NEW_PROMPTS_CSV):
        raise FileNotFoundError(f"Please run generate_new_prompts.py first → {NEW_PROMPTS_CSV}")
    ds_new = load_csv_dataset(NEW_PROMPTS_CSV)
    datasets.append(ds_new)

    # --- Concatenate all datasets into one DataFrame, then split ---
    combined_df = pd.concat([ds.to_pandas() for ds in datasets], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Now do an 80/10/10 stratified split by 'label'
    train_df, temp_df = train_test_split(
        combined_df,
        test_size=0.2,
        stratify=combined_df["label"],
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42
    )

    # Convert back to Hugging Face Datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds   = Dataset.from_pandas(val_df)
    test_ds  = Dataset.from_pandas(test_df)

    return DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})


# ──────────────────────────────────────────────────────────────────────────────
# TOKENIZE & PREPARE FOR TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_function(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

def prepare_for_training(datasets, tokenizer):
    """
    Applies tokenization to each split, renames 'label' → 'labels',
    and sets format to PyTorch.
    """
    tokenized = datasets.map(lambda b: tokenize_function(b, tokenizer), batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN: LOAD, TOKENIZE, AND CONTINUE TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Preparing the merged DatasetDict
    print("→ Loading & merging original + new prompts ...")
    datasets = prepare_merged_datasets()
    print(f"  • Train size: {len(datasets['train'])}")
    print(f"  • Validation size: {len(datasets['validation'])}")
    print(f"  • Test size: {len(datasets['test'])}")

    # Load tokenizer & current model
    print("→ Loading tokenizer & model from:", PREV_CHECKPOINT)
    tokenizer = DistilBertTokenizerFast.from_pretrained(PREV_CHECKPOINT)
    model     = DistilBertForSequenceClassification.from_pretrained(PREV_CHECKPOINT)

    # Tokenize all splits
    print("→ Tokenizing prompts ...")
    tokenized_datasets = prepare_for_training(datasets, tokenizer)

    # DataCollator (for dynamic padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_NEW_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        load_best_model_at_end=LOAD_BEST_AT_END,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_strategy="epoch",
        logging_steps=100
    )

    # 5f) Initialize Trainer
    print("→ Starting fine-tuning ...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 5g) Train!
    trainer.train()

    # 5h) Evaluate on held-out test set
    print("→ Evaluating on TEST set ...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("Test metrics:", test_metrics)

    # 5i) Save final model + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"→ Finished. New checkpoint saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
