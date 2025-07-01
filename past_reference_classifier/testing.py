# import csv
# import random
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from typing import Literal
import os
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Load the fine-tuned model and run inference on each prompt
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_dir="./past_ref_classifier/updated_model"):
    """
    Load tokenizer and model. Adjust model_dir if needed.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

@torch.no_grad()
def classify_prompts(df, tokenizer, model, max_length=128, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Take a DataFrame with 'text' column, run the classifier, and return:
    - pred_label: 0 or 1
    - prob_past: probability of label=1
    """
    model.to(device)
    pred_labels = []
    prob_pasts = []
    for i, txt in enumerate(df["text"]):
        inputs = tokenizer(
            txt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()  # shape: (2,)
        probs = torch.softmax(logits, dim=-1)
        prob_past = probs[1].item()
        pred_label = int(prob_past >= 0.5)

        pred_labels.append(pred_label)
        prob_pasts.append(prob_past)

        if (i + 1) % 50 == 0:
            print(f"Classified {i+1}/{len(df)} prompts")

    df["pred_label"] = pred_labels
    df["prob_past"] = prob_pasts
    return df


def read_txt_as_dataframe(txt_input):
    # Read and strip lines, dropping any blank ones
    if os.path.isfile(txt_input):
        with open(txt_input, 'r', encoding='utf-8') as f:
            raw = f.read()
    else:
        # Assume txt_input itself is the text content
        raw = txt_input

    # Split into lines, strip whitespace, remove blanks
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    # Remove 2nd line if it's "[" 
    if len(lines) > 1 and lines[0] == "[":
        lines.pop(0)

    # Remove last line if it's "]"
    if lines and lines[-1] == "]":
        lines.pop(-1)

    # Build DataFrame
    df = pd.DataFrame(lines, columns=['text'])
    return df

AllowedMode = Literal['txt_file_path', 'txt_file', 'csv_file_path', "csv_file"]
AllowedOut = Literal[True, False]

def run_tagging(mode: AllowedMode, data_or_path="", out_dir=".", prefix="data", out_as_a_df_variable: AllowedOut = False):
    
    
    if mode=="csv_file" or mode=="csv_file_path":
        df = pd.read_csv(data_or_path)
    elif mode=="txt_file_path" or mode=="txt_file":
        df = read_txt_as_dataframe(data_or_path)
    else:
        return 0
    # Load model + tokenizer
    tokenizer, model = load_model_and_tokenizer(
        model_dir="./past_ref_classifier/updated_model_3"
    )

    #Classify each prompt
    df_results = classify_prompts(df, tokenizer, model)

    # Print first 20 results to console, and save full CSV
    print("\nFirst 20 inference results:\n")
    print(df_results.head(20).to_string(index=False))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.csv"
    full_path = f"{out_dir.rstrip('/')}/{filename}"

    df_results.to_csv(full_path, index=False)
    print(f"\nSaved full results (with pred_label and prob_past) to {filename}")

    if out_as_a_df_variable ==True:
        return df_results


if __name__ == "__main__":
    runMode = int(input("Please select a running mode:\n\n1. Txt file path\n2. Csv file path\n\n"))
    if runMode>0 and runMode<5:
        if runMode==1:
            path_to_txt=input("Please provide path to the txt file\n")
            run_tagging(mode="txt_file_path", data_or_path=path_to_txt)
        elif runMode==2:
            path_to_csv=input("Please provide path to the csv file\n")
            run_tagging(mode="csv_file_path", data_or_path=path_to_csv)