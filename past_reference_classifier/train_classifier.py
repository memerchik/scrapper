from datasets import load_dataset
from transformers import DistilBertTokenizerFast

# 1. Load CSVs as Dataset objects
data_files = {
    "train": "train.csv",
    "validation": "dev.csv",
    "test": "test.csv"
}
dataset = load_dataset("csv", data_files=data_files)

# 2. Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 3. Preprocess function
def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# 4. Apply tokenization
dataset = dataset.map(tokenize_batch, batched=True)

# 5. Rename label column to “labels” if necessary
dataset = dataset.rename_column("label", "labels")

# 6. Set format for PyTorch
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(dataset["train"][0])

from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # binary classification: past-reference vs. not
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./past_ref_classifier",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2
)

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

metrics = trainer.evaluate(eval_dataset=dataset["test"])
print("Test set metrics:", metrics)

trainer.save_model("./past_ref_classifier/best_model")
tokenizer.save_pretrained("./past_ref_classifier/best_model")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # optional for nicer plot

# Get predictions on test1
preds_output = trainer.predict(dataset["test"])
preds = np.argmax(preds_output.predictions, axis=1)
labels = preds_output.label_ids

cm = confusion_matrix(labels, preds, labels=[0,1])
print("Confusion Matrix:\n", cm)

# Plot heatmap (if seaborn is installed)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NoPast", "Past"], yticklabels=["NoPast", "Past"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix on Test Set")
plt.show()

misclassified = []
for idx, (true_label, pred_label) in enumerate(zip(labels, preds)):
    if true_label != pred_label:
        misclassified.append({
            "text": dataset["test"][idx]["text"],
            "true_label": true_label,
            "pred_label": pred_label
        })
# Print first 10
for i, example in enumerate(misclassified[:10], 1):
    print(f"Example {i}:")
    print("Text:", example["text"])
    print("True Label:", example["true_label"], "Predicted:", example["pred_label"])
    print("-" * 50)