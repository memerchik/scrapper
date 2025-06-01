import pandas as pd

# Read positives
with open("positive_prompts.txt", "r", encoding="utf-8") as f_pos:
    positive = [line.strip() for line in f_pos if line.strip()]

# Read negatives
with open("negative_prompts.txt", "r", encoding="utf-8") as f_neg:
    negative = [line.strip() for line in f_neg if line.strip()]

# Build DataFrame
data = {
    "text": positive + negative,
    "label": [1] * len(positive) + [0] * len(negative)
}
df = pd.DataFrame(data)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv("dataset.csv", index=False)
print(f"Saved dataset with {len(df)} examples to dataset.csv")

train_split = 0.8
dev_split = 0.1

n = len(df)
n_train = int(n * train_split)
n_dev = int(n * dev_split)

df.iloc[:n_train].to_csv("train.csv", index=False)
df.iloc[n_train:n_train + n_dev].to_csv("dev.csv", index=False)
df.iloc[n_train + n_dev:].to_csv("test.csv", index=False)

print(f"Train: {n_train}, Dev: {n_dev}, Test: {n - n_train - n_dev}")
