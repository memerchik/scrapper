import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict

# time taken to read data
# s_time = time.time()
# df = pd.read_csv("people_dataset.csv")
# e_time = time.time()
# print("Read without chunks: ", (e_time-s_time), "seconds")

# # data
# df.sample(10)

# ----------------------------------------------------------------------------------------------------

def getListOfAllNames():
    CSV_FILE = 'people_dataset.csv'
    # SAMPLES_PER_NAME = 10
    CHUNKSIZE = 100_000

    # --- PASS 1: count frequencies ---
    name_counter = Counter()
    for chunk in pd.read_csv(CSV_FILE, usecols=['first_name'], chunksize=CHUNKSIZE):
        name_counter.update(chunk['first_name'].dropna())

    sorted_names = [name for name, _ in name_counter.most_common()]
    return sorted_names
# top_10 = sorted_names[:10]
# target_names = set(top_10)

# # --- PASS 2: collect up to 10 row numbers per name ---
# samples = defaultdict(list)
# row_offset = 0

# for chunk in pd.read_csv(CSV_FILE, usecols=['first_name'], chunksize=CHUNKSIZE):
#     fn_series = chunk['first_name'].fillna('')
#     for i, name in enumerate(fn_series):
#         if name in target_names and len(samples[name]) < SAMPLES_PER_NAME:
#             samples[name].append(row_offset + i + 1)
#     row_offset += len(chunk)
#     # stop early if we've got 10 rows for each top name
#     if all(len(samples[n]) >= SAMPLES_PER_NAME for n in top_10):
#         break

# # --- OUTPUT ---
# for name in top_10:
#     print(f"{name} ({name_counter[name]} occurrences):")
#     for row_num in samples[name]:
#         print(f"  • Row {row_num}")
#     print()

# --------------------------------------------------------------------------------------------------

CSV_FILE = 'people_dataset.csv'
CHUNKSIZE = 100_000
TOP_N = 60000
SAMPLE_SIZE = 100

def get_top_names():
    counter = Counter()
    for chunk in pd.read_csv(CSV_FILE, usecols=['first_name'], chunksize=CHUNKSIZE):
        counter.update(chunk['first_name'].dropna())
    top = [name for name, _ in counter.most_common(TOP_N)]
    counts = {name: counter[name] for name in top}
    return top, counts

def choose_name(top_names, counts):
    # Displaying the top names with counts
    print(f"Top {TOP_N} first names by frequency:\n")
    for idx, name in enumerate(top_names, 1):
        print(f"{idx:>3}. {name} ({counts[name]} occurrences)")
    print()
    while True:
        choice = input("Enter the number or the exact name you’d like to inspect: ").strip()
        # if the user typed in a numeric index
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(top_names):
                return top_names[idx - 1]
        # if the user typed an exact name
        if choice in counts:
            return choice
        print("↳ Invalid selection; please try again.")

def collect_rows_for(name):
    rows = []
    row_offset = 0
    for chunk in pd.read_csv(CSV_FILE, chunksize=CHUNKSIZE):
        for i, row in chunk.iterrows():
            if pd.isna(row.get('first_name')):
                continue
            if row['first_name'] == name:
                # Building a dictionary of all columns with the row number
                record = row.to_dict()
                record['_row_number'] = row_offset + (i % CHUNKSIZE) + 1
                rows.append(record)
                if len(rows) >= SAMPLE_SIZE:
                    break
        row_offset += len(chunk)
        if len(rows) >= SAMPLE_SIZE:
            break
    return pd.DataFrame(rows)

def main():
    top_names, counts = get_top_names()
    name = choose_name(top_names, counts)
    print(f"\nCollecting up to {SAMPLE_SIZE} rows for “{name}”…\n")
    df = collect_rows_for(name)
    if df.empty:
        print(f"No rows found for name: {name}")
    else:
        # Reordering columns so _row_number is first
        cols = ['_row_number'] + [c for c in df.columns if c != '_row_number']
        print(df[cols].to_string(index=False))

if __name__ == "__main__":
    main()