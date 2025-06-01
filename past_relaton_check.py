# import requests
# import json
# import math

from scrap2025 import get_prompts
from csv_parsing import getListOfAllNames

def clean_names(names: list[str], blacklist: list[str]) -> list[str]:
    """
    Given a list of names, return a new list where:
      - every name is lowercased
      - any name appearing in the blacklist (case-insensitive) is removed
    """
    # Precompute lowercase blacklist set for O(1) membership tests
    blacklist_lc = {n.lower() for n in blacklist}

    # Build result: lowercase each name, include only if not in blacklist
    return [
        name.lower()
        for name in names
        if name.lower() not in blacklist_lc
    ]

prompts_list = get_prompts(out="file+array")[1]

print(prompts_list)

forbidden_names = ["A", "AN", "DE", "HE", "THE", "SHE", "MANY", "UN", "UNE", "MAN", "TIME", "SEC", "F", "1","2","3","4","5","6","7","8","9","B","C","D", "ASSUME",
                   "I","NATURAL",]

all_names = getListOfAllNames()
all_names = all_names[:60_000]
print(len(all_names))
all_names = clean_names(all_names, forbidden_names)

counter = 0
for prompt in prompts_list:
    counter=counter+1
    split_prompt = prompt.replace(",", " ").replace("   ", " ").replace("  ", " ")
    split_prompt = split_prompt.split(" ")
    for word in split_prompt:
        if word in all_names:
            print(f"Found a match in prompt number {counter} - {prompt[:30]}; Word - {word}")