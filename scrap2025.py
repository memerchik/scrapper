import requests
import json
import math
from typing import Literal

AllowedOut = Literal['array', 'file', 'json', "file+array"]

def get_prompts(prompt="", quant=0, out: AllowedOut = "array"):

  def roundup(x):
    return int(math.ceil(x / 100.0)) * 100
  
  headers = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
  }

  if(len(prompt)==0 and quant<=0):
    QUERY = input("Please enter your desired request: ")
    QUANTITY = roundup(int(input("Please enter the number of results you wish to obtain (rounded to 100): ")))
  elif len(prompt)==0 and quant>0:
    QUERY = input("Please enter your desired request: ")
    QUANTITY=quant
  elif len(prompt)>0 and quant>0:
    QUERY=prompt
    QUANTITY=quant
  else:
    QUERY=prompt
    QUANTITY = roundup(int(input("Please enter the number of results you wish to obtain (rounded to 100): ")))
      

  r = requests.post('https://lexica.art/api/infinite-prompts', json={
    "text": QUERY,
    "model": "lexica-aperture-v3.5",
    "searchMode": "images",
    "source": "search",
    "cursor": 0
  }, headers=headers)

  CURSOR = 100
  current_cursor = 0
  prompt_array = []

  while current_cursor<QUANTITY:
      r = requests.post('https://lexica.art/api/infinite-prompts', json={
        "text": QUERY,
        "model": "lexica-aperture-v3.5",
        "searchMode": "images",
        "source": "search",
        "cursor": current_cursor
      }, headers=headers)
      print(f"Status Code: {r.status_code} ({current_cursor} - {current_cursor+CURSOR})")

      result = r.json()

      for prompt in result["prompts"]:
        prompt_array.append(prompt["prompt"])
      current_cursor=current_cursor+CURSOR
  
  if out=="array":
    return ["array", prompt_array]
  
  elif out=="file":
    with open(f"{QUERY}.txt", "w") as f:
      f.write(json.dumps(prompt_array, indent=2))
    return ["file"]
  
  elif out=="json":
    return ["json", json.dumps(prompt_array, indent=2)]
  
  elif out=="file+array":
    with open(f"{QUERY}.txt", "w") as f:
      f.write(json.dumps(prompt_array, indent=2))
    return ["file+array", prompt_array]
