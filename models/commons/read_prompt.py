from pathlib import Path
import json

def read_prompt(prompt_file):
    prompt_path =  "models/prompts/" + prompt_file
    with open(prompt_path, "r") as f:
        file_content = f.read()
    return file_content

def read_json(file_path: str):
    with open(file_path, 'r+') as f:
        company_details = json.load(f)
    return company_details