import json
import random
import re
from pathlib import Path

import openpyxl
from Simulated.simulated_patient.api_call import llm_api


# ---------- Utility functions ----------

def select_random_positions(lst, percentage):
    """Randomly select index positions from a list by percentage."""
    if not lst or percentage <= 0:
        return []
    k = int(len(lst) * (percentage / 100.0))
    k = max(0, min(k, len(lst)))
    if k == 0:
        return []
    return random.sample(range(len(lst)), k)


def split_string_by_punctuation(text: str):
    """
    Split text into segments of punctuation and non-punctuation, preserving the original order.
    e.g.: 'a,b' -> ['a', ',', 'b']
    """
    pattern = re.compile(r'[^\w\s]|[\w\s]+', flags=re.UNICODE)
    return [m.group(0) for m in pattern.finditer(text or '')]


def random_dropout(split_tokens):
    """
    Randomly drop segments around about 30% of positions using a simple heuristic:
    - if a digit is hit: remove that position and the next 1-2 positions
    - if a letter is hit: remove that position, and also remove preceding two if they are digits
    - for other symbols: try removing adjacent positions (including the previous digit, etc.)
    """
    selected = select_random_positions(split_tokens, 30)
    to_delete = set()

    for pos in selected:
        token = split_tokens[pos]
        if token.isdigit():
            to_delete.add(pos)
            if pos + 1 < len(split_tokens):
                to_delete.add(pos + 1)
            if pos + 2 < len(split_tokens):
                to_delete.add(pos + 2)
        elif token.isalpha():
            if pos - 2 >= 0 and split_tokens[pos - 2].isdigit():
                to_delete.add(pos - 2)
            to_delete.add(pos)
        else:
            if pos - 1 >= 0 and split_tokens[pos - 1].isdigit():
                to_delete.add(pos - 1)
            if pos + 1 < len(split_tokens):
                to_delete.add(pos + 1)

    return [t for i, t in enumerate(split_tokens) if i not in to_delete]


def dropout_vague(text: str) -> str:
    tokens = split_string_by_punctuation(text or "")
    kept = random_dropout(tokens)
    return "".join(kept)


# ---------- Business logic ----------

def get_patient_info(sheet_name: str, row_number: int, col_number: int):
    """
    Read a specified cell from the relative path dataset/patient_text.xlsx
    and write it to data["resource"][0] in Simulated/Prompt/prompt_data.json.
    """
    xlsx_path = Path("dataset") / "patient_text.xlsx"
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"File not found: {xlsx_path}")

    wb = openpyxl.load_workbook(xlsx_path)
    try:
        sheet = wb[sheet_name]
        cell_value = sheet[row_number][col_number].value
    finally:
        wb.close()

    text = str(cell_value or "")

    json_path = Path("Simulated") / "Prompt" / "prompt_data.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"File not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure the structure exists
    if "resource" not in data or not isinstance(data["resource"], list):
        data["resource"] = [""]
    if not data["resource"]:
        data["resource"].append("")
    data["resource"][0] = text

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return text


def get_vague_patient_info(sheet_name: str, row_number: int, col_number: int):
    """
    Read patient information -> apply 'vagueness' -> call LLM to generate a vague expression
    -> write back to data["vague_resource"][0] in prompt_data.json.
    Returns (original text, vague text)
    """
    patient_info = get_patient_info(sheet_name, row_number, col_number)
    patient_info_drop = dropout_vague(patient_info)

    json_path = Path("Simulated") / "Prompt" / "prompt_data.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"File not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Assemble vagueness prompt
    if "vagueness" not in data or not isinstance(data["vagueness"], list):
        raise KeyError("prompt_data.json is missing 'vagueness' configuration or its type is not a list")
    vagueness_prompt = "".join(data["vagueness"])
    prompt = vagueness_prompt.format(information=patient_info_drop)

    vague_patient_info = llm_api([{"role": "user", "content": prompt}])

    # Write back vague_resource
    if "vague_resource" not in data or not isinstance(data["vague_resource"], list):
        data["vague_resource"] = [""]
    if not data["vague_resource"]:
        data["vague_resource"].append("")
    data["vague_resource"][0] = vague_patient_info

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return patient_info, vague_patient_info
