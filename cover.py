from pathlib import Path
import os
import re
import json
import csv
import time

from dotenv import load_dotenv
from Simulated.simulated_patient.vagueness import get_vague_patient_info
from Simulated.simulated_patient.patient_agent import Patient
from Simulated.simulated_patient.api_call import llm_api
from Simulated.simulated_patient.agent_evolve import get_text_embedding


# ============== Environment variables (privacy hidden, keys read internally by llm_api) ==============
load_dotenv(override=True)
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not detected; please set it in the system or .env.")


# ============== Utility functions ==============
def match_star(context: str) -> str:
    """Match **...** and return the original text without asterisks; raise an exception if not found."""
    m = re.search(r"\*\*(.*?)\*\*", context, flags=re.DOTALL)
    if not m:
        raise ValueError("No match found")
    return re.sub(r"\*", "", m.group(0))


def read_prompt() -> dict:
    """Read and concatenate the prompts from Simulated/Prompt/prompt_data.json."""
    p = Path("Simulated/Prompt/prompt_data.json")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    final = {}
    for k, v in data.items():
        # v may be a list (each item a segment), join into a full prompt
        if isinstance(v, list):
            final[k] = "".join(v)
        else:
            final[k] = str(v)
    return final


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_token_count():
    """Clear token count files (relative path)."""
    tp_overall = Path("./make_task/token_count/token_overall.txt")
    tp_stream = Path("./make_task/token_count/token_stream.txt")
    ensure_parent(tp_overall)
    tp_overall.write_text("", encoding="utf-8")
    tp_stream.write_text("", encoding="utf-8")


def get_token_count() -> str:
    """Read accumulated tokens (return '0' if the file does not exist or is empty)."""
    tp_stream = Path("./make_task/token_count/token_stream.txt")
    if not tp_stream.exists():
        return "0"
    token_str = tp_stream.read_text(encoding="utf-8")
    nums = re.findall(r"\d+", token_str)
    return nums[-1] if nums else "0"


# ============== Main flow ==============
def cover(sheet_name: str = "病程记录_首次病程", row_number: int = 6, col_number: int = 1):
    clean_token_count()

    test_label = str(time.time())
    parent_folder = Path("pool")
    directory = parent_folder / test_label
    (directory / "doctor_record").mkdir(parents=True, exist_ok=True)
    print(f"Folder {test_label} has been created under {parent_folder}.")

    # Get patient's full and vague information
    resource, vague_info = get_vague_patient_info(sheet_name, row_number, col_number)

    # Save resource and vague text (relative path)
    (directory / "resource.txt").write_text(resource, encoding="utf-8")
    (directory / "vague.txt").write_text(vague_info, encoding="utf-8")

    prompt_data = read_prompt()
    patient = Patient(vague_info, resource, str(directory), prompt_data)

    # Assign department
    office = match_star(patient.assign_office())

    # Generate main complaint
    def generate_main_complaint() -> str:
        patient_question = patient.generate_patient_question()
        patient_answer = match_star(patient_question)
        print("Patient question:", patient_answer)
        return patient_answer

    main_complaint = generate_main_complaint()

    # Generate cover
    prompt = prompt_data["cover"].format(office, resource)
    messages = [{"role": "user", "content": prompt}]
    response = llm_api(messages)
    print(response)

    # Extract possible multiple matches from cover
    matched = re.findall(r"\*\*(.*?)\*\*", response, flags=re.DOTALL)

    # Embed main complaint and write to pool table
    emb = get_text_embedding(main_complaint)

    pool_csv = Path("dataset/pool.csv")
    ensure_parent(pool_csv)
    new_file = not pool_csv.exists()
    with pool_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["main_complaint", "embedding_main_complaint", "question"])
        writer.writerow([
            main_complaint,
            json.dumps(emb, ensure_ascii=False),         # Prevent commas from affecting CSV
            json.dumps(matched, ensure_ascii=False)      # Save list in JSON format
        ])


def cache() -> int:
    """Read/initialize the line number inside case_cache.txt."""
    cache_path = Path("./make_task/case_cache.txt")
    ensure_parent(cache_path)
    if not cache_path.exists():
        cache_path.write_text("0", encoding="utf-8")
        return 0
    txt = cache_path.read_text(encoding="utf-8").strip()
    return int(txt) if txt.isdigit() else 0


def write_cache(value: int):
    cache_path = Path("./make_task/case_cache.txt")
    ensure_parent(cache_path)
    cache_path.write_text(str(value), encoding="utf-8")


if __name__ == "__main__":
    col_number = 1
    sheet_name = "病程记录_首次病程"  # "Medical record_first visit"  # "Medical record_first visit"

    row_number = cache()
    while row_number <= 1300:
        row_number += 1
        cover(sheet_name, row_number, col_number)
        write_cache(row_number)
