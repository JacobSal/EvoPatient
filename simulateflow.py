from pathlib import Path
import re
import json
import csv
import time
import random
import shutil
import os

from dotenv import load_dotenv
from Simulated.simulated_patient.vagueness import get_vague_patient_info
from Simulated.simulated_patient.patient_agent import Patient
from Simulated.simulated_patient.doctor_agent import Doctor
from Simulated.simulated_patient.api_call import llm_api  # for LLM calls (should read env variables internally)
# If needed: from Simulated.simulated_patient.agent_evolve import get_text_embedding


# ====== Environment variables (keys not stored, actual reading inside llm_api) ======
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not detected; please set it in the system or .env.")


# ====== Utility functions ======
def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def match_star(context: str) -> str:
    """Match **...** and return the original text without asterisks; raise an exception if not found."""
    m = re.search(r"\*\*(.*?)\*\*", context, flags=re.DOTALL)
    if not m:
        raise ValueError("No match found")
    return re.sub(r"\*", "", m.group(0))


def read_prompt() -> dict:
    """Read and concatenate the prompts from Simulated/Prompt/prompt_data.json."""
    p = Path("Simulated/Prompt/prompt_data.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    final = {}
    for k, v in data.items():
        final[k] = "".join(v) if isinstance(v, list) else str(v)
    return final


def clean_token_count():
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


def count_chinese_characters(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


# ====== Main flow ======
def flow(sheet_name: str = "病程记录_首次病程", row_number: int = 6, col_number: int = 1):  # sheet name means "Medical record_first visit"
    clean_token_count()

    test_label = str(time.time())
    parent_folder = Path("exp1")
    directory = parent_folder / test_label
    (directory / "doctor_record").mkdir(parents=True, exist_ok=True)
    print(f"Folder {test_label} has been created under {parent_folder}.")

    # Read patient's complete information and vague information
    resource, vague_info = get_vague_patient_info(sheet_name, row_number, col_number)

    # Save the original resource and vague information
    (directory / "resource.txt").write_text(resource, encoding="utf-8")
    (directory / "vague.txt").write_text(vague_info, encoding="utf-8")

    prompt_data = read_prompt()
    patient = Patient(vague_info, resource, str(directory), prompt_data)

    # Assign department (extract department name from **...**)
    office = match_star(patient.assign_office())

    # Generate main complaint
    def generate_main_complaint() -> str:
        patient_question = patient.generate_patient_question()
        patient_answer = match_star(patient_question)
        print("Patient question:", patient_answer)
        return patient_answer

    main_complaint = generate_main_complaint()

    # Initialize doctor and make the first inquiry
    doctor = Doctor(patient, office, main_complaint, str(directory), prompt_data)
    resp = doctor.doctor_qus(main_complaint, 0, 0, 0, 0)
    try:
        doctor_question = match_star(resp)
    except Exception:
        doctor_question = resp

    # Patient's first answer
    patient_answer, score, rel, faith, human = patient.patient_ans(doctor_question)

    max_turn = 10
    cnt = 0
    auto = True
    random_crisis_num = random.randrange(int(max_turn / 2), max_turn)

    # Turn records (in experiment directory and top-level convenience file)
    exp_csv = directory / "question_record.csv"
    top_csv = Path("question_record.csv")
    for p in (exp_csv, top_csv):
        if not p.exists():
            ensure_parent(p)
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "row", "question", "answer", "token_count_doctor", "token_count_patient",
                    "resource", "doctor_time", "patient_time", "question_cnt", "answer_cnt"
                ])

    while cnt < max_turn:
        cnt += 1

        # Introduce a crisis event at a random time
        if cnt == random_crisis_num:
            patient_crisis = patient.crisis_begin()
            doctor_crisis_ans = doctor.doctor_crisis_answer(office, patient_crisis)
            patient_ans_after_crisis = patient.patient_crisis_ans(doctor_crisis_ans)

            crisis_file = directory / "crisis.txt"
            crisis_txt = (
                f"Round:**{cnt}**Patient crisis:**{patient_crisis}**"
                f"Doctor's response to crisis:**{doctor_crisis_ans}**"
                f"Patient's reaction to doctor's response:**{patient_ans_after_crisis}**"
            )
            crisis_file.write_text(crisis_txt, encoding="utf-8")

        start_time = time.time()

        if auto:
            # Doctor asks a question
            next_q = doctor.doctor_qus(patient_answer, score, rel, faith, human)
            # Fix: decision should be based on the current returned question rather than the initial resp
            if next_q == "skip":
                continue
            if next_q == "conclusion":
                print("Sufficient information obtained, stopping early")
                break
            try:
                doctor_question = match_star(next_q)
            except Exception:
                doctor_question = next_q
        else:
            doctor_question = input("Please enter a question: ")

        token_count_doctor = get_token_count()
        middle_time = time.time()

        # Patient answers
        patient_answer, score, rel, faith, human = patient.patient_ans(doctor_question)
        token_count_patient = get_token_count()
        end_time = time.time()

        doctor_time = middle_time - start_time
        patient_time = end_time - start_time

        # Write to both CSVs (experiment directory & top-level)
        row = [
            row_number,
            doctor_question.replace("\n", ""),
            patient_answer.replace("\n", ""),
            token_count_doctor,
            token_count_patient,
            resource,
            doctor_time,
            patient_time,
            count_chinese_characters(doctor_question),
            count_chinese_characters(patient_answer),
        ]
        for p in (exp_csv, top_csv):
            with p.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    # Conclusion
    conclusion = doctor.conclusion()
    (directory / "conclusion.txt").write_text(conclusion, encoding="utf-8")

    # Calculate elapsed time (starting from the last round)
    time_cost = time.time() - start_time
    (directory / "time_cost.txt").write_text(str(time_cost), encoding="utf-8")

    # Migrate token count files
    source_folder = Path("./make_task/token_count")
    destination_folder = directory / "token_count"
    destination_folder.mkdir(parents=True, exist_ok=True)
    if source_folder.exists():
        for file_path in source_folder.glob("*.txt"):
            shutil.move(str(file_path), str(destination_folder))
            print(f"File {file_path.name} has been moved to {destination_folder}")
