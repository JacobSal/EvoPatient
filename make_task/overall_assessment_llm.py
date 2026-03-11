import json
from pathlib import Path
from Simulated.simulated_patient.api_call import llm_api


def overall_assessment_patient(question: str, useful_info: str, ans: str, profile: str):
    """
    Assess the patient's answer quality using LLM.
    Returns: score (int), relevance (int), faithfulness (int), human_likeness (int)
    """
    prompt_data_path = Path("Simulated/Prompt/prompt_data.json")
    with prompt_data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    assessment_prompt = "".join(data.get("overall_assessment", ["Please assess the patient's answer on a scale of 1-5 for overall quality, relevance, faithfulness, and human-likeness. Return as JSON: {\"score\": int, \"rel\": int, \"faith\": int, \"human\": int}"]))
    
    prompt = assessment_prompt.format(question=question, information=useful_info, answer=ans, profile=profile)
    messages = [{"role": "user", "content": prompt}]
    response = llm_api(messages)
    
    # Parse response as JSON
    try:
        result = json.loads(response)
        score = result.get("score", 0)
        rel = result.get("rel", 0)
        faith = result.get("faith", 0)
        human = result.get("human", 0)
    except (json.JSONDecodeError, KeyError):
        # Fallback to defaults if parsing fails
        score = rel = faith = human = 0
    
    return score, rel, faith, human