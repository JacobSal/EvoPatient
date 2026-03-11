import os
from dotenv import load_dotenv
from openai import OpenAI


def get_embeddings(text: str):
    """Read configuration from environment variables and call DashScope/OpenAI API to generate text embeddings."""
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not detected; please set it in .env or the system environment.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    completion = client.embeddings.create(
        model="text-embedding-v3",
        input=text,
        encoding_format="float"
    ).model_dump()

    return completion["data"][0]["embedding"]
