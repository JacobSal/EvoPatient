from openai import OpenAI
from dotenv import load_dotenv  # For loading environment variables from .env file
load_dotenv(override=True)

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role":"user","content":"Say hello"}]
)

print(response.choices[0].message.content)