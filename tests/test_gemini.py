
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env file.")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
   model="gemini-3.5-flash-lite",
    contents="Explain what a perceptron is in two simple sentences."
)

print("Gemini API connected successfully!")
print("\nResponse:")
print(response.text)
