import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API, but only if the key exists
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def generate_semantic_description(emotion, action):
    """Generates a semantic description using Gemini, with a fallback for errors."""
    # Provide a fallback if the API key is missing or invalid
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found. Using fallback description.")
        return f"The person appears to be {action} and feeling {emotion}."

    try:
        prompt = (
            f"Write a short, single-sentence, literal description of a person who is currently {action} and appears to be feeling {emotion}. "
            "Focus only on the action and emotion."
        )
        # Use a stable, generally available model
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        # Fallback to a simple, pre-formatted description on API failure
        return f"The person appears to be {action} and feeling {emotion}."
