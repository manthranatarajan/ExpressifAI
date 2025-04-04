import openai # maybe swap with another api
import google.generativeai as genai
import os
from dotenv import load_dotenv

# load_dotenv() 
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = "AIzaSyC5IK6_np32zYxebmKvY8nWreZO2dLoAs8"
genai.configure(api_key=GEMINI_API_KEY)

def generate_semantic_description(emotion, action):

    try:
        model = genai.GenerativeModel("gemini-1.5-pro")  # gemini pro
        response = model.generate_content(f"Describe someone who is {action} while feeling {emotion} in a short, natural-sounding sentence.")
        return response.text.strip()  # extracting response text
    
    except Exception as e:
        print(f"Error in API call: {e}")  # Print error to console, there are some limits
        return f"Error generating description: {e}"
    
# print(generate_semantic_description("Happy", "Waving"))  # test


# Uncomment the following lines if you wanna use OpenAI API instead of Gemini

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY") 

# # Generates a descriptive sentence from detected action and emotion
# def generate_semantic_description(action: str, emotion: str) -> str:
        
#     prompt = (f"Given the action '{action}' and emotion '{emotion}', generate a natural-sounding sentence describing "
#               "what is happening in a human-readable way.")
    
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o",  # Replace with the preferred API model
#             messages=[{"role": "system", "content": "You are an assistant that describes human activities."},
#                       {"role": "user", "content": prompt}],
#             # max_tokens=50
#         )
        
#         # return response['choices'][0]['message']['content'].strip()   for the older api
#         return response.choices[0].message.content
    
#     except Exception as e:
#         print(f"Error in API call: {e}")  # Print error to console
#         return f"Error generating description: {e}"
    