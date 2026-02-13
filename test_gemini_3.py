import os
import PIL.Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-3-flash-preview"
OUTPUT_FILE = "gemini_test_output.txt"

def run_test():
    
    image_path = "MrBeast_Thumbnail.jpg"
    
    if os.path.exists(image_path):
        print(f"\nVision Test on {image_path}")
        img = PIL.Image.open(image_path)
        
        try:
            img_response = client.models.generate_content(
                model=MODEL_ID,
                contents=["Describe the details of this image.", img],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="medium")
                )
            )
            
            
            print(f"Vision Response received. Writing to {OUTPUT_FILE}...")

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write(f"MODEL: {MODEL_ID}\n")
                f.write(f"IMAGE: {image_path}\n")
                f.write("-" * 30 + "\n")
                f.write(img_response.text)
            
            print("Successfully saved.")

        except Exception as e:
            print(f"Error during generation: {e}")
    else:
        print(f"Error: {image_path} not found.")

if __name__ == "__main__":
    run_test()