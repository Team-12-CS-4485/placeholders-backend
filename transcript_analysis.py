import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

MODEL_ID = "gemini-3-flash-preview"
OUTPUT_FILE = "transcript_analysis.txt"

def analyze_transcript():

    transcript_path = "transcript.txt"
    
    with open(transcript_path, "r") as file:
        transcript = file.read()    
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents = ["Analyze the following transcript and provide insights, key points, and a summary:\n\n" + transcript],
            config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="medium")
                )
        )
        
        print("Transcript analysis completed successfully.")

        analysis = response.text
        with open(OUTPUT_FILE, "w") as output_file:
            output_file.write(analysis)
        print(f"Transcript analysis saved to {OUTPUT_FILE}")


    except Exception as e:
        
        print(f"An error occurred during transcript analysis: {e}")
if __name__ == "__main__":
    analyze_transcript()



