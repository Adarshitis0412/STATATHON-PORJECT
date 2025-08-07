import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GoogleAIReport:
    def __init__(self, model="gemini-1.5-flash"):
        # Get API key from .env
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
        
        genai.configure(api_key=api_key)#Ye API key Gemini client me set karta hai, taaki tum model ko call kar sako
        self.model = model  # Store the model name for later use in report generation
    # Function to generate an AI-based report
    # Takes two inputs: stats_df (data statistics in DataFrame) and logs (data cleaning steps)
    def generate_report(self, stats_df: pd.DataFrame, logs: list):
        prompt = f"""
        You are an expert data analyst.
        Based on the statistics and cleaning log below, generate a structured survey analysis report.

        Statistics:
        {stats_df.head(10).to_string()}

        Cleaning Log:
        {', '.join(logs)}

        Write the report in sections:
        1. Introduction
        2. Data Cleaning Summary
        3. Key Insights
        4. Recommendations
        Keep it concise and professional.
        """

        model = genai.GenerativeModel(self.model)  # Create a Gemini model object using the selected model (default = gemini-1.5-flash)
        response = model.generate_content(prompt)# Send the prompt to the Gemini model and wait for its generated response
        return response.text # Extract only the text part of the response and return it