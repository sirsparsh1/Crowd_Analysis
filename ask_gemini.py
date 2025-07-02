import pandas as pd
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel

# analysing WITH GEMINI
def analyze_data_with_gemini(csv_path='crowd_data.csv', project_id="project_id", location="us-central1"):
    """Reads CSV data and asks Gemini for insights."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found. Run main.py first to generate data.")
        return

    # Initializing Vertex AI
    vertexai.init(project=project_id, location=location)
    
    # Loading the Gemini 1.5 Flash model
    model = GenerativeModel("gemini-2.5-flash")

    # Converting dataframe to a string format for the prompt
    data_string = df.to_string()

    # promot to Gemini
    prompt = f"""
    You are a crowd analytics assistant. Based on the following data from a CSV file, please provide an overview of the activity.
    What are the most common objects detected? What is the general sentiment of the people?

    Data:
    {data_string}
    """

    print("--- Asking Gemini for an analysis... ---")
    response = model.generate_content(prompt)
    print(response.text)


# RUN IT 
if __name__ == '__main__':
    
    GCP_PROJECT_ID = "project_id" 

    if GCP_PROJECT_ID == "your-gcp-project-id":
        print("Error: Please replace 'your-gcp-project-id' with your actual Google Cloud Project ID.")
    else:
        analyze_data_with_gemini(project_id=GCP_PROJECT_ID)