import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

API_TOKEN = os.getenv('API_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/WakandianEngineer/RoBERTa-stress-detection"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def get_label_score(data):
    # Rename labels
    label_map = {'LABEL_1': '1', 'LABEL_0': '0'}
    
    # Flatten the list (in case it's nested)
    flat_list = [item for sublist in data for item in sublist]
    
    # Find the item with the highest score
    highest = max(flat_list, key=lambda x: x['score'])
    
    # Return the transformed label and score
    return {label_map[highest['label']]: highest['score']}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()