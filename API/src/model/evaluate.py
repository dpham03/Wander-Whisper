from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import numpy as np
import os
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../../', 'fine_tuned_models', 'checkpoint-3000')

def clean_and_extract_values(text):
    """
    Removes unnecessary symbols like [], '', (), ensures spaces are retained for readability,
    and keeps only one space between words while preserving numerical values correctly.
    """
    text = re.sub(r'\(([^)]*\d+[^)]*)\)', r'\1', text)  # Remove parentheses but keep content inside if it contains numbers
    text = re.sub(r'[\[\]"]', ' ', text)  # Replace brackets and double quotes with spaces
    text = re.sub(r'\s*:\s*', ': ', text)  # Normalize colons
    text = re.sub(r'\s*,\s*', ', ', text)  # Normalize commas
    text = re.sub(r'\s*\'\s*', ' ', text)  # Replace single quotes with spaces
    text = re.sub(r'(?<=\d)\s*,\s*(?=\d{3})', '', text)  # Ensure numbers like 2,000 remain intact
    text = re.sub(r'\s+', ' ', text)  # Ensure only one space between words
    return text.strip()


def extract_criteria(generated_text, criteria_list):
    """
    Extracts structured criteria from generated text using regex.
    """
    extracted_criteria = {}
    for criterion in criteria_list:
        pattern = rf'"{criterion}":\s*"?(.*?)"?(,|$)'
        match = re.search(pattern, generated_text)
        if match:
            extracted_criteria[criterion] = match.group(1).strip()
    return extracted_criteria

import re

def extract_criteria2(text, criteria_list):
    """
    Extracts key-value pairs based on criteria from structured text.

    Args:
        text (dict or str): The structured dictionary containing criteria.
        criteria_list (list): List of criteria to extract.

    Returns:
        dict: Extracted structured criteria.
    """
    extracted_data = {}

    if isinstance(text, dict):  # If input is already a dictionary
        for key in criteria_list:
            if key in text:
                value = text[key]
                if isinstance(value, list):  # Convert list values to a string
                    value = ", ".join(value)
                extracted_data[key] = value
        return extracted_data

    # If text is not a dictionary, treat it as a raw string
    for key in criteria_list:
        pattern = rf'"{key}":\s*"?(.*?)"?(,|$)'
        match = re.search(pattern, text)
        if match:
            extracted_data[key] = match.group(1).strip()

    return extracted_data


def user_preferences_to_embedding(cleaned_output, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Converts structured user preferences into a 512D embedding.
    """
    model = SentenceTransformer(model_name)

    # Encode each structured attribute separately
    text_inputs = [f"{key}: {', '.join(value) if isinstance(value, list) else value}" for key, value in cleaned_output.items()]
    embeddings = model.encode(text_inputs, normalize_embeddings=True)  # Shape: (num_features, 512)

    # Mean pooling for final 512D embedding
    combined_embedding = np.mean(embeddings, axis=0)  # Shape: (512,)

    return combined_embedding

def evaluate_t5(input_text):
    """
    Takes an input paragraph, extracts structured attributes using T5, and returns a 512D embedding.
    """
    BASE_T5_MODEL = "t5-base"

    # Load T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = T5Tokenizer.from_pretrained(BASE_T5_MODEL, legacy=True)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_text = clean_and_extract_values(input_text)

    criteria_list = [
        "departure_location", "departure_month", "return_month", "budget", "weather_preference",
        "destination_type", "travel_companions", "preferred_activities", "food_preference", "travel_duration",
        "accommodation_preference", "transportation_mode", "transportation_preference", "season", "event_interest",
        "safety_preference", "language_preference", "visa_requirement", "travel_theme",
        "sustainability_focus", "trip_intensity", "cultural_preference", "shopping_style", "internet_availability",
        "luxury_rating", "pet_friendly", "wellness_activities", "adventure_level", "nightlife_preferences",
        "currency_preference", "insurance_preference", "travel_addon"
    ]

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate structured attributes
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            #num_beams=5,
            #repetition_penalty=1.2
        )

    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract structured attributes
    structured_output = extract_criteria2(generated_text, criteria_list)
    #print(structured_output)
    #print(structured_output)
    # Convert structured attributes into a 512D embedding
    user_embedding = user_preferences_to_embedding(structured_output)

    return user_embedding

# Example usage
if __name__ == "__main__":
    input_text = "I am departing from Toronto, Canada in July and will return in August. My budget is adventure travel budget ($1,000 - $3,000 for guided tours), and I prefer local delicacies. I will be traveling solo for one week, and I enjoy hiking. I prefer a mountainous destination with cool ocean breeze weather. I will travel via high-speed train and prefer to use local currency for transactions. My accommodation choice is eco-lodge, and my transportation preference is walking. I want an adventure experience with wildlife conservation focus. My trip should be extreme adventure, and I love indigenous culture. I am interested in Carnival in Rio and will need full travel insurance. I prefer locations with female-friendly and wheelchair access support. For nightlife, I prefer casual bars, and my adventure level is high. I will also be adding guided city tours to my trip."
    #print(input2)
    #print(extract)
    embedding = evaluate_t5(input_text)
    print(f"User Embedding Shape: {embedding.shape}")  # (512,)
    print(f"Sample Embedding: {embedding[:5]}")
