import json
import faiss
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from model.evaluate import evaluate_t5
from model.evaluate import extract_criteria2, user_preferences_to_embedding
from embedding_extract.implicit_user_embedding import get_user_overall_embedding

def generate_city_embeddings(city_json_file):
    """
    Reads synthetic city data, extracts embeddings, and stores them in FAISS.

    Args:
        city_json_file (str): Path to JSON file containing city details.
    """
    with open(city_json_file, "r", encoding="utf-8") as file:
        cities = json.load(file)

    city_embeddings = []
    city_names = []

    for city in cities:
        city_name = city["name"] 
        city_metadata = city["metadata"]  # City description
        image_folder = city["image_folder"]  # Path to images
        
        # 🔹 Flatten city metadata into a string
        print(str(city_metadata))
        #city_metadata_text = " ".join([f"{key}: {value}" for key, value in city_metadata.items()])
        #print(city_metadata_text)
        #print(type(city_metadata_text))
        city_meta_text_structured = extract_criteria2(city_metadata, ["description", "weather", "landscape", "transportation", "activities", "cuisine"])
        print(city_meta_text_structured)
        city_meta_text_embedidng = user_preferences_to_embedding(city_meta_text_structured)
        print(city_meta_text_embedidng.shape)
        # 🔹 Check if images exist
        image_folder_exists = os.path.exists(image_folder) and os.listdir(image_folder)

        # 🔹 Compute city embedding (gracefully handling missing images or text)
        if image_folder_exists:
            city_embedding = get_user_overall_embedding(
                image_folder_path=image_folder, 
                prompt_path=None,  # Text should not be passed as a file path
                alpha=0.5, beta=0.5
            )
        else:
            #city_embedding = evaluate_t5(city_metadata_text)  # Use text-only if no images
            #print(city_embedding.shape)
            city_embedding = city_meta_text_embedidng
        # 🔹 Validate and store only valid embeddings
        if city_embedding is not None and not np.isnan(city_embedding).any() and len(city_embedding) > 0:
            city_embeddings.append(city_embedding)
            city_names.append(city_name)
        else:
            print(f"⚠️ Warning: Skipping city '{city_name}' due to invalid embedding!")

    # Ensure at least one valid embedding before proceeding
    if len(city_embeddings) == 0:
        raise ValueError("❌ No valid city embeddings were generated!")

    # Convert to FAISS index
    city_embeddings = np.array(city_embeddings).astype("float32")
    index = faiss.IndexFlatL2(city_embeddings.shape[1])
    index.add(city_embeddings)

    # Save FAISS index
    faiss.write_index(index, "city_embeddings.index")

    # Save city names for later retrieval
    with open("city_names.json", "w") as f:
        json.dump(city_names, f)

    print("✅ City embeddings stored in FAISS and saved as 'city_embeddings.index'")

# Run the function
generate_city_embeddings(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'synthetic_prompts', 'synthetic_city_database.json'))
