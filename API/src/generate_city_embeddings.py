import json
import faiss
import numpy as np
import os
import sys
from pymongo import MongoClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from implicit_user_embedding import get_user_overall_embedding
from API.src.model.evaluate import evaluate_t5
from API.src.model.evaluate import extract_criteria2, user_preferences_to_embedding

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
        #print(str(city_metadata))
        print(city_name)
        city_names.append(city_name)
        #city_metadata_text = " ".join([f"{key}: {value}" for key, value in city_metadata.items()])
        #print(city_metadata_text)
        #print(type(city_metadata_text))
        city_meta_text_structured = extract_criteria2(city_metadata, ["description", "weather", "landscape", "transportation", "activities", "cuisine"])
        #print(city_meta_text_structured)
        city_meta_text_embedidng = user_preferences_to_embedding(city_meta_text_structured)
        #print(city_meta_text_embedidng.shape)
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


    """
    Reads city data from MongoDB, extracts embeddings, and stores them in FAISS.
    
    Args:
        mongo_uri (str): MongoDB connection URI for Atlas.
        db_name (str): Database name in MongoDB.
        collection_name (str): Collection name in MongoDB containing city data.
    """
    # Connect to MongoDB Atlas
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Fetch all cities data
    cities = collection.find()

    city_embeddings = []
    city_names = []

    for city in cities:
        city_name = city["name"]
        city_metadata = city["metadata"]  # City description
        image_folder = city["image_folder"]  # Path to images
        
        # 🔹 Flatten city metadata into a string
        print(city_name)
        city_names.append(city_name)
        
        # Extract structured metadata for embeddings
        city_meta_text_structured = extract_criteria2(city_metadata, ["description", "weather", "landscape", "transportation", "activities", "cuisine"])
        city_meta_text_embedidng = user_preferences_to_embedding(city_meta_text_structured)

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
    faiss.write_index(index, "src/faiss_indexing/city_embeddings.index")

    # Save city names for later retrieval
    with open("src/faiss_indexing/city_names.json", "w") as f:
        json.dump(city_names, f)

    print("✅ City embeddings stored in FAISS and saved as 'city_embeddings.index'")

def generate_city_embeddings_mongodb(mongo_uri, db_name, collection_name):
    """
    Reads city data from MongoDB, extracts embeddings, and updates FAISS index incrementally.
    """
    # Connect to MongoDB Atlas
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Load existing city names if available
    city_names_path = "/Users/apple/Documents/GitHub/Wander-Whisper/API/data/embeddings/city_names_mongo.json"
    if os.path.exists(city_names_path):
        with open(city_names_path, "r") as f:
            existing_city_names = json.load(f)
    else:
        existing_city_names = []
    
    # Fetch all cities data from MongoDB
    all_cities = list(collection.find())
    all_city_names = [city["name"] for city in all_cities]
    
    # Identify new cities that are not already processed
    new_cities = [city for city in all_cities if city["name"] not in existing_city_names]
    print(f"🆕 Found {len(new_cities)} new cities to process.")
    
    new_city_embeddings = []
    new_city_names = []
    
    for city in new_cities:
        city_name = city["name"]
        print(city_name)
        city_metadata = city["metadata"]
        image_folder = city["image_folder"]
        
        # Extract structured metadata for embeddings
        city_meta_text_structured = extract_criteria2(city_metadata, ["description", "weather", "landscape", "transportation", "activities", "cuisine"])
        city_meta_text_embedding = user_preferences_to_embedding(city_meta_text_structured)
        
        # Check if images exist
        image_folder_exists = os.path.exists(image_folder) and os.listdir(image_folder)
        
        # Compute overall embedding
        if image_folder_exists:
            city_embedding = get_user_overall_embedding(
                image_folder_path=image_folder, 
                prompt_path=None, 
                alpha=0.5, beta=0.5
            )
        else:
            city_embedding = city_meta_text_embedding
        
        # Validate embedding
        if city_embedding is not None and not np.isnan(city_embedding).any() and len(city_embedding) > 0:
            new_city_embeddings.append(city_embedding)
            new_city_names.append(city_name)
    
    if new_city_embeddings:
        new_city_embeddings = np.array(new_city_embeddings).astype("float32")
        
        # Load existing FAISS index or create a new one
        index_path = "/Users/apple/Documents/GitHub/Wander-Whisper/API/data/embeddings/city_embeddings_mongo.index"
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            index = faiss.IndexFlatL2(new_city_embeddings.shape[1])
        
        # Add new embeddings to the index
        index.add(new_city_embeddings)
        
        # Save the updated FAISS index
        faiss.write_index(index, index_path)
        
        # Update city names list and save
        existing_city_names.extend(new_city_names)
        with open(city_names_path, "w") as f:
            json.dump(existing_city_names, f)
        
        print("🚀 FAISS index updated with new city embeddings!")
    else:
        print("🔄 No new embeddings to add.")


# Example usage:
mongo_uri = f"mongodb+srv://dtp39:WanderWhisperPassword@wanderwhisperer.18iuu.mongodb.net/WanderWhisper?retryWrites=true&w=majority"
db_name = "WanderWhisper"
collection_name = "top500"

generate_city_embeddings_mongodb(mongo_uri, db_name, collection_name)
# Run the function
#generate_city_embeddings("data/dataset/us_cities.json")
