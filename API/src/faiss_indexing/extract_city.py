import os
import faiss
import json
import numpy as np
import sys

# Add the src folder to the system path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from embedding_extract.implicit_user_embedding import get_user_overall_embedding
import datetime

# Get the absolute path of the current working directory (terminal location)
SCRIPT_DIR = os.getcwd()

def recommend_cities(user_embedding, top_k=None):
    """
    Finds the most similar city embeddings using FAISS and prints similarity scores for all cities.

    Args:
        user_embedding (np.array): The final user embedding vector.
        top_k (int, optional): Number of top cities to retrieve. If None, shows all cities.

    Returns:
        List of recommended city names with similarity scores.
    """
    # Dynamically resolve the path for the FAISS index
    index_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'embeddings', 'city_embeddings.index')    
    index = faiss.read_index(index_path)

    # Ensure user embedding matches FAISS index dimension
    user_embedding = np.array(user_embedding).astype("float32").reshape(1, -1)

    if user_embedding.shape[1] < index.d:
        user_embedding = np.pad(user_embedding, ((0, 0), (0, index.d - user_embedding.shape[1])), mode='constant')
    elif user_embedding.shape[1] > index.d:
        user_embedding = user_embedding[:, :index.d]

    # Search FAISS for all city embeddings
    distances, indices = index.search(user_embedding, index.ntotal)  # Retrieve all cities

    # Convert L2 distances to similarity scores (1 / (1 + distance))
    similarity_scores = 1 / (1 + distances[0])


    # Load city names
    city_names_path = os.path.join(SCRIPT_DIR, "data/embeddings/city_names.json")
    print("PTH: " + city_names_path)
    with open(city_names_path, "r") as f:
        city_names = json.load(f)

    # Pair city names with similarity scores
    city_scores = [(city_names[idx], similarity_scores[i]) for i, idx in enumerate(indices[0])]

    # Sort by similarity score (descending order)
    city_scores = sorted(city_scores, key=lambda x: x[1], reverse=True)

    # Return top-k cities if specified
    if top_k:
        return city_scores[:top_k]
    return city_scores

def explanation(city_name):
    """
    Provides an explanation for the recommendation.

    Args:
        city_name (str): Name of the recommended city.

    Returns:
        Explanation for the recommendation.
    """
    # Dynamically resolve the path for the city explanations
    city_explanations_path = resolve_file_path("/data/embeddings/city_explanations.json")
    with open(city_explanations_path, "r") as f:
        city_explanations = json.load(f)

    # Return explanation for the recommended city
    return city_explanations[city_name]

def get_recommendations_with_time(image_folder_path, prompt, alpha, beta, top_k=5):
    """
    Generates city recommendations based on user embedding and records the running time.

    Args:
        image_folder_path (str): Path to the folder containing user images.
        prompt_path (str): Path to the folder containing tokenized prompts.
        alpha (float): Weight for image embeddings.
        beta (float): Weight for prompt embeddings.
        top_k (int, optional): Number of top cities to retrieve. Defaults to 5.

    Returns:
        Tuple containing the list of recommended cities with similarity scores and the running time.
    """
    start = datetime.datetime.now()

    user_embedding = get_user_overall_embedding(image_folder_path, prompt, alpha, beta)
    recommendations = recommend_cities(user_embedding, top_k=top_k)

    end = datetime.datetime.now()
    running_time = end - start

    return recommendations, running_time

# Example Usage
# #image_folder_path = os.path.abspath(os.path.join(SCRIPT_DIR, "data/images"))
# image_folder_path = "/home/derrick/Documents/Wander Whisper/Wander-Whisper/data/images"
# prompt = "I am departing from Toronto, Canada in July and will return in August. My budget is adventure travel budget ($1,000 - $3,000 for guided tours), and I prefer local delicacies. I will be traveling solo for one week, and I enjoy hiking. I prefer a mountainous destination with cool ocean breeze weather. I will travel via high-speed train and prefer to use local currency for transactions. My accommodation choice is eco-lodge, and my transportation preference is walking. I want an adventure experience with wildlife conservation focus. My trip should be extreme adventure, and I love indigenous culture. I am interested in Carnival in Rio and will need full travel insurance. I prefer locations with female-friendly and wheelchair access support. For nightlife, I prefer casual bars, and my adventure level is high. I will also be adding guided city tours to my trip."
# alpha = 0.5
# beta = 0.5
# top_k = 5

# recommendations, running_time = get_recommendations_with_time(image_folder_path, prompt, alpha, beta, top_k)

# print("\n**Top Recommended Cities:**")
# for city, score in recommendations:
#     print(f"{city} - Similarity Score: {score*100:.2f}/100")
#     #print(f"Explanation: {explanation(city)}\n")

# print("Time taken:", running_time)
