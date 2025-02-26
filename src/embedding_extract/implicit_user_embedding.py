import os
#from datasets import load_from_disk
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.embedding_extract.image_embeddings_extraction import extract_clip_image_embeddings
from src.model.evaluate import evaluate_t5

# Get the absolute path of the current script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_user_overall_embedding(image_folder_path, prompt, alpha, beta):
    """
    Extracts user overall embedding by running image and text embedding extraction in parallel.
    If either image or text folder is missing, only the available embedding is used.
    """
    start_time = datetime.now()
    image_embedding, text_embedding = None, None

    # Convert relative paths to absolute paths
    image_folder_path = os.path.abspath(os.path.join(SCRIPT_DIR, image_folder_path))
    # Extract image embedding only if the folder exists
    if os.path.exists(image_folder_path):
        def extract_image_embedding():
            return extract_clip_image_embeddings(image_folder_path)
    else:
        extract_image_embedding = None  # No image embedding
        print("No image folder found")
    # Extract text embedding only if the text dataset exists
    if prompt:
        def extract_text_embedding():
            return evaluate_t5(prompt)
        print(prompt)
    else:
        extract_text_embedding = None  # No text embedding
        print("No prompt provided")

    # Run tasks in parallel if both exist
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_image = executor.submit(extract_image_embedding) if extract_image_embedding else None
        future_text = executor.submit(extract_text_embedding) if extract_text_embedding else None

        image_embedding = future_image.result() if future_image else None
        text_embedding = future_text.result() if future_text else None

    #image_embedding = extract_image_embedding()
    #text_embedding = extract_text_embedding()
    # If only one type of embedding is available, return it directly
    if image_embedding is None and text_embedding is not None:
        return normalize_embedding(text_embedding)
    elif text_embedding is None and image_embedding is not None:
        return normalize_embedding(image_embedding)
    elif image_embedding is None and text_embedding is None:
        raise ValueError("❌ No valid image or text embeddings found!")

    # Normalize embeddings
    image_embedding = normalize_embedding(image_embedding)
    text_embedding = normalize_embedding(text_embedding)
    print(image_embedding.shape, text_embedding.shape)
    # Ensure both embeddings have the same dimension
    pad_image_embedding, pad_text_embedding = pad_embeddings(image_embedding, text_embedding)

    # Compute final user embedding
    final_user_embedding = compute_final_user_embedding(alpha, beta, pad_image_embedding, pad_text_embedding)
    
    return final_user_embedding


def normalize_embedding(embedding):
    """ Normalizes an embedding to unit length. """
    norm = np.linalg.norm(embedding, keepdims=True)
    return embedding / norm
def pad_embeddings(image_embedding, text_embedding):
    """
    Pads the smaller embedding with zeros to match the larger embedding's dimension.
    """
    img_dim = image_embedding.shape[0]
    text_dim = text_embedding.shape[0]

    dominant_dim = max(img_dim, text_dim)

    padded_image = np.pad(image_embedding, (0, dominant_dim - img_dim), mode='constant') if img_dim < dominant_dim else image_embedding
    padded_text = np.pad(text_embedding, (0, dominant_dim - text_dim), mode='constant') if text_dim < dominant_dim else text_embedding

    return padded_image, padded_text
def compute_final_user_embedding(alpha, beta, pad_image_embedding, pad_text_embedding):
    """ Computes the final weighted user embedding. """
    return alpha * pad_image_embedding + beta * pad_text_embedding


# Example usage
if __name__ == "__main__":
    image_folder_path = "../../data/images"
    prompt = "I am departing from Toronto, Canada in July and will return in August. My budget is adventure travel budget ($1,000 - $3,000 for guided tours), and I prefer local delicacies. I will be traveling solo for one week, and I enjoy hiking. I prefer a mountainous destination with cool ocean breeze weather. I will travel via high-speed train and prefer to use local currency for transactions. My accommodation choice is eco-lodge, and my transportation preference is walking. I want an adventure experience with wildlife conservation focus. My trip should be extreme adventure, and I love indigenous culture. I am interested in Carnival in Rio and will need full travel insurance. I prefer locations with female-friendly and wheelchair access support. For nightlife, I prefer casual bars, and my adventure level is high. I will also be adding guided city tours to my trip."
    
    alpha = 0.5
    beta = 0.5
    start_time = datetime.now()
    final_user_embedding = get_user_overall_embedding(image_folder_path, prompt, alpha, beta)

    print("Final user embedding shape:", final_user_embedding.shape)
    print("Total time taken:", datetime.now() - start_time)
