import os
import clip
import torch
import numpy as np
from PIL import Image

def extract_clip_image_embeddings(image_folder, model_name="ViT-B/16", device=None):
    """
    Extracts CLIP image embeddings from all images in a given folder and returns the aggregated 512D embedding.

    Args:
        image_folder (str): Path to the folder containing images.
        model_name (str): CLIP model variant to use. Default is 'ViT-B/32'.
        device (str, optional): Device to use ('cuda' or 'cpu'). Default is auto-detect.

    Returns:
        np.ndarray: Aggregated 512D image embedding (mean-pooled across all images).
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model
    model, preprocess = clip.load(model_name, device)

    # Initialize list to store embeddings
    image_embeddings = []

    # Process each image in the folder
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)

        try:
            # Open and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)

            # Compute the image embedding
            with torch.no_grad():
                image_feature = model.encode_image(image_input)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)  # Normalize embedding

            # Convert to numpy and store
            image_embeddings.append(image_feature.cpu().numpy().flatten())

            #print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Aggregate embeddings (mean-pooling across all images)
    if image_embeddings:
        aggregated_embedding = np.mean(image_embeddings, axis=0)  # Shape: (512,)
    else:
        aggregated_embedding = None

    return aggregated_embedding

# Example usage
image_folder_path = "/home/derrick/Documents/Wander Whisper/Wander-Whisper/data/images"
image_embedding_vector = extract_clip_image_embeddings(image_folder_path)
print(image_embedding_vector.shape)
