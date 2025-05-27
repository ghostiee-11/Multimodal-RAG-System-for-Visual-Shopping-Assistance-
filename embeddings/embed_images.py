import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import time

# --- Configuration ---
VALID_IMAGES_CSV_PATH = 'valid_product_images.csv' # Output from manage_image_paths.py

# CLIP model for image embeddings
IMAGE_MODEL_NAME = 'clip-ViT-B-32' 

OUTPUT_IMAGE_EMBEDDINGS_NUMPY = 'image_embeddings_clip.npy'
# We will store the 'full_image_path' as the ID for these embeddings,
# as each image is unique. We also need product_id for linking.
OUTPUT_IMAGE_IDENTIFIERS_NUMPY = 'image_identifiers_clip.npy' # Will store full_image_path
OUTPUT_IMAGE_PRODUCT_IDS_NUMPY = 'image_product_ids_clip.npy' # Product_id for each image

# Global model variable
image_embedding_model = None

def load_valid_image_data():
    """Loads the validated image data CSV."""
    try:
        df = pd.read_csv(VALID_IMAGES_CSV_PATH)
        print(f"Loaded {VALID_IMAGES_CSV_PATH}, shape: {df.shape}")
        # Ensure we only process unique image paths if there are any duplicates in the CSV
        # (though manage_image_paths.py should have handled this for 'full_image_path')
        df.drop_duplicates(subset=['full_image_path'], inplace=True)
        print(f"Shape after ensuring unique image paths: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Valid image data CSV not found at {VALID_IMAGES_CSV_PATH}")
        return None

def initialize_image_embedding_model():
    """Initializes the CLIP model."""
    global image_embedding_model

    if image_embedding_model is None:
        print(f"Loading image embedding model: {IMAGE_MODEL_NAME}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        image_embedding_model = SentenceTransformer(IMAGE_MODEL_NAME, device=device)
        print("Image embedding model loaded.")

def generate_image_embeddings_from_df(df_valid_images):
    """Generates CLIP embeddings for images listed in the DataFrame."""
    global image_embedding_model

    if df_valid_images is None or df_valid_images.empty:
        print("No valid image data provided for embedding.")
        return np.array([]), [], []

    if image_embedding_model is None:
        print("Image embedding model not initialized. Initializing now...")
        initialize_image_embedding_model()

    image_paths_to_embed = df_valid_images['full_image_path'].tolist()
    corresponding_product_ids = df_valid_images['product_id'].tolist() # Keep product_ids aligned
    
    print(f"\nGenerating embeddings for {len(image_paths_to_embed)} unique images...")
    start_time = time.time()

    pil_images = []
    # We will use image_paths_to_embed as the identifiers for the embeddings
    # because full_image_path is unique for each image file.
    
    for img_path in image_paths_to_embed:
        try:
            pil_img = Image.open(img_path).convert("RGB")
            pil_images.append(pil_img)
        except Exception as e:
            print(f"Critical Error: Could not open image {img_path} that was previously validated. Error: {e}")
            # This shouldn't happen if manage_image_paths.py worked correctly and files weren't moved/deleted.
            # We might need to decide how to handle this (e.g., skip and misalign or stop).
            # For now, let's assume it won't happen. If it does, we'll need to debug.
            # A robust way would be to build a new list of paths/pids that were successfully opened.
            return np.array([]), [], [] # Or handle more gracefully

    if not pil_images:
        print("No images could be loaded for embedding.")
        return np.array([]), [], []

    image_embeddings_np = image_embedding_model.encode(
        pil_images, 
        show_progress_bar=True, 
        batch_size=16 # Batch size for images can be smaller than for text due to memory
    )
    
    end_time = time.time()
    print(f"Image embeddings generated successfully in {end_time - start_time:.2f} seconds.")
    print(f"Shape of image embeddings array: {image_embeddings_np.shape}") # (num_images, clip_embedding_dim) -> (120, 512) for ViT-B/32

    return image_embeddings_np, image_paths_to_embed, corresponding_product_ids



if __name__ == '__main__':
    df_valid_images_data = load_valid_image_data()

    if df_valid_images_data is not None and not df_valid_images_data.empty:
        # Initialize model once
        try:
            initialize_image_embedding_model()
        except Exception as e:
            print(f"Failed to initialize image embedding model: {e}")
            exit()

        # CORRECTED LINE HERE:
        img_embeddings, img_identifiers, img_product_ids = generate_image_embeddings_from_df(df_valid_images_data)

        if img_embeddings.size > 0: 
            np.save(OUTPUT_IMAGE_EMBEDDINGS_NUMPY, img_embeddings)
            
            np.save(OUTPUT_IMAGE_IDENTIFIERS_NUMPY, np.array(img_identifiers, dtype=object)) 
            np.save(OUTPUT_IMAGE_PRODUCT_IDS_NUMPY, np.array(img_product_ids, dtype=object))

            print(f"\nSaved image embeddings to {OUTPUT_IMAGE_EMBEDDINGS_NUMPY}")
            print(f"Saved corresponding image identifiers (paths) to {OUTPUT_IMAGE_IDENTIFIERS_NUMPY}")
            print(f"Saved corresponding image product IDs to {OUTPUT_IMAGE_PRODUCT_IDS_NUMPY}")

            # Verification
            print(f"\nVerification: Loaded image embeddings shape: {img_embeddings.shape}")
            print(f"Number of image identifiers: {len(img_identifiers)}")
            print(f"Number of image product IDs: {len(img_product_ids)}")
            if img_embeddings.shape[0] == len(img_identifiers) == len(img_product_ids):
                print("Verification successful: Counts match for embeddings, identifiers, and product_ids.")
            else:
                print("Verification FAILED: Mismatch in counts.")
            
            print("\nStep 2.2: Image Embedding Generation Complete.")
        else:
            print("Image embedding generation failed or produced no embeddings.")
    else:
        print("Failed to load valid image data or no valid images found. Aborting image embedding.")