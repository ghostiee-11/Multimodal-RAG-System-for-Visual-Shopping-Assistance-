import os
import time
from pinecone import Pinecone, ServerlessSpec # Or PodSpec if needed
import numpy as np
# import pandas as pd # Not strictly needed if we only use the .npy files
from dotenv import load_dotenv

# --- Configuration ---
IMAGE_EMBEDDINGS_NUMPY_PATH = 'image_embeddings_clip.npy'
IMAGE_IDENTIFIERS_NUMPY_PATH = 'image_identifiers_clip.npy' # These are full image paths
IMAGE_PRODUCT_IDS_NUMPY_PATH = 'image_product_ids_clip.npy'

# Pinecone Configuration for the new Image Index
IMAGE_INDEX_NAME = "product-image-embeddings" # New, unique index name
IMAGE_EMBEDDING_DIMENSION = 512 # CLIP ViT-B/32 produces 512-dim embeddings
IMAGE_METRIC = "cosine" # Cosine similarity is good for CLIP embeddings

def load_env_vars():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not found in environment variables.")
        return None
    return api_key

def load_image_embedding_data():
    """Loads image embeddings, their identifiers (paths), and associated product_ids."""
    try:
        image_embeddings = np.load(IMAGE_EMBEDDINGS_NUMPY_PATH)
        # Ensure identifiers are loaded as a list of strings
        image_identifiers = np.load(IMAGE_IDENTIFIERS_NUMPY_PATH, allow_pickle=True).tolist()
        image_product_ids = np.load(IMAGE_PRODUCT_IDS_NUMPY_PATH, allow_pickle=True).tolist()
        
        print(f"Loaded image embeddings shape: {image_embeddings.shape}")
        print(f"Loaded {len(image_identifiers)} image identifiers.")
        print(f"Loaded {len(image_product_ids)} image product IDs.")

        if not (image_embeddings.shape[0] == len(image_identifiers) == len(image_product_ids)):
            print("Error: Mismatch in lengths of loaded image data arrays.")
            return None, None, None
        
        return image_embeddings, image_identifiers, image_product_ids

    except FileNotFoundError as e:
        print(f"Error loading image embedding NPY files: {e}")
        return None, None, None

def initialize_pinecone_client(api_key):
    try:
        pc = Pinecone(api_key=api_key)
        print("Pinecone client initialized successfully for image index.")
        return pc
    except Exception as e:
        print(f"Error initializing Pinecone client: {e}")
        return None

def create_pinecone_image_index_if_not_exists(pc_client, index_name, dimension, metric):
    existing_indexes = [index_info["name"] for index_info in pc_client.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Image Index '{index_name}' does not exist. Creating...")
        try:
            pc_client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws", # Verify this from your Pinecone project (aws, gcp, azure)
                    region="us-east-1"  # Verify this region from your Pinecone project
                )
                # pod_spec = PodSpec(environment=os.getenv("PINECONE_ENVIRONMENT"), pod_type="p1", pods=1) # For pod-based
            )
            while not pc_client.describe_index(index_name).status['ready']:
                print("Waiting for image index to be ready...")
                time.sleep(5)
            print(f"Image Index '{index_name}' created and ready.")
        except Exception as e:
            print(f"Error creating Pinecone image index: {e}")
            return None
    else:
        print(f"Image Index '{index_name}' already exists.")
    
    return pc_client.Index(index_name)

def upsert_images_to_pinecone(index_obj, index_name_str, embeddings_data, image_identifiers, image_product_ids):
    if embeddings_data is None or not image_identifiers or not image_product_ids:
        print("Embeddings, identifiers, or product_ids are missing. Cannot proceed.")
        return False

    print(f"\nPreparing image data for upserting to Pinecone index '{index_name_str}'...")
    
    vectors_to_upsert = []
    for i, img_path_id in enumerate(image_identifiers):
        meta = {
            "product_id": str(image_product_ids[i]), # Link to the product
            "image_path": str(img_path_id) # Store the path as metadata too (optional, as it's the ID)
            # We will add "image_caption" here later if we do Step 2.3
        }
        vectors_to_upsert.append({
            "id": str(img_path_id), # Pinecone ID must be a string; image path is a good unique ID
            "values": embeddings_data[i].tolist(), # Embedding vector
            "metadata": meta
        })

    batch_size = 100 
    num_items = len(vectors_to_upsert)
    print(f"Upserting {num_items} image items in batches of {batch_size}...")

    for i in range(0, num_items, batch_size):
        batch = vectors_to_upsert[i : i + batch_size]
        try:
            print(f"Upserting image batch {i//batch_size + 1}/{(num_items + batch_size -1)//batch_size} ({len(batch)} items)")
            index_obj.upsert(vectors=batch)
        except Exception as e:
            print(f"Error upserting image batch: {e}")
            return False
    
    print("Image upsert operation complete.")
    time.sleep(10) 
    try:
        stats = index_obj.describe_index_stats()
        print(f"Image Index '{index_name_str}' now has approximately {stats.total_vector_count} vectors.")
    except Exception as e:
        print(f"Could not fetch image index stats: {e}")
    return True

if __name__ == '__main__':
    pinecone_api_key = load_env_vars()
    if not pinecone_api_key:
        exit()

    pc_client = initialize_pinecone_client(pinecone_api_key)
    if not pc_client:
        exit()

    img_embeddings, img_ids, img_pids = load_image_embedding_data()

    if img_embeddings is not None and img_ids is not None and img_pids is not None:
        pinecone_image_index = create_pinecone_image_index_if_not_exists(
            pc_client, IMAGE_INDEX_NAME, IMAGE_EMBEDDING_DIMENSION, IMAGE_METRIC
        )
        
        if pinecone_image_index:
            success = upsert_images_to_pinecone(
                pinecone_image_index, IMAGE_INDEX_NAME, img_embeddings, img_ids, img_pids
            )
            if success:
                print("\nStep 2.4: Pinecone Image Vector Database Setup and Population Complete.")

                # --- Example Query (for testing the image index) ---
                print("\n--- Example Image Index Query ---")
                try:
                    stats = pinecone_image_index.describe_index_stats()
                    if stats.total_vector_count > 0:
                        # Query with the embedding of the first image in our dataset
                        sample_query_image_vector = img_embeddings[0].tolist() 
                        queried_image_path = img_ids[0]
                        print(f"Querying with embedding of image: {queried_image_path}")
                        
                        query_response = pinecone_image_index.query(
                            vector=sample_query_image_vector,
                            top_k=3, # Find top 3 similar images
                            include_metadata=True
                        )
                        
                        print("\nImage Query Results:")
                        for match in query_response['matches']:
                            print(f"  ID (Image Path): {match['id']}")
                            print(f"  Score: {match['score']:.4f}")
                            print(f"  Metadata (Product ID): {match['metadata'].get('product_id')}")
                            print("-" * 20)
                    else:
                        print("Image index is empty or stats not updated yet, skipping example query.")
                except Exception as e:
                    print(f"Error during example image query or fetching stats: {e}")
            else:
                print("Failed to upsert data to Pinecone image index.")
        else:
            print("Failed to create or connect to Pinecone image index.")
    else:
        print("Failed to load necessary image embedding data. Aborting Pinecone image DB setup.")