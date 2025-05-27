import os
import time
from pinecone import Pinecone
import pandas as pd
from dotenv import load_dotenv

# --- Configuration ---
CAPTIONS_CSV_PATH = 'image_captions.csv'
IMAGE_INDEX_NAME = "product-image-embeddings"

def load_env_vars():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not found in environment variables.")
        return None
    return api_key

def load_image_captions():
    try:
        df_captions = pd.read_csv(CAPTIONS_CSV_PATH)
        df_captions.dropna(subset=['generated_caption'], inplace=True)
        df_captions['generated_caption'] = df_captions['generated_caption'].astype(str)
        print(f"Loaded {len(df_captions)} captions from {CAPTIONS_CSV_PATH}")
        return df_captions
    except FileNotFoundError:
        print(f"Error: Captions CSV not found at {CAPTIONS_CSV_PATH}")
        return None
    except Exception as e:
        print(f"Error loading captions CSV: {e}")
        return None

def initialize_pinecone_client(api_key):
    try:
        pc = Pinecone(api_key=api_key)
        print("Pinecone client initialized successfully.")
        return pc
    except Exception as e:
        print(f"Error initializing Pinecone client: {e}")
        return None

def connect_to_image_index(pc_client):
    existing_indexes = [index_info["name"] for index_info in pc_client.list_indexes()]
    if IMAGE_INDEX_NAME not in existing_indexes:
        print(f"Error: Image Index '{IMAGE_INDEX_NAME}' does not exist. Please create it first.")
        return None
    print(f"Connecting to Pinecone image index: {IMAGE_INDEX_NAME}...")
    index_obj = pc_client.Index(IMAGE_INDEX_NAME)
    print(f"Connected to image index. Current vector count: {index_obj.describe_index_stats().total_vector_count}")
    return index_obj

def update_metadata_in_pinecone(index_obj, df_captions_data):
    if index_obj is None or df_captions_data is None or df_captions_data.empty:
        print("Pinecone index or captions data is missing/empty. Cannot update.")
        return False

    print(f"\nStarting metadata update for {len(df_captions_data)} image vectors in index '{IMAGE_INDEX_NAME}'...")
    
    batch_size = 50 
    updated_count = 0
    failed_updates = []

    for i in range(0, len(df_captions_data), batch_size):
        batch_df = df_captions_data.iloc[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(df_captions_data) + batch_size -1)//batch_size}...")
        
        for _, row in batch_df.iterrows():
            image_id = str(row['full_image_path']) 
            product_id_val = str(row['product_id'])
            caption_val = str(row['generated_caption'])

            try:
                metadata_to_set = {
                    "product_id": product_id_val,
                    "image_path": image_id, 
                    "generated_caption": caption_val
                }
                index_obj.update(
                    id=image_id,
                    set_metadata=metadata_to_set
                )
                updated_count += 1
            except Exception as e:
                print(f"  Error updating metadata for ID {image_id}: {e}")
                failed_updates.append(image_id)
        
        print(f"  Batch processed. Sleeping for a moment to respect rate limits if any...")
        time.sleep(1)

    print(f"\nMetadata update complete. Successfully updated: {updated_count} vectors.")
    if failed_updates:
        print(f"Failed to update metadata for {len(failed_updates)} IDs: {failed_updates}")
    return updated_count == len(df_captions_data)


if __name__ == '__main__':
    pinecone_api_key = load_env_vars()
    if not pinecone_api_key:
        exit()

    pc_client = initialize_pinecone_client(pinecone_api_key)
    if not pc_client:
        exit()

    df_captions = load_image_captions()
    if df_captions is None or df_captions.empty:
        print("No captions loaded. Exiting.")
        exit()

    pinecone_image_index_obj = connect_to_image_index(pc_client)
    
    if pinecone_image_index_obj:
        success = update_metadata_in_pinecone(pinecone_image_index_obj, df_captions)
        
        if success:
            print("\nSuccessfully updated image index metadata with captions.")

            # --- Verifying metadata for a sample image (CORRECTED) ---
            print("\n--- Verifying metadata for a sample image ---")
            if not df_captions.empty:
                sample_image_id_to_fetch = str(df_captions.iloc[0]['full_image_path'])
                try:
                    # FetchResponse is a Pydantic model, access attributes directly
                    fetch_response = pinecone_image_index_obj.fetch(ids=[sample_image_id_to_fetch])
                    
                    # Check if vectors were returned and if our specific ID is in the vectors dictionary
                    if fetch_response and hasattr(fetch_response, 'vectors') and sample_image_id_to_fetch in fetch_response.vectors:
                        vector_data = fetch_response.vectors[sample_image_id_to_fetch] # This is a Vector object
                        
                        if hasattr(vector_data, 'metadata') and vector_data.metadata:
                            print(f"Fetched metadata for ID '{sample_image_id_to_fetch}':")
                            print(vector_data.metadata) # Pydantic model, access as dict or attributes
                            if 'generated_caption' in vector_data.metadata:
                                print("Caption successfully added to metadata!")
                            else:
                                print("Caption NOT found in fetched metadata.")
                        else:
                            print(f"No metadata found for ID '{sample_image_id_to_fetch}' in the fetched vector.")
                    else:
                        print(f"Could not fetch vector data for ID '{sample_image_id_to_fetch}'. Response: {fetch_response}")
                except Exception as e:
                    print(f"Error fetching sample vector or accessing its data: {e}")
        else:
            print("Failed to update all image metadata with captions.")
    else:
        print("Failed to connect to Pinecone image index.")