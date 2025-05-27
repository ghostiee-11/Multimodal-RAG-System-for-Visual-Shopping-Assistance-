import os
import time
from pinecone import Pinecone
import pandas as pd
from dotenv import load_dotenv
import json # For parsing JSON string of captions

# --- Configuration ---
CAPTIONS_CSV_PATH = 'image_captions_multiple.csv' # Use the new CSV
IMAGE_INDEX_NAME = "product-image-embeddings"

# ... (load_env_vars, initialize_pinecone_client, connect_to_image_index are same) ...
def load_env_vars():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not found in environment variables.")
        return None
    return api_key

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

def load_image_captions(): # Modified to load JSON list
    try:
        df_captions = pd.read_csv(CAPTIONS_CSV_PATH)
        # Ensure the JSON string column is not null before trying to parse
        df_captions.dropna(subset=['generated_captions_json'], inplace=True)
        # Parse the JSON string into a list of strings
        df_captions['generated_captions_list'] = df_captions['generated_captions_json'].apply(json.loads)
        print(f"Loaded and parsed {len(df_captions)} caption entries from {CAPTIONS_CSV_PATH}")
        return df_captions
    except FileNotFoundError:
        print(f"Error: Captions CSV not found at {CAPTIONS_CSV_PATH}")
        return None
    except Exception as e:
        print(f"Error loading/parsing captions CSV: {e}")
        return None

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
            captions_list_val = row['generated_captions_list'] # This is now a list

            # Ensure captions_list_val is a list of strings
            if not isinstance(captions_list_val, list) or not all(isinstance(cap, str) for cap in captions_list_val):
                print(f"  Warning: Captions for ID {image_id} is not a list of strings. Skipping. Value: {captions_list_val}")
                failed_updates.append(image_id)
                continue
            if not captions_list_val: # If empty list after json.loads
                captions_list_val = ["No valid caption generated."]


            try:
                metadata_to_set = {
                    "product_id": product_id_val,
                    "image_path": image_id, 
                    "generated_captions_list": captions_list_val, # Store the list
                    "primary_caption": captions_list_val[0] if captions_list_val else "N/A" # Also store the first one for easy access
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
        time.sleep(1) # Adjust if needed

    print(f"\nMetadata update complete. Successfully updated: {updated_count} vectors.")
    if failed_updates:
        print(f"Failed to update/skipped metadata for {len(failed_updates)} IDs.") # {failed_updates[:10]}...
    return updated_count > 0 # Consider it a success if at least some updates went through


if __name__ == '__main__':
    # ... (same setup as your existing script) ...
    pinecone_api_key = load_env_vars()
    if not pinecone_api_key: exit()
    pc_client = initialize_pinecone_client(pinecone_api_key)
    if not pc_client: exit()
    df_captions = load_image_captions()
    if df_captions is None or df_captions.empty: print("No captions loaded. Exiting."); exit()
    pinecone_image_index_obj = connect_to_image_index(pc_client)
    
    if pinecone_image_index_obj:
        success = update_metadata_in_pinecone(pinecone_image_index_obj, df_captions)
        if success:
            print("\nSuccessfully updated image index metadata with multiple captions.")
            # ... (verification logic as before, checking for 'generated_captions_list') ...
            print("\n--- Verifying metadata for a sample image ---")
            if not df_captions.empty:
                sample_image_id_to_fetch = str(df_captions.iloc[0]['full_image_path'])
                try:
                    fetch_response = pinecone_image_index_obj.fetch(ids=[sample_image_id_to_fetch])
                    if fetch_response and hasattr(fetch_response, 'vectors') and sample_image_id_to_fetch in fetch_response.vectors:
                        vector_data = fetch_response.vectors[sample_image_id_to_fetch]
                        if hasattr(vector_data, 'metadata') and vector_data.metadata:
                            print(f"Fetched metadata for ID '{sample_image_id_to_fetch}':")
                            print(f"  Product ID: {vector_data.metadata.get('product_id')}")
                            print(f"  Captions List: {vector_data.metadata.get('generated_captions_list')}")
                            if 'generated_captions_list' in vector_data.metadata and isinstance(vector_data.metadata['generated_captions_list'], list):
                                print("Multiple captions successfully added to metadata!")
                            else:
                                print("Multiple captions NOT found or not a list in fetched metadata.")
                        else: print(f"No metadata found for ID '{sample_image_id_to_fetch}'.")
                    else: print(f"Could not fetch vector for ID '{sample_image_id_to_fetch}'.")
                except Exception as e: print(f"Error fetching sample vector: {e}")
        else: print("Failed to update all image metadata with captions.")
    else: print("Failed to connect to Pinecone image index.")