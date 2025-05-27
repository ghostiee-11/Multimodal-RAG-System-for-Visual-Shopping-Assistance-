# setup_text_vector_db_pinecone.py
import os
import time
from pinecone import Pinecone, ServerlessSpec, PodSpec # Ensure PodSpec is imported if you might use it
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --- Configuration ---
TEXT_EMBEDDINGS_NUMPY_PATH = 'text_embeddings_mpnet.npy' 
TEXT_IDS_NUMPY_PATH = 'text_chunk_ids_mpnet.npy'     
TEXT_CHUNKS_CSV_FOR_METADATA = 'prepared_text_chunks.csv' # Source of metadata for each chunk

# Pinecone Configuration for the Text Index
PINECONE_INDEX_NAME = "product-text-embeddings" 
TEXT_EMBEDDING_DIMENSION = 768 # all-mpnet-base-v2 produces 768-dim embeddings
TEXT_METRIC = "cosine" # Cosine similarity is good for sentence embeddings

def load_env_vars():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    # environment = os.getenv("PINECONE_ENVIRONMENT") # Needed for pod-based indexes
    if not api_key:
        print("Error: PINECONE_API_KEY not found in environment variables.")
        return None, None
    # if not environment and "serverless" not in PINECONE_INDEX_NAME.lower(): # Or some other flag indicating serverless
    #     print("Error: PINECONE_ENVIRONMENT not found for pod-based index.")
    #     return None, None
    return api_key #, environment (return if using podspec)

def initialize_pinecone_client(api_key):
    try:
        pc = Pinecone(api_key=api_key)
        print("Pinecone client initialized successfully for text index.")
        return pc
    except Exception as e:
        print(f"Error initializing Pinecone client: {e}")
        return None

def create_pinecone_text_index_if_not_exists(pc_client, index_name, dimension, metric):
    """
    Checks if the index exists. If not, creates a new Serverless index.
    Modify spec for PodSpec if using a pod-based environment.
    """
    existing_indexes = [index_info["name"] for index_info in pc_client.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Text Index '{index_name}' does not exist. Creating...")
        try:
            # For Serverless:
            pc_client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"), # Default to aws, can be set in .env
                    region=os.getenv("PINECONE_REGION", "us-east-1") # Default region
                )
            )
            # For Pod-based (example):
            # pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
            # if not pinecone_environment:
            #     raise ValueError("PINECONE_ENVIRONMENT must be set for pod-based indexes.")
            # pc_client.create_index(
            #     name=index_name,
            #     dimension=dimension,
            #     metric=metric,
            #     spec=PodSpec(
            #         environment=pinecone_environment,
            #         pod_type="p1.x1" # Choose appropriate pod type
            #     )
            # )
            
            # Wait for the index to be ready
            while not pc_client.describe_index(index_name).status['ready']:
                print(f"Waiting for text index '{index_name}' to be ready...")
                time.sleep(5)
            print(f"Text Index '{index_name}' created and ready.")
        except Exception as e:
            print(f"Error creating Pinecone text index '{index_name}': {e}")
            return None
    else:
        print(f"Text Index '{index_name}' already exists.")
    
    return pc_client.Index(index_name)

def load_text_embedding_data(embeddings_path, ids_path):
    """Loads text embeddings and their corresponding chunk_ids."""
    try:
        embeddings = np.load(embeddings_path)
        # Ensure chunk_ids are loaded as a list of strings
        chunk_ids = np.load(ids_path, allow_pickle=True).tolist()
        # Convert all IDs to string just in case
        chunk_ids = [str(id_val) for id_val in chunk_ids]

        print(f"Loaded text embeddings from {embeddings_path}, shape: {embeddings.shape}")
        print(f"Loaded {len(chunk_ids)} text chunk IDs from {ids_path}.")

        if not (embeddings.shape[0] == len(chunk_ids)):
            print("Error: Mismatch in lengths of loaded text embeddings and chunk_ids.")
            return None, None
        
        return embeddings, chunk_ids
    except FileNotFoundError as e:
        print(f"Error loading text embedding NPY files: {e}")
        return None, None
    except Exception as e_load:
        print(f"An unexpected error occurred while loading NPY files: {e_load}")
        return None, None


def load_metadata_source(csv_path):
    """Loads the CSV containing all text chunk details for metadata."""
    try:
        df = pd.read_csv(csv_path)
        # Ensure text_chunk_id is string for matching
        df['text_chunk_id'] = df['text_chunk_id'].astype(str)
        print(f"Successfully loaded metadata source: {csv_path}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Metadata source file '{csv_path}' not found.")
        return None
    except Exception as e_csv:
        print(f"Error loading metadata CSV '{csv_path}': {e_csv}")
        return None


def upsert_text_data_to_pinecone(index_obj, index_name_str, embeddings_data, chunk_ids, df_metadata_source):
    if embeddings_data is None or not chunk_ids or df_metadata_source is None:
        print("Embeddings, chunk_ids, or metadata source DataFrame are missing. Cannot proceed with upsert.")
        return False

    print(f"\nPreparing text data for upserting to Pinecone index '{index_name_str}'...")
    
    vectors_to_upsert = []
    missing_metadata_count = 0

    for i, chunk_id_str in enumerate(chunk_ids):
        # Retrieve the row from df_metadata_source corresponding to the chunk_id
        metadata_row = df_metadata_source[df_metadata_source['text_chunk_id'] == chunk_id_str]
        
        meta = {}
        if not metadata_row.empty:
            chunk_info = metadata_row.iloc[0]
            meta = {
                "product_id": str(chunk_info.get("product_id", "N/A")),
                # original_doc_id for OCR text is the full_image_path, for others it's their source doc_id
                "original_doc_id": str(chunk_info.get("original_doc_id", "N/A")), 
                "text_type": str(chunk_info.get("text_type", "N/A")),
                "aspect": str(chunk_info.get("aspect", "N/A")),
                "sentiment": str(chunk_info.get("sentiment", "N/A")),
                # Store a snippet for quick reference, ensure it's not too long for Pinecone metadata limits
                "text_content": str(chunk_info.get("text_content", ""))[:450] # Max metadata value size can be an issue
            }
            # Specifically add image_filename if the chunk is OCR-derived
            if chunk_info.get("text_type") == 'image_ocr_text':
                if pd.notna(chunk_info.get('image_filename')):
                    meta['image_filename_source'] = str(chunk_info.get('image_filename'))
                elif pd.notna(chunk_info.get('original_doc_id')): # Fallback if 'image_filename' was missed
                    try:
                        meta['image_filename_source'] = os.path.basename(str(chunk_info.get('original_doc_id')))
                    except Exception: # Handle cases where original_doc_id might not be a path
                         meta['image_filename_source'] = str(chunk_info.get('original_doc_id'))
            
            # You can add other relevant fields from prepared_text_chunks.csv here
            # e.g., meta['source_file'] = str(chunk_info.get("source_file", "N/A"))

        else:
            print(f"Warning: Metadata not found for chunk_id '{chunk_id_str}'. Using minimal metadata.")
            meta = {"text_content": "Full metadata missing, see original text chunk."} # Placeholder
            missing_metadata_count +=1
            
        vectors_to_upsert.append({
            "id": chunk_id_str, # Pinecone ID must be a string
            "values": embeddings_data[i].tolist(), # Embedding vector
            "metadata": meta
        })

    if missing_metadata_count > 0:
        print(f"Warning: Metadata was missing for {missing_metadata_count} chunk(s).")

    batch_size = 100 # Pinecone recommends batch sizes up to 100 for general purpose, or 1MB payload
    num_items = len(vectors_to_upsert)
    print(f"Upserting {num_items} text items in batches of {batch_size}...")

    for i_batch in range(0, num_items, batch_size):
        batch = vectors_to_upsert[i_batch : i_batch + batch_size]
        try:
            print(f"Upserting text batch {i_batch//batch_size + 1}/{(num_items + batch_size -1)//batch_size} ({len(batch)} items)")
            index_obj.upsert(vectors=batch)
        except Exception as e:
            print(f"Error upserting text batch: {e}")
            # You might want to implement retries or save failed batches for later.
            # For now, we'll report and potentially stop.
            return False # Or continue and report summary of failures
    
    print("Text upsert operation complete.")
    time.sleep(10) # Give Pinecone a moment to update stats
    try:
        stats = index_obj.describe_index_stats()
        print(f"Text Index '{index_name_str}' now has approximately {stats.total_vector_count} vectors.")
    except Exception as e:
        print(f"Could not fetch text index stats after upsert: {e}")
    return True

if __name__ == '__main__':
    pinecone_api_key_main = load_env_vars()
    if not pinecone_api_key_main: # load_env_vars now returns only api_key
        exit()

    pc_client = initialize_pinecone_client(pinecone_api_key_main)
    if not pc_client:
        exit()

    # Load text embeddings and their IDs
    text_embeddings, text_chunk_ids = load_text_embedding_data(
        TEXT_EMBEDDINGS_NUMPY_PATH, TEXT_IDS_NUMPY_PATH
    )
    if text_embeddings is None or text_chunk_ids is None:
        print("Failed to load text embeddings or chunk IDs. Aborting.")
        exit()

    # Load the full metadata source CSV
    df_all_chunks_metadata = load_metadata_source(TEXT_CHUNKS_CSV_FOR_METADATA)
    if df_all_chunks_metadata is None:
        print("Failed to load metadata source CSV. Aborting.")
        exit()
    
    # Create or connect to the Pinecone index for text
    # Set create_index=True if you want this script to handle creation.
    # Otherwise, ensure the index exists.
    # For this example, let's assume we always want to try creating if not exists.
    pinecone_text_index = create_pinecone_text_index_if_not_exists(
        pc_client, PINECONE_INDEX_NAME, TEXT_EMBEDDING_DIMENSION, TEXT_METRIC
    )
        
    if pinecone_text_index:
        print(f"Preparing to upsert data into text index: {PINECONE_INDEX_NAME}")
        success = upsert_text_data_to_pinecone(
            pinecone_text_index, 
            PINECONE_INDEX_NAME,
            text_embeddings, 
            text_chunk_ids, 
            df_all_chunks_metadata
        )
        if success:
            print(f"\nSuccessfully upserted text data (including OCR) to Pinecone index '{PINECONE_INDEX_NAME}'.")

            # --- Example Query (for testing the text index) ---
            print("\n--- Example Text Index Query (testing one of the uploaded vectors) ---")
            try:
                stats = pinecone_text_index.describe_index_stats()
                if stats.total_vector_count > 0 and len(text_embeddings) > 0:
                    sample_query_vector = text_embeddings[0].tolist() # Use the first embedding as a sample query
                    sample_chunk_id = text_chunk_ids[0]
                    
                    # Fetch the metadata for this sample chunk to see what we're querying for
                    sample_chunk_metadata_info = df_all_chunks_metadata[df_all_chunks_metadata['text_chunk_id'] == sample_chunk_id]
                    if not sample_chunk_metadata_info.empty:
                        print(f"Querying with vector of chunk_id: '{sample_chunk_id}'")
                        print(f"  Text content of query chunk: \"{sample_chunk_metadata_info.iloc[0]['text_content'][:100]}...\"")
                        print(f"  Text type of query chunk: {sample_chunk_metadata_info.iloc[0]['text_type']}")
                    else:
                        print(f"Querying with vector of chunk_id: '{sample_chunk_id}' (metadata not found for display)")

                    query_response = pinecone_text_index.query(
                        vector=sample_query_vector,
                        top_k=3, 
                        include_metadata=True
                    )
                    
                    print("\nText Query Results:")
                    for match in query_response['matches']:
                        print(f"  ID (Chunk ID): {match['id']}")
                        print(f"  Score: {match['score']:.4f}")
                        print(f"  Metadata: {match['metadata']}")
                        print("-" * 20)
                else:
                    print("Text index is empty or no embeddings loaded, skipping example query.")
            except Exception as e:
                print(f"Error during example text query or fetching stats: {e}")
        else:
            print(f"Failed to upsert all data to Pinecone text index '{PINECONE_INDEX_NAME}'.")
    else:
        print(f"Failed to create or connect to Pinecone text index '{PINECONE_INDEX_NAME}'.")