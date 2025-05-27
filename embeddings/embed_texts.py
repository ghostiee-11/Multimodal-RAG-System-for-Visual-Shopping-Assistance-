import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import time # To estimate processing time
import torch # For checking CUDA

# --- Configuration ---
PREPARED_TEXT_CHUNKS_CSV = 'prepared_text_chunks.csv'
OUTPUT_EMBEDDINGS_NUMPY = 'text_embeddings_mpnet.npy' 
OUTPUT_IDS_NUMPY = 'text_chunk_ids_mpnet.npy'     


MODEL_NAME = 'all-mpnet-base-v2' 


def load_text_chunks(csv_path):
    """Loads the prepared text chunks from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {csv_path}, shape: {df.shape}")
        df['text_content'] = df['text_content'].astype(str).fillna('')
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None

def generate_embeddings(df_texts, model_name):
    """Generates embeddings for the text_content column of the DataFrame."""
    if df_texts is None or 'text_content' not in df_texts.columns:
        print("DataFrame is None or 'text_content' column is missing.")
        return None, None

    print(f"\nLoading sentence transformer model: {model_name}...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SentenceTransformer(model_name, device=device)
    print("Model loaded.")

    texts_to_embed = df_texts['text_content'].tolist()
    
    print(f"\nGenerating embeddings for {len(texts_to_embed)} text chunks...")
    print(f"This might take a while depending on the number of texts, model size, and your hardware.")
    
    start_time = time.time()
    
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, batch_size=32) 
    end_time = time.time()
    
    print(f"Embeddings generated successfully in {end_time - start_time:.2f} seconds.")
    print(f"Shape of embeddings array: {embeddings.shape}") 

    return embeddings, df_texts['text_chunk_id'].tolist()


if __name__ == '__main__':
    # 1. Load the prepared text chunks
    df_retrievable_texts = load_text_chunks(PREPARED_TEXT_CHUNKS_CSV)

    if df_retrievable_texts is not None:
        # 2. Generate embeddings
        text_embeddings, text_ids = generate_embeddings(df_retrievable_texts, MODEL_NAME)

        if text_embeddings is not None:
            # Save embeddings and corresponding IDs as NumPy arrays
            np.save(OUTPUT_EMBEDDINGS_NUMPY, text_embeddings)
            np.save(OUTPUT_IDS_NUMPY, np.array(text_ids)) 
            print(f"\nSaved text embeddings to {OUTPUT_EMBEDDINGS_NUMPY}")
            print(f"Saved corresponding text_chunk_ids to {OUTPUT_IDS_NUMPY}")

            
            try:
                loaded_embeddings = np.load(OUTPUT_EMBEDDINGS_NUMPY)
                loaded_ids = np.load(OUTPUT_IDS_NUMPY)
                print(f"\nVerification: Loaded embeddings shape: {loaded_embeddings.shape}, Loaded IDs shape: {loaded_ids.shape}")
                if loaded_embeddings.shape[0] == len(loaded_ids):
                    print("Verification successful: Number of embeddings matches number of IDs.")
                else:
                    print("Verification FAILED: Mismatch between number of embeddings and IDs.")
            except Exception as e:
                print(f"Error during verification load: {e}")


            print("\nStep 3: Text Embedding Generation Complete.")
        else:
            print("Embedding generation failed.")
    else:
        print("Failed to load text chunks. Aborting embedding generation.")