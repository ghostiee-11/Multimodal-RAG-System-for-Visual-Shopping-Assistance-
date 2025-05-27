# retriever.py
import os
from pinecone import Pinecone as PineconeClient
from sentence_transformers import SentenceTransformer, CrossEncoder 
import torch
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
import numpy as np # Make sure numpy is imported

# --- Text Retrieval Configuration ---
TEXT_INDEX_NAME = "product-text-embeddings"
TEXT_EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/stsb-roberta-base'

# --- Image Retrieval Configuration ---
IMAGE_INDEX_NAME = "product-image-embeddings"
CLIP_MODEL_HF_ID = "openai/clip-vit-base-patch32"

# --- Global variables ---
pc_client_text = None
pinecone_text_index = None
text_embedding_bi_encoder_model = None
text_cross_encoder_model = None # Instance of CrossEncoder

pc_client_image = None
pinecone_image_index = None
hf_clip_model = None
hf_clip_processor = None

def load_env_vars_for_retriever():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not found.")
        return None
    return api_key

def initialize_text_retrieval_models():
    global text_embedding_bi_encoder_model, text_cross_encoder_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for text models.")

    if text_embedding_bi_encoder_model is None:
        try:
            print(f"Loading text bi-encoder model: {TEXT_EMBEDDING_MODEL_NAME}...")
            text_embedding_bi_encoder_model = SentenceTransformer(TEXT_EMBEDDING_MODEL_NAME, device=device)
            print(f"Text bi-encoder model loaded.")
        except Exception as e:
            print(f"Error loading text bi-encoder model: {e}")
            raise

    if text_cross_encoder_model is None:
        try:
            print(f"Loading text cross-encoder model: {CROSS_ENCODER_MODEL_NAME}...")
            text_cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device, max_length=512)
            print(f"Text cross-encoder model loaded.")
        except Exception as e:
            print(f"Error loading text cross-encoder model: {e}")
            raise

def initialize_image_retrieval_models():
    global hf_clip_model, hf_clip_processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for CLIP model.")

    if hf_clip_model is None or hf_clip_processor is None:
        try:
            print(f"Loading CLIP model and processor from Hugging Face: {CLIP_MODEL_HF_ID}...")
            hf_clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_HF_ID)
            hf_clip_model = CLIPModel.from_pretrained(CLIP_MODEL_HF_ID).to(device)
            hf_clip_model.eval()
            print(f"CLIP model and processor loaded.")
        except Exception as e:
            print(f"Error loading CLIP model/processor: {e}")
            raise

def initialize_pinecone_connections():
    global pc_client_text, pinecone_text_index, pc_client_image, pinecone_image_index
    
    pinecone_api_key = load_env_vars_for_retriever()
    if not pinecone_api_key:
        raise EnvironmentError("Pinecone API key not configured.")

    if pc_client_text is None:
        try:
            print("Initializing Pinecone client...")
            pc_client_text = PineconeClient(api_key=pinecone_api_key)
            pc_client_image = pc_client_text 
            print("Pinecone client initialized.")
        except Exception as e:
            print(f"Error initializing Pinecone client: {e}")
            raise

    if pinecone_text_index is None:
        try:
            existing_indexes = [index_info["name"] for index_info in pc_client_text.list_indexes()]
            if TEXT_INDEX_NAME not in existing_indexes:
                raise NameError(f"Error: Text Index '{TEXT_INDEX_NAME}' does not exist.")
            print(f"Connecting to Pinecone text index: {TEXT_INDEX_NAME}...")
            pinecone_text_index = pc_client_text.Index(TEXT_INDEX_NAME)
        except Exception as e:
            print(f"Error connecting to Pinecone text index '{TEXT_INDEX_NAME}': {e}")
            raise

    if pinecone_image_index is None:
        try:
            existing_indexes = [index_info["name"] for index_info in pc_client_image.list_indexes()]
            if IMAGE_INDEX_NAME not in existing_indexes:
                raise NameError(f"Error: Image Index '{IMAGE_INDEX_NAME}' does not exist.")
            print(f"Connecting to Pinecone image index: {IMAGE_INDEX_NAME}...")
            pinecone_image_index = pc_client_image.Index(IMAGE_INDEX_NAME)
        except Exception as e:
            print(f"Error connecting to Pinecone image index '{IMAGE_INDEX_NAME}': {e}")
            raise

def initialize_retriever_resources(): 
    print("Initializing ALL retriever resources (Models & Pinecone Connections)...")
    initialize_text_retrieval_models()
    initialize_image_retrieval_models()
    initialize_pinecone_connections() 
    print("All retriever resources should be initialized.")

def retrieve_and_rerank_text_chunks(query_text, initial_top_k=20, final_top_k=5, filter_dict=None): # ADDED filter_dict
    global pinecone_text_index, text_embedding_bi_encoder_model, text_cross_encoder_model

    # ... (initialization checks as before) ...
    if not all([pinecone_text_index, text_embedding_bi_encoder_model, text_cross_encoder_model]):
        print("Text retrieval/reranking resources not fully initialized. Attempting to re-initialize.")
        try: initialize_retriever_resources()
        except Exception as e: print(f"Failed to re-initialize resources: {e}"); return []


    print(f"\nOriginal text query for retrieval & reranking: '{query_text}'")
    if filter_dict:
        print(f"Applying Pinecone filter: {filter_dict}")

    print("Embedding text query with bi-encoder...")
    try: query_embedding = text_embedding_bi_encoder_model.encode(query_text)
    except Exception as e: print(f"Error embedding query: {e}"); return []

    print(f"Querying Pinecone text index for initial top {initial_top_k} candidates...")
    try:
        query_response = pinecone_text_index.query(
            vector=query_embedding.tolist(), 
            top_k=initial_top_k,
            include_metadata=True,
            filter=filter_dict # USE THE FILTER HERE
        )
    except Exception as e: print(f"Error during Pinecone query: {e}"); return []

    # ... (rest of the function: processing initial_chunks, reranking, etc. remains the SAME) ...
    # ... (make sure to handle the case where initial_chunks might be empty due to filtering) ...
    initial_chunks_from_pinecone = []
    if query_response and query_response.get('matches'):
        for match in query_response['matches']:
            initial_chunks_from_pinecone.append({
                "id": match.get('id', 'N/A'),
                "text_content": str(match.get('metadata', {}).get('text_content', '')).strip(),
                "metadata": match.get('metadata', {}),
                "pinecone_score": match.get('score', 0.0) 
            })
    if not initial_chunks_from_pinecone: 
        print(f"No initial matches found from Pinecone (with filter: {filter_dict})."); 
        return []

    valid_initial_chunks = []
    sentence_pairs = []
    # print("DEBUG: Validating chunks for cross-encoder...") # Optional
    for chunk in initial_chunks_from_pinecone:
        if chunk['text_content']: 
            sentence_pairs.append([query_text, chunk['text_content']])
            valid_initial_chunks.append(chunk)
        # else: # Optional: print(f"  DEBUG: Skipping chunk ID {chunk['id']} due to empty text_content for cross-encoder.")
    
    if not valid_initial_chunks:
        print("DEBUG: No valid chunks with text_content to rerank after filtering Pinecone results.")
        for chunk in initial_chunks_from_pinecone: chunk['score'] = chunk.get('pinecone_score', 0.0)
        return sorted(initial_chunks_from_pinecone, key=lambda x: x['score'], reverse=True)[:final_top_k]

    print(f"Reranking {len(valid_initial_chunks)} valid candidates with cross-encoder...")
    try:
        cross_encoder_scores_raw = text_cross_encoder_model.predict(sentence_pairs, show_progress_bar=False)
        # print(f"DEBUG: Raw cross_encoder_scores from model: {cross_encoder_scores_raw}") # Optional
        if not isinstance(cross_encoder_scores_raw, np.ndarray):
            cross_encoder_scores_np = np.array(cross_encoder_scores_raw, dtype=float)
        else:
            cross_encoder_scores_np = cross_encoder_scores_raw.astype(float)
        cross_encoder_scores_processed = [float(s) if not np.isnan(s) else -float('inf') for s in cross_encoder_scores_np.flatten()]
    except Exception as e:
        print(f"Error during cross-encoder prediction or score processing: {e}")
        valid_initial_chunks.sort(key=lambda x: x.get('pinecone_score', 0.0), reverse=True)
        for chunk in valid_initial_chunks: chunk['score'] = chunk.get('pinecone_score', 0.0)
        return valid_initial_chunks[:final_top_k]

    for chunk, new_score in zip(valid_initial_chunks, cross_encoder_scores_processed):
        chunk['score'] = new_score 
    reranked_chunks = sorted(valid_initial_chunks, key=lambda x: x['score'], reverse=True)
    print(f"Reranking complete. Returning top {len(reranked_chunks[:final_top_k])} results.")
    return reranked_chunks[:final_top_k]


def retrieve_relevant_images_from_text(query_text, top_k=3, filter_dict=None): # ADDED filter_dict
    global pinecone_image_index, hf_clip_model, hf_clip_processor
    # ... (initialization checks as before) ...
    if not all([pinecone_image_index, hf_clip_model, hf_clip_processor]):
        print("Image retriever resources not fully initialized. Attempting to re-initialize.")
        try: initialize_retriever_resources()
        except Exception as e: print(f"Failed to re-initialize resources: {e}"); return []

    print(f"\nOriginal text query for image retrieval: '{query_text}'")
    if filter_dict:
        print(f"Applying Pinecone filter to image search: {filter_dict}")
    
    print("Embedding text query with Hugging Face CLIP model...")
    # ... (embedding logic as before) ...
    try:
        inputs = hf_clip_processor(text=[query_text], return_tensors="pt", padding=True).to(hf_clip_model.device)
        with torch.no_grad(): text_features = hf_clip_model.get_text_features(**inputs)
        query_embedding = text_features[0].cpu().numpy()
    except Exception as e: print(f"Error embedding query with HF CLIP: {e}"); return []

    print(f"Querying Pinecone image index for top {top_k} images...")
    try:
        query_response = pinecone_image_index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict # USE THE FILTER HERE
        )
    # ... (rest of the function: processing image results, etc. remains the SAME) ...
    except Exception as e: print(f"Error during Pinecone image query: {e}"); return []
    retrieved_images_info = []
    if query_response and query_response.get('matches'):
        for match in query_response['matches']:
            metadata = match.get('metadata', {})
            retrieved_images_info.append({
                "id": match.get('id', 'N/A'), 
                "score": match.get('score', 0.0),
                "image_path": metadata.get('image_path', match.get('id', 'N/A')),
                "product_id": metadata.get('product_id', 'N/A'),
                "caption": metadata.get('generated_caption', 'N/A')
            })
    else: print(f"No image matches found (with filter: {filter_dict}).")
    return retrieved_images_info

# Aliases remain the same
retrieve_relevant_chunks = retrieve_and_rerank_text_chunks
if __name__ == '__main__':
    try:
        print("Attempting to initialize ALL retriever resources for testing...")
        initialize_retriever_resources()
        # ... (single pair test as before) ...
        print("\n--- Testing Cross-Encoder with a single controlled pair ---")
        if text_cross_encoder_model:
            try:
                test_pair_simple = [["good battery life", "This headphone has excellent battery life, lasting over 20 hours."]]
                test_score_simple = text_cross_encoder_model.predict(test_pair_simple)
                print(f"DEBUG: Controlled test pair score: {test_score_simple}")
            except Exception as e:
                print(f"DEBUG: Error on controlled single pair test: {e}")
        else:
            print("DEBUG: Cross-encoder model not initialized for single pair test.")
        print("--- End of Single Pair Test ---\n")

        print("All retriever resources initialized for testing.")
    except Exception as e:
        print(f"Could not initialize retriever during test: {e}")
        exit()

    test_queries_with_filters = [
        ("What is the battery life of the Sony WH-1000XM4 headphones?", {"product_id": "B0863FR3S9"}),
        ("blue color", {"product_id": "B08QTVL6C5"}) # JBL Tune 510BT Blue
    ]

    for query, p_filter in test_queries_with_filters:
        print(f"\n{'='*20} TESTING TEXT QUERY (FILTERED): '{query}' with filter {p_filter} {'='*20}")
        retrieved_texts = retrieve_relevant_chunks(query, initial_top_k=5, final_top_k=2, filter_dict=p_filter) 
        if retrieved_texts:
            print(f"\n--- Top {len(retrieved_texts)} RERANKED Filtered Text Chunks ---")
            for i, chunk in enumerate(retrieved_texts):
                print(f"  Text Result {i+1}: ID: {chunk['id']}, Rerank_Score: {chunk.get('score', 0.0):.4f}, Product: {chunk['metadata'].get('product_id')}")
        
        print(f"\n{'='*20} TESTING IMAGE QUERY (FILTERED): '{query}' with filter {p_filter} {'='*20}")
        retrieved_imgs = retrieve_relevant_images_from_text(query, top_k=2, filter_dict=p_filter)
        if retrieved_imgs:
            print(f"\n--- Top {len(retrieved_imgs)} Retrieved Filtered Images ---")
            for i, img_info in enumerate(retrieved_imgs):
                print(f"  Image Result {i+1}: Path: {img_info['image_path']}, CLIP_Score: {img_info['score']:.4f}, Product: {img_info['product_id']}")