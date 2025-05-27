# retriever_new_one.py
import os
from pinecone import Pinecone as PineconeClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel, ViltProcessor, ViltForImageAndTextRetrieval
from PIL import Image
import numpy as np
import pandas as pd 
import json         

# --- Text Retrieval Configuration ---
TEXT_INDEX_NAME = "product-text-embeddings"
TEXT_EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/stsb-roberta-base'

# --- Image Retrieval (CLIP) Configuration ---
IMAGE_INDEX_NAME = "product-image-embeddings" # As defined in setup_image_vector_db_pinecone.py
CLIP_MODEL_HF_ID = "openai/clip-vit-base-patch32" # Matches embed_images.py if using SentenceTransformer('clip-ViT-B-32')

# --- Image Reranking (ViLT) Configuration ---
VILT_RERANKER_MODEL_HF_ID = "dandelin/vilt-b32-finetuned-coco"

# --- Path to image captions/texts file (used if Pinecone metadata is incomplete) ---
IMAGE_METADATA_CSV_PATH = 'image_combined_blip_ocr_filtered_final.csv'

# --- Global variables ---
pc_client = None # Single client for both text and image
pinecone_text_index = None
text_embedding_bi_encoder_model = None
text_cross_encoder_model = None

pinecone_image_index = None
hf_clip_model = None
hf_clip_processor = None

hf_vilt_reranker_model = None
hf_vilt_reranker_processor = None

df_image_metadata_local_cache = None # For local fallback of image captions/texts

def load_env_vars_for_retriever():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Retriever Error: PINECONE_API_KEY not found in .env file.")
        return None
    return api_key

def initialize_local_image_metadata_cache():
    global df_image_metadata_local_cache
    if df_image_metadata_local_cache is None:
        try:
            df_image_metadata_local_cache = pd.read_csv(IMAGE_METADATA_CSV_PATH)
            if 'full_image_path' not in df_image_metadata_local_cache.columns or \
               'generated_texts_json' not in df_image_metadata_local_cache.columns:
                print(f"Retriever Warning: '{IMAGE_METADATA_CSV_PATH}' is missing 'full_image_path' or 'generated_texts_json'. Local caption fallback may fail.")
                df_image_metadata_local_cache = pd.DataFrame() # Empty to avoid errors
            else:
                 # Pre-parse the JSON to avoid repeated parsing
                df_image_metadata_local_cache['all_captions_list_parsed'] = df_image_metadata_local_cache['generated_texts_json'].apply(
                    lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else []
                )
                def get_primary(lst):
                    if lst and isinstance(lst, list) and len(lst) > 0 and isinstance(lst[0], dict) and 'text' in lst[0]:
                        return str(lst[0]['text'])
                    return "Caption unavailable."
                df_image_metadata_local_cache['primary_caption_parsed'] = df_image_metadata_local_cache['all_captions_list_parsed'].apply(get_primary)

            print(f"Retriever: Local image metadata cache loaded from '{IMAGE_METADATA_CSV_PATH}'. Shape: {df_image_metadata_local_cache.shape}")
        except FileNotFoundError:
            print(f"Retriever Warning: Local image metadata file '{IMAGE_METADATA_CSV_PATH}' not found. Will rely solely on Pinecone metadata for images.")
            df_image_metadata_local_cache = pd.DataFrame() # Empty DataFrame
        except Exception as e:
            print(f"Retriever Error: Loading local image metadata cache: {e}")
            df_image_metadata_local_cache = pd.DataFrame()

def get_captions_for_image_path_local(image_full_path):
    """Fallback to get captions from local CSV if Pinecone metadata is missing/incomplete."""
    if df_image_metadata_local_cache is None or df_image_metadata_local_cache.empty:
        return "Caption unavailable (local cache empty).", []

    row_data = df_image_metadata_local_cache[df_image_metadata_local_cache['full_image_path'] == image_full_path]
    if not row_data.empty:
        primary = row_data.iloc[0].get('primary_caption_parsed', "Caption unavailable (local).")
        all_list_of_dicts = row_data.iloc[0].get('all_captions_list_parsed', [])
        return primary, all_list_of_dicts
    return "Caption unavailable (not in local cache).", []


def initialize_text_retrieval_models():
    global text_embedding_bi_encoder_model, text_cross_encoder_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Retriever: Using device '{device}' for text models.")

    if text_embedding_bi_encoder_model is None:
        try:
            print(f"Retriever: Loading text bi-encoder model '{TEXT_EMBEDDING_MODEL_NAME}'...")
            text_embedding_bi_encoder_model = SentenceTransformer(TEXT_EMBEDDING_MODEL_NAME, device=device)
            print(f"Retriever: Text bi-encoder model '{TEXT_EMBEDDING_MODEL_NAME}' loaded.")
        except Exception as e:
            print(f"Retriever Error: Loading text bi-encoder model: {e}"); raise

    if text_cross_encoder_model is None:
        try:
            print(f"Retriever: Loading text cross-encoder model '{CROSS_ENCODER_MODEL_NAME}'...")
            text_cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device, max_length=512)
            print(f"Retriever: Text cross-encoder model '{CROSS_ENCODER_MODEL_NAME}' loaded.")
        except Exception as e:
            print(f"Retriever Error: Loading text cross-encoder model: {e}"); raise

def initialize_clip_image_retrieval_models():
    global hf_clip_model, hf_clip_processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Retriever: Using device '{device}' for CLIP model.")

    if hf_clip_model is None or hf_clip_processor is None:
        try:
            print(f"Retriever: Loading CLIP model and processor '{CLIP_MODEL_HF_ID}'...")
            hf_clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_HF_ID)
            hf_clip_model = CLIPModel.from_pretrained(CLIP_MODEL_HF_ID).to(device)
            hf_clip_model.eval()
            print(f"Retriever: CLIP model '{CLIP_MODEL_HF_ID}' and processor loaded.")
        except Exception as e:
            print(f"Retriever Error: Loading CLIP model/processor: {e}"); raise

def initialize_vilt_reranker_models():
    global hf_vilt_reranker_model, hf_vilt_reranker_processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Retriever: Using device '{device}' for ViLT reranker model.")

    if hf_vilt_reranker_model is None or hf_vilt_reranker_processor is None:
        try:
            print(f"Retriever: Loading ViLT reranker model and processor '{VILT_RERANKER_MODEL_HF_ID}'...")
            hf_vilt_reranker_processor = ViltProcessor.from_pretrained(VILT_RERANKER_MODEL_HF_ID)
            hf_vilt_reranker_model = ViltForImageAndTextRetrieval.from_pretrained(VILT_RERANKER_MODEL_HF_ID).to(device)
            hf_vilt_reranker_model.eval()
            print(f"Retriever: ViLT reranker model '{VILT_RERANKER_MODEL_HF_ID}' and processor loaded.")
        except Exception as e:
            print(f"Retriever Error: Loading ViLT reranker model/processor: {e}"); raise

def initialize_pinecone_connections():
    global pc_client, pinecone_text_index, pinecone_image_index
    
    pinecone_api_key = load_env_vars_for_retriever()
    if not pinecone_api_key: raise EnvironmentError("Retriever Error: Pinecone API key not configured.")

    if pc_client is None:
        try:
            print("Retriever: Initializing Pinecone client...")
            pc_client = PineconeClient(api_key=pinecone_api_key)
            print("Retriever: Pinecone client initialized.")
        except Exception as e:
            print(f"Retriever Error: Initializing Pinecone client: {e}"); raise

    if pinecone_text_index is None:
        try:
            print(f"Retriever: Connecting to Pinecone text index '{TEXT_INDEX_NAME}'...")
            if TEXT_INDEX_NAME not in [idx['name'] for idx in pc_client.list_indexes()]:
                raise ConnectionError(f"Pinecone text index '{TEXT_INDEX_NAME}' does not exist.")
            pinecone_text_index = pc_client.Index(TEXT_INDEX_NAME)
            print(f"Retriever: Text index '{TEXT_INDEX_NAME}' stats: {pinecone_text_index.describe_index_stats()}")
        except Exception as e:
            print(f"Retriever Error: Connecting to Pinecone text index '{TEXT_INDEX_NAME}': {e}"); raise

    if pinecone_image_index is None:
        try:
            print(f"Retriever: Connecting to Pinecone image index '{IMAGE_INDEX_NAME}'...")
            if IMAGE_INDEX_NAME not in [idx['name'] for idx in pc_client.list_indexes()]:
                 raise ConnectionError(f"Pinecone image index '{IMAGE_INDEX_NAME}' does not exist.")
            pinecone_image_index = pc_client.Index(IMAGE_INDEX_NAME)
            print(f"Retriever: Image index '{IMAGE_INDEX_NAME}' stats: {pinecone_image_index.describe_index_stats()}")
        except Exception as e:
            print(f"Retriever Error: Connecting to Pinecone image index '{IMAGE_INDEX_NAME}': {e}"); raise

def initialize_retriever_resources(): 
    print("Retriever: Initializing ALL retriever resources (Models, Pinecone Connections, Local Cache)...")
    initialize_text_retrieval_models()
    initialize_clip_image_retrieval_models()
    initialize_vilt_reranker_models()
    initialize_pinecone_connections() 
    initialize_local_image_metadata_cache()
    print("Retriever: All retriever resources initialized successfully.")

def retrieve_and_rerank_text_chunks(query_text, initial_top_k=20, final_top_k=5, filter_dict=None):
    global pinecone_text_index, text_embedding_bi_encoder_model, text_cross_encoder_model

    if not all([pinecone_text_index, text_embedding_bi_encoder_model, text_cross_encoder_model]):
        print("Retriever Error: Text retrieval/reranking resources not fully initialized.")
        return []

    # print(f"\n🔎 Retriever: Retrieving text for query: '{query_text}' with filter: {filter_dict}")
    try:
        query_embedding = text_embedding_bi_encoder_model.encode(query_text)
        # print(f"  Generated query embedding shape: {query_embedding.shape}")
        query_response = pinecone_text_index.query(
            vector=query_embedding.tolist(), 
            top_k=initial_top_k,
            include_metadata=True, # Crucial for getting text_type, original_doc_id etc.
            filter=filter_dict
        )
    except Exception as e:
        print(f"Retriever Error: During Pinecone text query or embedding: {e}"); return []

    initial_chunks = []
    if query_response and query_response.get('matches'):
        for match in query_response['matches']:
            # Metadata from Pinecone should already contain what was set up in setup_text_vector_db_pinecone.py
            # including text_content, product_id, text_type, original_doc_id, image_filename_source (for OCR/BLIP)
            initial_chunks.append({
                "id": match.get('id', 'N/A_id'), # Pinecone vector ID (text_chunk_id)
                "text_content": str(match.get('metadata', {}).get('text_content', '')).strip(), # Snippet from metadata
                "metadata": match.get('metadata', {}), # The whole metadata dict from Pinecone
                "pinecone_score": match.get('score', 0.0) 
            })
    if not initial_chunks:
        print("Retriever: No initial text matches found from Pinecone for the query."); return []

    # Rerank only if there's text_content to rerank with
    rerank_candidates = [chunk for chunk in initial_chunks if chunk['text_content']]
    if not rerank_candidates:
        print("Retriever: No valid text content found in initial matches for reranking. Returning based on Pinecone scores.")
        initial_chunks.sort(key=lambda x: x.get('pinecone_score', 0.0), reverse=True)
        for chunk in initial_chunks: chunk['score'] = chunk.get('pinecone_score', 0.0) # Use pinecone_score as final score
        return initial_chunks[:final_top_k]

    sentence_pairs = [[query_text, chunk['text_content']] for chunk in rerank_candidates]
        
    # print(f"Retriever: Reranking {len(sentence_pairs)} text candidates with cross-encoder...")
    try:
        cross_encoder_scores = text_cross_encoder_model.predict(sentence_pairs, show_progress_bar=False)
        if not isinstance(cross_encoder_scores, np.ndarray):
            cross_encoder_scores = np.array(cross_encoder_scores)

        for i, chunk in enumerate(rerank_candidates): 
            # Ensure score is float, handle potential NaN from cross-encoder if input is problematic
            chunk['score'] = float(cross_encoder_scores[i]) if not np.isnan(cross_encoder_scores[i]) else -float('inf')
            
    except Exception as e:
        print(f"Retriever Error: During cross-encoder prediction: {e}")
        # Fallback to Pinecone scores if cross-encoder fails
        for chunk in rerank_candidates: chunk['score'] = chunk.get('pinecone_score', 0.0)
        rerank_candidates.sort(key=lambda x: x['score'], reverse=True)
        return rerank_candidates[:final_top_k]

    # Combine chunks that were reranked with those that might have been filtered out due to empty text_content
    # This part might be redundant if rerank_candidates is derived from initial_chunks filtering only on empty text
    all_scored_chunks_dict = {chunk['id']: chunk for chunk in rerank_candidates} 

    for chunk in initial_chunks: # Ensure all initial chunks get a score
        if chunk['id'] not in all_scored_chunks_dict: # If it wasn't rerankable (e.g. empty text)
            chunk['score'] = chunk.get('pinecone_score', 0.0) # Use its original pinecone score
            all_scored_chunks_dict[chunk['id']] = chunk
    
    final_sorted_chunks = sorted(list(all_scored_chunks_dict.values()), key=lambda x: x.get('score', -float('inf')), reverse=True)
    # print(f"Retriever: Text reranking complete. Returning top {min(final_top_k, len(final_sorted_chunks))} results.")
    return final_sorted_chunks[:final_top_k]

def retrieve_relevant_images_from_text_clip(query_text, top_k=5, filter_dict=None):
    global pinecone_image_index, hf_clip_model, hf_clip_processor
    if not all([pinecone_image_index, hf_clip_model, hf_clip_processor]):
        print("Retriever Error: CLIP image retrieval resources not fully initialized.")
        return []

    # print(f"\n🖼️ Retriever: Retrieving images with CLIP for query: '{query_text}' with filter: {filter_dict}")
    try:
        inputs = hf_clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True).to(hf_clip_model.device)
        with torch.no_grad(): text_features = hf_clip_model.get_text_features(**inputs)
        query_embedding = text_features[0].cpu().numpy()
        
        query_response = pinecone_image_index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True, # Essential for product_id, image_path, and captions
            filter=filter_dict
        )
    except Exception as e:
        print(f"Retriever Error: During Pinecone image query or CLIP embedding: {e}"); return []

    retrieved_images = []
    if query_response and query_response.get('matches'):
        for match in query_response['matches']:
            metadata = match.get('metadata', {})
            image_path = metadata.get('image_path', match.get('id', 'N/A_path')) # ID is the image_path
            
            # Try to get captions from Pinecone metadata first
            primary_caption_pinecone = metadata.get('primary_caption', "Caption unavailable.")
            all_captions_list_pinecone = metadata.get('generated_captions_list', []) # This should be list of dicts

            # Fallback to local CSV if Pinecone metadata is incomplete or missing
            if primary_caption_pinecone == "Caption unavailable." or not all_captions_list_pinecone:
                # print(f"  Retriever Info: Pinecone metadata for {image_path} incomplete/missing captions. Trying local cache.")
                primary_caption_local, all_captions_list_local = get_captions_for_image_path_local(image_path)
                final_primary_caption = primary_caption_local if primary_caption_local != "Caption unavailable (local cache empty)." else primary_caption_pinecone
                final_all_captions_list = all_captions_list_local if all_captions_list_local else all_captions_list_pinecone
            else:
                final_primary_caption = primary_caption_pinecone
                final_all_captions_list = all_captions_list_pinecone


            retrieved_images.append({
                "id": match.get('id', image_path), 
                "score": match.get('score', 0.0), 
                "image_path": image_path,
                "product_id": metadata.get('product_id', 'N/A_pid'),
                "primary_caption": final_primary_caption, # String
                "all_captions": final_all_captions_list # Should be list of dicts {'source':..., 'text':...}
            })
    # if not retrieved_images: print("Retriever: No initial image matches found from CLIP search.")
    return retrieved_images

def rerank_images_with_vilt(query_text, candidate_images_data, top_k=2):
    global hf_vilt_reranker_model, hf_vilt_reranker_processor
    if not all([hf_vilt_reranker_model, hf_vilt_reranker_processor]):
        print("Retriever Warning: ViLT reranker model not initialized. Returning candidates sorted by original CLIP score.")
        return sorted(candidate_images_data, key=lambda x: x.get('score', 0.0), reverse=True)[:top_k]

    if not candidate_images_data: return []

    # print(f"\n⚡ Retriever: Reranking {len(candidate_images_data)} image candidates with ViLT for query: '{query_text}'")
    
    rerank_candidates_with_scores = []
    for item_data in candidate_images_data:
        image_path = item_data.get("image_path")
        
        # For ViLT, we primarily use the `primary_caption` (which is a single string)
        # The `all_captions` (list of dicts) is more for the final LLM context.
        caption_for_vilt = item_data.get("primary_caption", "")
        if not caption_for_vilt or caption_for_vilt == "Caption unavailable.":
            # If primary is bad, try to make a simple one from 'all_captions' list of dicts
            all_caps_dicts = item_data.get("all_captions", [])
            if all_caps_dicts and isinstance(all_caps_dicts, list) and len(all_caps_dicts) > 0:
                 # Prefer BLIP or OCR text for ViLT if available
                blip_texts = [d['text'] for d in all_caps_dicts if isinstance(d,dict) and d.get('source','').lower().startswith('blip') and d.get('text')]
                ocr_texts = [d['text'] for d in all_caps_dicts if isinstance(d,dict) and d.get('source','').lower().startswith('ocr') and d.get('text')]
                if blip_texts: caption_for_vilt = blip_texts[0]
                elif ocr_texts: caption_for_vilt = ocr_texts[0]
                elif isinstance(all_caps_dicts[0], dict) and 'text' in all_caps_dicts[0]:
                    caption_for_vilt = all_caps_dicts[0]['text']

        if not caption_for_vilt or caption_for_vilt == "Caption unavailable.":
             caption_for_vilt = "product image" # Generic fallback

        if not image_path or not os.path.exists(image_path):
            # print(f"  Retriever ViLT Warning: Skipping missing image path: {image_path}"); 
            item_data['vilt_score'] = -float('inf') # Penalize missing images heavily
            rerank_candidates_with_scores.append(item_data)
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            
            # Prepare text for ViLT: combine query and the best single caption.
            # Truncate to avoid overly long inputs for ViLT.
            query_tokens = query_text.split()
            caption_tokens = caption_for_vilt.split()
            processed_query_for_vilt = " ".join(query_tokens[:20]) 
            processed_caption_for_vilt = " ".join(caption_tokens[:18])

            if processed_query_for_vilt and processed_caption_for_vilt:
                text_input_for_vilt = f"{processed_query_for_vilt} [SEP] {processed_caption_for_vilt}"
            elif processed_query_for_vilt:
                text_input_for_vilt = processed_query_for_vilt
            else: 
                text_input_for_vilt = processed_caption_for_vilt if processed_caption_for_vilt else "image context" # Absolute fallback

            # Max length for ViLT is typically around 40 tokens for text.
            inputs = hf_vilt_reranker_processor(image, text_input_for_vilt, return_tensors="pt", padding="max_length", truncation=True, max_length=40).to(hf_vilt_reranker_model.device)
            
            with torch.no_grad():
                outputs = hf_vilt_reranker_model(**inputs)
            
            # The logits usually represent relevance score for image-text matching.
            # For dandelin/vilt-b32-finetuned-coco, logits[0,0] is often used for retrieval score.
            vilt_relevance_score = outputs.logits[0, 0].item() 
            
            item_data['vilt_score'] = vilt_relevance_score
            rerank_candidates_with_scores.append(item_data)

        except Exception as e:
            print(f"  Retriever ViLT Error: Reranking image {os.path.basename(image_path)}: {e}")
            item_data['vilt_score'] = -float('inf') # Penalize errors
            rerank_candidates_with_scores.append(item_data)

    reranked_items = sorted(rerank_candidates_with_scores, key=lambda x: x.get("vilt_score", -float('inf')), reverse=True)
    # print(f"Retriever: ViLT Image Reranking complete. Returning top {min(top_k, len(reranked_items))} results.")
    return reranked_items[:top_k]

# Alias for consistency if used elsewhere
retrieve_relevant_chunks = retrieve_and_rerank_text_chunks

if __name__ == '__main__':
    try:
        print("Retriever Test: Initializing ALL retriever resources...")
        initialize_retriever_resources()
        print("Retriever Test: All retriever resources initialized.")
    except Exception as e:
        print(f"Retriever Test Error: Could not initialize retriever: {e}"); exit()

    test_query = "Are Sony WH-1000XM4 headphones good for flights and noise cancellation?"
    test_filter_text = {"product_id": {"$in": ["B0863FR3S9", "B09XS7JWHH"]}} # Example filter
    test_filter_image = {"product_id": {"$in": ["B0863FR3S9", "B09XS7JWHH"]}}

    print(f"\n{'='*20} Retriever Test: TEXT RETRIEVAL: '{test_query}' {'='*20}")
    retrieved_texts = retrieve_relevant_chunks(test_query, initial_top_k=5, final_top_k=2, filter_dict=test_filter_text)
    if retrieved_texts:
        print(f"\n--- Top {len(retrieved_texts)} RERANKED Text Chunks ---")
        for i, chunk in enumerate(retrieved_texts):
            meta_display = chunk.get('metadata', {})
            print(f"  Text Result {i+1}: ID: {chunk['id']}, Score: {chunk.get('score',0.0):.4f}, ProdID: {meta_display.get('product_id')}, Type: {meta_display.get('text_type')}")
            print(f"    Text: {chunk.get('text_content', '')[:80]}...")
            if meta_display.get('text_type', '').startswith('image_'):
                print(f"    Image Source: {meta_display.get('image_filename_source', meta_display.get('original_doc_id'))}")
    else:
        print("Retriever Test: No text chunks retrieved.")

    print(f"\n{'='*20} Retriever Test: IMAGE RETRIEVAL (CLIP): '{test_query}' {'='*20}")
    clip_retrieved_images = retrieve_relevant_images_from_text_clip(test_query, top_k=3, filter_dict=test_filter_image)
    if clip_retrieved_images:
        print(f"\n--- Top {len(clip_retrieved_images)} CLIP Retrieved Images ---")
        for i, img_info in enumerate(clip_retrieved_images):
            print(f"  Image {i+1}: Path: {os.path.basename(img_info['image_path'])}, CLIP_Score: {img_info['score']:.4f}, ProdID: {img_info['product_id']}")
            print(f"    Primary Caption: {str(img_info.get('primary_caption','N/A'))[:70]}...")
            # print(f"    All Captions/Texts: {img_info.get('all_captions')}") # This is now list of dicts

        print(f"\n{'='*20} Retriever Test: IMAGE RERANKING (ViLT) on CLIP results {'='*20}")
        vilt_reranked_images_test = rerank_images_with_vilt(test_query, clip_retrieved_images, top_k=2)
        if vilt_reranked_images_test:
            print(f"\n--- Top {len(vilt_reranked_images_test)} ViLT RERANKED Images ---")
            for i, img_info in enumerate(vilt_reranked_images_test):
                print(f"  Reranked Image {i+1}: Path: {os.path.basename(img_info['image_path'])}, ViLT_Score: {img_info.get('vilt_score',0.0):.4f}, ProdID: {img_info['product_id']}")
                print(f"    Primary Caption: {str(img_info.get('primary_caption','N/A'))[:70]}...")
    else:
        print("Retriever Test: No images retrieved by CLIP to rerank with ViLT.")