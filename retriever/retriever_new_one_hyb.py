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
from rank_bm25 import BM25Okapi # <-- Added for BM25
from collections import defaultdict # <-- Added for RRF

# --- Text Retrieval Configuration ---
TEXT_INDEX_NAME = "product-text-embeddings"
TEXT_EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/stsb-roberta-base'
TEXT_CHUNKS_CSV_FOR_METADATA_AND_BM25 = 'prepared_text_chunks.csv' # <-- Path to your prepared chunks

# --- Image Retrieval (CLIP) Configuration ---
IMAGE_INDEX_NAME = "product-image-embeddings"
CLIP_MODEL_HF_ID = "openai/clip-vit-base-patch32"

# --- Image Reranking (ViLT) Configuration ---
VILT_RERANKER_MODEL_HF_ID = "dandelin/vilt-b32-finetuned-coco"

# --- Path to image captions/texts file (used if Pinecone metadata is incomplete) ---
IMAGE_METADATA_CSV_PATH = 'image_combined_blip_ocr_filtered_final.csv'

# --- Global variables ---
pc_client = None
pinecone_text_index = None
text_embedding_bi_encoder_model = None
text_cross_encoder_model = None

pinecone_image_index = None
hf_clip_model = None
hf_clip_processor = None

hf_vilt_reranker_model = None
hf_vilt_reranker_processor = None

df_image_metadata_local_cache = None
df_prepared_text_chunks_global = None # <-- Added for BM25 corpus and metadata lookup
bm25_index_text = None                 # <-- Added for BM25 index

# ... (load_env_vars_for_retriever, initialize_local_image_metadata_cache, get_captions_for_image_path_local remain the same) ...
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
                df_image_metadata_local_cache = pd.DataFrame() 
            else:
                df_image_metadata_local_cache['all_captions_list_parsed'] = df_image_metadata_local_cache['generated_texts_json'].apply(
                    lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else []
                )
                def get_primary(lst_of_dicts): # Assuming V3 structure: list of dicts
                    if lst_of_dicts and isinstance(lst_of_dicts, list) and len(lst_of_dicts) > 0:
                        first_item = lst_of_dicts[0]
                        if isinstance(first_item, dict) and 'text' in first_item:
                            return str(first_item['text'])
                        elif isinstance(first_item, str): # Fallback if it's list of strings
                            return first_item
                    return "Caption unavailable."
                df_image_metadata_local_cache['primary_caption_parsed'] = df_image_metadata_local_cache['all_captions_list_parsed'].apply(get_primary)
            print(f"Retriever: Local image metadata cache loaded from '{IMAGE_METADATA_CSV_PATH}'. Shape: {df_image_metadata_local_cache.shape}")
        except FileNotFoundError:
            print(f"Retriever Warning: Local image metadata file '{IMAGE_METADATA_CSV_PATH}' not found. Will rely solely on Pinecone metadata for images.")
            df_image_metadata_local_cache = pd.DataFrame()
        except Exception as e:
            print(f"Retriever Error: Loading local image metadata cache: {e}")
            df_image_metadata_local_cache = pd.DataFrame()

def get_captions_for_image_path_local(image_full_path):
    if df_image_metadata_local_cache is None or df_image_metadata_local_cache.empty:
        return "Caption unavailable (local cache empty).", []
    row_data = df_image_metadata_local_cache[df_image_metadata_local_cache['full_image_path'] == image_full_path]
    if not row_data.empty:
        primary = row_data.iloc[0].get('primary_caption_parsed', "Caption unavailable (local).")
        # For V3, 'all_captions_list_parsed' is expected to be a list of dicts like [{'source':'BLIP', 'text':'...'}, {'source':'OCR', 'text':'...'}]
        all_list_of_dicts = row_data.iloc[0].get('all_captions_list_parsed', [])
        return primary, all_list_of_dicts
    return "Caption unavailable (not in local cache).", []

def initialize_text_retrieval_models_and_bm25(): # Renamed to include BM25
    global text_embedding_bi_encoder_model, text_cross_encoder_model, df_prepared_text_chunks_global, bm25_index_text
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

    # Initialize BM25 Index
    if df_prepared_text_chunks_global is None or bm25_index_text is None:
        try:
            print(f"Retriever: Loading '{TEXT_CHUNKS_CSV_FOR_METADATA_AND_BM25}' for BM25 index...")
            df_prepared_text_chunks_global = pd.read_csv(TEXT_CHUNKS_CSV_FOR_METADATA_AND_BM25)
            df_prepared_text_chunks_global['text_chunk_id'] = df_prepared_text_chunks_global['text_chunk_id'].astype(str)
            df_prepared_text_chunks_global['text_content'] = df_prepared_text_chunks_global['text_content'].fillna('').astype(str)
            
            corpus_for_bm25 = df_prepared_text_chunks_global['text_content'].tolist()
            tokenized_corpus_for_bm25 = [doc.lower().split() for doc in corpus_for_bm25]
            bm25_index_text = BM25Okapi(tokenized_corpus_for_bm25)
            print(f"Retriever: BM25 index built from '{TEXT_CHUNKS_CSV_FOR_METADATA_AND_BM25}' with {len(corpus_for_bm25)} documents.")
        except FileNotFoundError:
            print(f"Retriever CRITICAL Error: '{TEXT_CHUNKS_CSV_FOR_METADATA_AND_BM25}' not found. BM25 index cannot be built.")
            # Potentially raise an error or set a flag indicating BM25 is unavailable
            bm25_index_text = None 
            df_prepared_text_chunks_global = pd.DataFrame() # Empty df to prevent errors
        except Exception as e:
            print(f"Retriever Error: Failed to build BM25 index: {e}")
            bm25_index_text = None
            df_prepared_text_chunks_global = pd.DataFrame()


# ... (initialize_clip_image_retrieval_models, initialize_vilt_reranker_models, initialize_pinecone_connections remain the same) ...
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
    print("Retriever: Initializing ALL retriever resources (Models, Pinecone Connections, Local Cache, BM25)...")
    initialize_text_retrieval_models_and_bm25() # <-- Modified to include BM25
    initialize_clip_image_retrieval_models()
    initialize_vilt_reranker_models()
    initialize_pinecone_connections()
    initialize_local_image_metadata_cache()
    print("Retriever: All retriever resources initialized successfully.")


def retrieve_relevant_chunks(query_text, initial_top_k_dense=15, initial_top_k_sparse=15, rerank_top_p_hybrid=20, final_top_k=5, filter_dict=None):
    global pinecone_text_index, text_embedding_bi_encoder_model, text_cross_encoder_model, bm25_index_text, df_prepared_text_chunks_global

    if not all([pinecone_text_index, text_embedding_bi_encoder_model, text_cross_encoder_model, bm25_index_text, df_prepared_text_chunks_global is not None]):
        print("Retriever Error: Text retrieval/reranking resources not fully initialized (including BM25).")
        return []

    # 1. Dense Retrieval (Pinecone)
    dense_results_list = []
    try:
        query_embedding = text_embedding_bi_encoder_model.encode(query_text)
        query_response_dense = pinecone_text_index.query(
            vector=query_embedding.tolist(),
            top_k=initial_top_k_dense,
            include_metadata=True,
            filter=filter_dict
        )
        if query_response_dense and query_response_dense.get('matches'):
            for rank, match in enumerate(query_response_dense['matches']):
                dense_results_list.append({
                    "id": match.get('id', 'N/A_id'),
                    "text_content": str(match.get('metadata', {}).get('text_content', '')).strip(),
                    "metadata": match.get('metadata', {}),
                    "score": match.get('score', 0.0), # Pinecone score (dense)
                    "rank": rank + 1,
                    "retrieval_type": "dense"
                })
    except Exception as e:
        print(f"Retriever Error: During Pinecone dense query: {e}")
    
    # 2. Sparse Retrieval (BM25)
    sparse_results_list = []
    try:
        tokenized_query = query_text.lower().split()
        bm25_scores = bm25_index_text.get_scores(tokenized_query)
        
        # Get top N indices from BM25 scores. Consider more than initial_top_k_sparse before filtering.
        num_bm25_candidates_to_consider = initial_top_k_sparse * 3 
        top_n_bm25_original_indices = np.argsort(bm25_scores)[::-1][:num_bm25_candidates_to_consider]

        temp_sparse_results = []
        for original_corpus_idx in top_n_bm25_original_indices:
            if bm25_scores[original_corpus_idx] > 0: # Only consider non-zero scores
                chunk_data_row = df_prepared_text_chunks_global.iloc[original_corpus_idx]
                # Apply filter_dict if present
                if filter_dict:
                    target_pids = set(filter_dict.get('product_id',{}).get('$in',[]))
                    if target_pids and chunk_data_row['product_id'] not in target_pids:
                        continue # Skip if product_id doesn't match filter
                
                temp_sparse_results.append({
                    "id": chunk_data_row['text_chunk_id'],
                    "text_content": chunk_data_row['text_content'],
                    "metadata": chunk_data_row.to_dict(), # Store all metadata from the df row
                    "score": bm25_scores[original_corpus_idx], # BM25 score
                    "retrieval_type": "sparse"
                })
        # Sort by BM25 score and assign rank
        temp_sparse_results.sort(key=lambda x: x['score'], reverse=True)
        for rank, item in enumerate(temp_sparse_results[:initial_top_k_sparse]):
            item['rank'] = rank + 1
            sparse_results_list.append(item)
            
    except Exception as e:
        print(f"Retriever Error: During BM25 retrieval: {e}")

    # 3. Reciprocal Rank Fusion (RRF)
    K_RRF = 60  # Standard RRF constant
    rrf_scores_map = defaultdict(float)
    all_retrieved_docs_map = {} # To store full doc info by id

    for res in dense_results_list:
        rrf_scores_map[res['id']] += 1 / (K_RRF + res['rank'])
        if res['id'] not in all_retrieved_docs_map: all_retrieved_docs_map[res['id']] = res
    
    for res in sparse_results_list:
        rrf_scores_map[res['id']] += 1 / (K_RRF + res['rank'])
        if res['id'] not in all_retrieved_docs_map: all_retrieved_docs_map[res['id']] = res
            
    hybrid_results_for_rerank = []
    for doc_id, rrf_score_val in rrf_scores_map.items():
        doc_data = all_retrieved_docs_map[doc_id]
        hybrid_results_for_rerank.append({
            "id": doc_id,
            "text_content": doc_data['text_content'],
            "metadata": doc_data['metadata'], # Keep full metadata
            "rrf_score": rrf_score_val # Add RRF score for sorting
        })
    
    hybrid_results_for_rerank.sort(key=lambda x: x['rrf_score'], reverse=True)
    
    # Select top P for cross-encoder
    candidates_for_cross_encoder = hybrid_results_for_rerank[:rerank_top_p_hybrid]
    
    if not candidates_for_cross_encoder:
        print("Retriever: No candidates after RRF. Returning empty list.")
        return []

    # 4. Cross-Encoder Reranking
    sentence_pairs = [[query_text, chunk['text_content']] for chunk in candidates_for_cross_encoder if chunk['text_content']]
    if not sentence_pairs:
        print("Retriever: No valid sentence pairs for cross-encoder. Returning RRF sorted list.")
        # Add a 'score' field based on rrf_score if cross-encoder is skipped
        for chunk in candidates_for_cross_encoder: chunk['score'] = chunk.get('rrf_score', 0.0)
        return candidates_for_cross_encoder[:final_top_k]

    try:
        cross_encoder_scores_raw = text_cross_encoder_model.predict(sentence_pairs, show_progress_bar=False)
        if not isinstance(cross_encoder_scores_raw, np.ndarray):
            cross_encoder_scores_np = np.array(cross_encoder_scores_raw, dtype=float)
        else:
            cross_encoder_scores_np = cross_encoder_scores_raw.astype(float)
        
        cross_encoder_scores_processed = [float(s) if not np.isnan(s) else -float('inf') for s in cross_encoder_scores_np.flatten()]

        for i, chunk in enumerate(candidates_for_cross_encoder): # Only update score for those passed to CE
            if i < len(cross_encoder_scores_processed):
                 chunk['score'] = cross_encoder_scores_processed[i] # Final score is from cross-encoder
            else: # Should not happen if sentence_pairs matches candidates_for_cross_encoder
                 chunk['score'] = chunk.get('rrf_score', 0.0) # Fallback
            
    except Exception as e:
        print(f"Retriever Error: During cross-encoder prediction: {e}")
        for chunk in candidates_for_cross_encoder: chunk['score'] = chunk.get('rrf_score', 0.0) # Fallback to RRF score

    final_reranked_chunks = sorted(candidates_for_cross_encoder, key=lambda x: x.get('score', -float('inf')), reverse=True)
    return final_reranked_chunks[:final_top_k]

# ... (retrieve_relevant_images_from_text_clip and rerank_images_with_vilt remain the same) ...
def retrieve_relevant_images_from_text_clip(query_text, top_k=5, filter_dict=None):
    global pinecone_image_index, hf_clip_model, hf_clip_processor
    if not all([pinecone_image_index, hf_clip_model, hf_clip_processor]):
        print("Retriever Error: CLIP image retrieval resources not fully initialized.")
        return []
    try:
        inputs = hf_clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True).to(hf_clip_model.device)
        with torch.no_grad(): text_features = hf_clip_model.get_text_features(**inputs)
        query_embedding = text_features[0].cpu().numpy()
        query_response = pinecone_image_index.query(
            vector=query_embedding.tolist(), top_k=top_k, include_metadata=True, filter=filter_dict
        )
    except Exception as e:
        print(f"Retriever Error: During Pinecone image query or CLIP embedding: {e}"); return []
    retrieved_images = []
    if query_response and query_response.get('matches'):
        for match in query_response['matches']:
            metadata = match.get('metadata', {})
            image_path = metadata.get('image_path', match.get('id', 'N/A_path'))
            primary_caption_pinecone = metadata.get('primary_caption', "Caption unavailable.")
            # For V3, 'generated_texts_json' in Pinecone metadata contains the list of dicts
            all_captions_list_json_str = metadata.get('generated_texts_json', '[]')
            try:
                all_captions_list_dicts = json.loads(all_captions_list_json_str)
                if not (isinstance(all_captions_list_dicts, list) and all(isinstance(item, dict) and 'text' in item and 'source' in item for item in all_captions_list_dicts)):
                    all_captions_list_dicts = [] # Default to empty if structure is wrong
            except json.JSONDecodeError:
                all_captions_list_dicts = []

            if primary_caption_pinecone == "Caption unavailable." or not all_captions_list_dicts:
                primary_caption_local, all_captions_list_local_dicts = get_captions_for_image_path_local(image_path)
                final_primary_caption = primary_caption_local if primary_caption_local != "Caption unavailable (local cache empty)." else primary_caption_pinecone
                final_all_captions_list_dicts = all_captions_list_local_dicts if all_captions_list_local_dicts else all_captions_list_dicts
            else:
                final_primary_caption = primary_caption_pinecone
                final_all_captions_list_dicts = all_captions_list_dicts
            retrieved_images.append({
                "id": match.get('id', image_path), "score": match.get('score', 0.0), "image_path": image_path,
                "product_id": metadata.get('product_id', 'N/A_pid'),
                "primary_caption": final_primary_caption, # This should be a string
                "all_captions_with_source": final_all_captions_list_dicts # This is the list of dicts
            })
    return retrieved_images

def rerank_images_with_vilt(query_text, candidate_images_data, top_k=2):
    global hf_vilt_reranker_model, hf_vilt_reranker_processor
    if not all([hf_vilt_reranker_model, hf_vilt_reranker_processor]):
        print("Retriever Warning: ViLT reranker model not initialized. Returning candidates sorted by original CLIP score.")
        return sorted(candidate_images_data, key=lambda x: x.get('score', 0.0), reverse=True)[:top_k]
    if not candidate_images_data: return []
    rerank_candidates_with_scores = []
    for item_data in candidate_images_data:
        image_path = item_data.get("image_path")
        # Use primary_caption (string) for ViLT text input.
        # The `all_captions_with_source` (list of dicts) is for LLM context enrichment.
        caption_for_vilt = item_data.get("primary_caption", "")
        if not caption_for_vilt or caption_for_vilt == "Caption unavailable.":
            all_caps_dicts = item_data.get("all_captions_with_source", []) # This is list of dicts
            if all_caps_dicts and isinstance(all_caps_dicts, list) and len(all_caps_dicts) > 0:
                # Prioritize BLIP or OCR for ViLT if available from the list of dicts
                blip_texts = [d['text'] for d in all_caps_dicts if isinstance(d,dict) and d.get('source','').lower().startswith('blip') and d.get('text')]
                ocr_texts = [d['text'] for d in all_caps_dicts if isinstance(d,dict) and d.get('source','').lower().startswith('ocr') and d.get('text')]
                if blip_texts: caption_for_vilt = blip_texts[0]
                elif ocr_texts: caption_for_vilt = ocr_texts[0]
                elif isinstance(all_caps_dicts[0], dict) and 'text' in all_caps_dicts[0]:
                    caption_for_vilt = all_caps_dicts[0]['text'] # Fallback to first text in dict
        if not caption_for_vilt or caption_for_vilt == "Caption unavailable.":
             caption_for_vilt = "product image"
        if not image_path or not os.path.exists(image_path):
            item_data['vilt_score'] = -float('inf'); rerank_candidates_with_scores.append(item_data); continue
        try:
            image = Image.open(image_path).convert("RGB")
            query_tokens = query_text.split(); caption_tokens = caption_for_vilt.split()
            processed_query_for_vilt = " ".join(query_tokens[:20]); processed_caption_for_vilt = " ".join(caption_tokens[:18])
            if processed_query_for_vilt and processed_caption_for_vilt:
                text_input_for_vilt = f"{processed_query_for_vilt} [SEP] {processed_caption_for_vilt}"
            elif processed_query_for_vilt: text_input_for_vilt = processed_query_for_vilt
            else: text_input_for_vilt = processed_caption_for_vilt if processed_caption_for_vilt else "image context"
            inputs = hf_vilt_reranker_processor(image, text_input_for_vilt, return_tensors="pt", padding="max_length", truncation=True, max_length=40).to(hf_vilt_reranker_model.device)
            with torch.no_grad(): outputs = hf_vilt_reranker_model(**inputs)
            vilt_relevance_score = outputs.logits[0, 0].item() 
            item_data['vilt_score'] = vilt_relevance_score
            rerank_candidates_with_scores.append(item_data)
        except Exception as e:
            print(f"  Retriever ViLT Error: Reranking image {os.path.basename(image_path)}: {e}")
            item_data['vilt_score'] = -float('inf'); rerank_candidates_with_scores.append(item_data)
    reranked_items = sorted(rerank_candidates_with_scores, key=lambda x: x.get("vilt_score", -float('inf')), reverse=True)
    return reranked_items[:top_k]


if __name__ == '__main__':
    try:
        print("Retriever Test: Initializing ALL retriever resources...")
        initialize_retriever_resources()
        print("Retriever Test: All retriever resources initialized.")
    except Exception as e:
        print(f"Retriever Test Error: Could not initialize retriever: {e}"); exit()

    test_query = "Are Sony WH-1000XM4 headphones good for flights and noise cancellation?"
    test_filter_text = {"product_id": {"$in": ["B0863FR3S9", "B09XS7JWHH"]}}
    test_filter_image = {"product_id": {"$in": ["B0863FR3S9", "B09XS7JWHH"]}}

    print(f"\n{'='*20} Retriever Test: HYBRID TEXT RETRIEVAL: '{test_query}' {'='*20}")
    retrieved_texts = retrieve_relevant_chunks(test_query, 
                                               initial_top_k_dense=10, 
                                               initial_top_k_sparse=10, 
                                               rerank_top_p_hybrid=10, 
                                               final_top_k=3, 
                                               filter_dict=test_filter_text)
    if retrieved_texts:
        print(f"\n--- Top {len(retrieved_texts)} HYBRID + RERANKED Text Chunks ---")
        for i, chunk in enumerate(retrieved_texts):
            meta_display = chunk.get('metadata', {})
            print(f"  Text Result {i+1}: ID: {chunk['id']}, Score: {chunk.get('score',0.0):.4f}, ProdID: {meta_display.get('product_id')}, Type: {meta_display.get('text_type')}")
            print(f"    Text: {chunk.get('text_content', '')[:100]}...")
            if meta_display.get('text_type', '').startswith('image_'):
                print(f"    Image Source: {meta_display.get('image_filename_source', meta_display.get('original_doc_id'))}")
    else:
        print("Retriever Test: No text chunks retrieved with hybrid search.")

    # ... (image retrieval and reranking tests can remain the same) ...
    print(f"\n{'='*20} Retriever Test: IMAGE RETRIEVAL (CLIP): '{test_query}' {'='*20}")
    clip_retrieved_images = retrieve_relevant_images_from_text_clip(test_query, top_k=3, filter_dict=test_filter_image)
    if clip_retrieved_images:
        print(f"\n--- Top {len(clip_retrieved_images)} CLIP Retrieved Images ---")
        for i, img_info in enumerate(clip_retrieved_images):
            print(f"  Image {i+1}: Path: {os.path.basename(img_info['image_path'])}, CLIP_Score: {img_info['score']:.4f}, ProdID: {img_info['product_id']}")
            print(f"    Primary Caption (for ViLT): {str(img_info.get('primary_caption','N/A'))[:70]}...")
            print(f"    All Captions/Texts (for LLM, sample): {str(img_info.get('all_captions_with_source',[]))[:150]}...") # list of dicts

        print(f"\n{'='*20} Retriever Test: IMAGE RERANKING (ViLT) on CLIP results {'='*20}")
        vilt_reranked_images_test = rerank_images_with_vilt(test_query, clip_retrieved_images, top_k=2)
        if vilt_reranked_images_test:
            print(f"\n--- Top {len(vilt_reranked_images_test)} ViLT RERANKED Images ---")
            for i, img_info in enumerate(vilt_reranked_images_test):
                print(f"  Reranked Image {i+1}: Path: {os.path.basename(img_info['image_path'])}, ViLT_Score: {img_info.get('vilt_score',0.0):.4f}, ProdID: {img_info['product_id']}")
                print(f"    Primary Caption (used for ViLT): {str(img_info.get('primary_caption','N/A'))[:70]}...")
    else:
        print("Retriever Test: No images retrieved by CLIP to rerank with ViLT.")