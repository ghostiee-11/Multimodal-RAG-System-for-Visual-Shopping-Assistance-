# retriever.py
import os
from pinecone import Pinecone as PineconeClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel, ViltProcessor, ViltForImageAndTextRetrieval
from PIL import Image
import numpy as np

# --- Text Retrieval Configuration ---
TEXT_INDEX_NAME = "product-text-embeddings"
TEXT_EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/stsb-roberta-base'

# --- Image Retrieval (CLIP) Configuration ---
IMAGE_INDEX_NAME = "product-image-embeddings"
CLIP_MODEL_HF_ID = "openai/clip-vit-base-patch32"

# --- Image Reranking (ViLT) Configuration ---
VILT_RERANKER_MODEL_HF_ID = "dandelin/vilt-b32-finetuned-coco"

# --- Global variables ---
pc_client_text = None
pinecone_text_index = None
text_embedding_bi_encoder_model = None
text_cross_encoder_model = None

pc_client_image = None
pinecone_image_index = None
hf_clip_model = None
hf_clip_processor = None

hf_vilt_reranker_model = None
hf_vilt_reranker_processor = None

def load_env_vars_for_retriever():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Retriever Error: PINECONE_API_KEY not found in .env file.")
        return None
    return api_key

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
    global pc_client_text, pinecone_text_index, pc_client_image, pinecone_image_index
    
    pinecone_api_key = load_env_vars_for_retriever()
    if not pinecone_api_key: raise EnvironmentError("Retriever Error: Pinecone API key not configured.")

    if pc_client_text is None:
        try:
            print("Retriever: Initializing Pinecone client...")
            pc_client_text = PineconeClient(api_key=pinecone_api_key)
            pc_client_image = pc_client_text 
            print("Retriever: Pinecone client initialized.")
        except Exception as e:
            print(f"Retriever Error: Initializing Pinecone client: {e}"); raise

    if pinecone_text_index is None:
        try:
            print(f"Retriever: Connecting to Pinecone text index '{TEXT_INDEX_NAME}'...")
            pinecone_text_index = pc_client_text.Index(TEXT_INDEX_NAME)
            print(f"Retriever: Text index '{TEXT_INDEX_NAME}' stats: {pinecone_text_index.describe_index_stats()}")
        except Exception as e:
            print(f"Retriever Error: Connecting to Pinecone text index '{TEXT_INDEX_NAME}': {e}"); raise

    if pinecone_image_index is None:
        try:
            print(f"Retriever: Connecting to Pinecone image index '{IMAGE_INDEX_NAME}'...")
            pinecone_image_index = pc_client_image.Index(IMAGE_INDEX_NAME)
            print(f"Retriever: Image index '{IMAGE_INDEX_NAME}' stats: {pinecone_image_index.describe_index_stats()}")
        except Exception as e:
            print(f"Retriever Error: Connecting to Pinecone image index '{IMAGE_INDEX_NAME}': {e}"); raise

def initialize_retriever_resources(): 
    print("Retriever: Initializing ALL retriever resources (Models & Pinecone Connections)...")
    initialize_text_retrieval_models()
    initialize_clip_image_retrieval_models()
    initialize_vilt_reranker_models()
    initialize_pinecone_connections() 
    print("Retriever: All retriever resources initialized successfully.")

def retrieve_and_rerank_text_chunks(query_text, initial_top_k=20, final_top_k=5, filter_dict=None):
    global pinecone_text_index, text_embedding_bi_encoder_model, text_cross_encoder_model

    if not all([pinecone_text_index, text_embedding_bi_encoder_model, text_cross_encoder_model]):
        print("Retriever Error: Text retrieval/reranking resources not fully initialized.")
        return []

    # print(f"\n🔎 Retriever: Retrieving text for query: '{query_text}' with filter: {filter_dict}")
    try:
        query_embedding = text_embedding_bi_encoder_model.encode(query_text)
        query_response = pinecone_text_index.query(
            vector=query_embedding.tolist(), 
            top_k=initial_top_k,
            include_metadata=True,
            filter=filter_dict
        )
    except Exception as e:
        print(f"Retriever Error: During Pinecone text query or embedding: {e}"); return []

    initial_chunks = []
    if query_response and query_response.get('matches'):
        for match in query_response['matches']:
            initial_chunks.append({
                "id": match.get('id', 'N/A'),
                "text_content": str(match.get('metadata', {}).get('text_content', '')).strip(),
                "metadata": match.get('metadata', {}), 
                "pinecone_score": match.get('score', 0.0) 
            })
    if not initial_chunks:
        print("Retriever: No initial text matches found from Pinecone."); return []

    rerank_candidates = [chunk for chunk in initial_chunks if chunk['text_content']]
    if not rerank_candidates:
        print("Retriever: No valid text content found in initial matches for reranking.")
        initial_chunks.sort(key=lambda x: x.get('pinecone_score', 0.0), reverse=True)
        for chunk in initial_chunks: chunk['score'] = chunk.get('pinecone_score', 0.0)
        return initial_chunks[:final_top_k]

    sentence_pairs = [[query_text, chunk['text_content']] for chunk in rerank_candidates]
        
    # print(f"Retriever: Reranking {len(sentence_pairs)} text candidates with cross-encoder...")
    try:
        cross_encoder_scores = text_cross_encoder_model.predict(sentence_pairs, show_progress_bar=False)
        if not isinstance(cross_encoder_scores, np.ndarray):
            cross_encoder_scores = np.array(cross_encoder_scores)

        for i, chunk in enumerate(rerank_candidates): 
            chunk['score'] = float(cross_encoder_scores[i]) if not np.isnan(cross_encoder_scores[i]) else -float('inf')
            
    except Exception as e:
        print(f"Retriever Error: During cross-encoder prediction: {e}")
        for chunk in rerank_candidates: chunk['score'] = chunk.get('pinecone_score', 0.0)
        rerank_candidates.sort(key=lambda x: x['score'], reverse=True)
        return rerank_candidates[:final_top_k]

    all_scored_chunks = {chunk['id']: chunk for chunk in rerank_candidates} 

    for chunk in initial_chunks: 
        if chunk['id'] not in all_scored_chunks:
            chunk['score'] = chunk.get('pinecone_score', 0.0)
            all_scored_chunks[chunk['id']] = chunk
    
    final_sorted_chunks = sorted(list(all_scored_chunks.values()), key=lambda x: x.get('score', -float('inf')), reverse=True)
    # print(f"Retriever: Text reranking complete. Returning top {min(final_top_k, len(final_sorted_chunks))} results.")
    return final_sorted_chunks[:final_top_k]

def retrieve_relevant_images_from_text_clip(query_text, top_k=5, filter_dict=None):
    global pinecone_image_index, hf_clip_model, hf_clip_processor
    if not all([pinecone_image_index, hf_clip_model, hf_clip_processor]):
        print("Retriever Error: CLIP image retrieval resources not fully initialized.")
        return []

    # print(f"\n🖼️ Retriever: Retrieving images with CLIP for query: '{query_text}' with filter: {filter_dict}")
    try:
        inputs = hf_clip_processor(text=[query_text], return_tensors="pt", padding=True).to(hf_clip_model.device)
        with torch.no_grad(): text_features = hf_clip_model.get_text_features(**inputs)
        query_embedding = text_features[0].cpu().numpy()
        
        query_response = pinecone_image_index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
    except Exception as e:
        print(f"Retriever Error: During Pinecone image query or CLIP embedding: {e}"); return []

    retrieved_images = []
    if query_response and query_response.get('matches'):
        for match in query_response['matches']:
            metadata = match.get('metadata', {})
            retrieved_images.append({
                "id": match.get('id', metadata.get('image_path', 'N/A')), 
                "score": match.get('score', 0.0), 
                "image_path": metadata.get('image_path', match.get('id', 'N/A')),
                "product_id": metadata.get('product_id', 'N/A'),
                "primary_caption": metadata.get('primary_caption', 'N/A'),
                "all_captions": metadata.get('generated_captions_list', []) 
            })
    # if not retrieved_images: print("Retriever: No initial image matches found from CLIP search.")
    return retrieved_images

def rerank_images_with_vilt(query_text, candidate_images_data, top_k=2):
    global hf_vilt_reranker_model, hf_vilt_reranker_processor
    if not all([hf_vilt_reranker_model, hf_vilt_reranker_processor]):
        print("Retriever Warning: ViLT reranker model not initialized. Returning candidates sorted by original score.")
        return sorted(candidate_images_data, key=lambda x: x.get('score', 0.0), reverse=True)[:top_k]

    if not candidate_images_data: return []

    # print(f"\n⚡ Retriever: Reranking {len(candidate_images_data)} image candidates with ViLT for query: '{query_text}'")
    
    rerank_candidates_with_scores = []
    for item_data in candidate_images_data:
        image_path = item_data.get("image_path")
        
        caption_for_vilt = item_data.get("primary_caption", "")
        if not caption_for_vilt and item_data.get("all_captions"):
            caption_for_vilt = item_data["all_captions"][0] if item_data["all_captions"] else "product image"
        elif not caption_for_vilt:
            caption_for_vilt = "product image"

        if not image_path or not os.path.exists(image_path):
            # print(f"  Retriever ViLT Warning: Skipping missing image path: {image_path}"); 
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            
            query_tokens = query_text.split()
            caption_tokens = caption_for_vilt.split()
            processed_query = " ".join(query_tokens[:20]) 
            processed_caption = " ".join(caption_tokens[:18])

            if processed_query and processed_caption:
                text_input_for_vilt = f"{processed_query} [SEP] {processed_caption}"
            elif processed_query:
                text_input_for_vilt = processed_query
            else: 
                text_input_for_vilt = processed_caption if processed_caption else "image context"

            inputs = hf_vilt_reranker_processor(image, text_input_for_vilt, return_tensors="pt", padding="max_length", truncation=True, max_length=40).to(hf_vilt_reranker_model.device)
            
            with torch.no_grad():
                outputs = hf_vilt_reranker_model(**inputs)
            
            vilt_relevance_score = outputs.logits[0, 0].item() 
            
            item_data['vilt_score'] = vilt_relevance_score
            rerank_candidates_with_scores.append(item_data)

        except Exception as e:
            print(f"  Retriever ViLT Error: Reranking image {os.path.basename(image_path)}: {e}")
            item_data['vilt_score'] = -float('inf') 
            rerank_candidates_with_scores.append(item_data)

    reranked_items = sorted(rerank_candidates_with_scores, key=lambda x: x.get("vilt_score", -float('inf')), reverse=True)
    # print(f"Retriever: ViLT Image Reranking complete. Returning top {min(top_k, len(reranked_items))} results.")
    return reranked_items[:top_k]

retrieve_relevant_chunks = retrieve_and_rerank_text_chunks

if __name__ == '__main__':
    try:
        print("Retriever Test: Initializing ALL retriever resources...")
        initialize_retriever_resources()
        print("Retriever Test: All retriever resources initialized.")
    except Exception as e:
        print(f"Retriever Test Error: Could not initialize retriever: {e}"); exit()

    test_query = "Are Sony WH-1000XM4 headphones good for flights and noise cancellation?"
    test_filter = None 

    print(f"\n{'='*20} Retriever Test: TEXT RETRIEVAL: '{test_query}' {'='*20}")
    retrieved_texts = retrieve_relevant_chunks(test_query, initial_top_k=5, final_top_k=2, filter_dict=test_filter)
    if retrieved_texts:
        print(f"\n--- Top {len(retrieved_texts)} RERANKED Text Chunks ---")
        for i, chunk in enumerate(retrieved_texts):
            print(f"  Text Result {i+1}: ID: {chunk['id']}, Score: {chunk.get('score',0.0):.4f}, Prod: {chunk['metadata'].get('product_id')}, Text: {chunk['text_content'][:60]}...")
    else:
        print("Retriever Test: No text chunks retrieved.")

    print(f"\n{'='*20} Retriever Test: IMAGE RETRIEVAL (CLIP): '{test_query}' {'='*20}")
    clip_retrieved_images = retrieve_relevant_images_from_text_clip(test_query, top_k=3, filter_dict=test_filter)
    if clip_retrieved_images:
        print(f"\n--- Top {len(clip_retrieved_images)} CLIP Retrieved Images ---")
        for i, img_info in enumerate(clip_retrieved_images):
            print(f"  Image {i+1}: Path: {os.path.basename(img_info['image_path'])}, CLIP_Score: {img_info['score']:.4f}, Prod: {img_info['product_id']}")

        print(f"\n{'='*20} Retriever Test: IMAGE RERANKING (ViLT) on CLIP results {'='*20}")
        vilt_reranked_images = rerank_images_with_vilt(test_query, clip_retrieved_images, top_k=2)
        if vilt_reranked_images:
            print(f"\n--- Top {len(vilt_reranked_images)} ViLT RERANKED Images ---")
            for i, img_info in enumerate(vilt_reranked_images):
                print(f"  Reranked Image {i+1}: Path: {os.path.basename(img_info['image_path'])}, ViLT_Score: {img_info.get('vilt_score',0.0):.4f}, Prod: {img_info['product_id']}")
    else:
        print("Retriever Test: No images retrieved by CLIP to rerank with ViLT.")