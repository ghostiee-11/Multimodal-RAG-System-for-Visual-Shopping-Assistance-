# main_assistant.py
import os
import pandas as pd
from dotenv import load_dotenv 
import re 
from rapidfuzz import process, fuzz, utils as rapidfuzz_utils
import json

# Custom module imports
from data_loader import load_and_clean_data 
from retriever_new import (
    initialize_retriever_resources, 
    retrieve_relevant_chunks, 
    retrieve_relevant_images_from_text_clip,
    rerank_images_with_vilt
)
from llm_handler_new import (
    configure_gemini, 
    generate_answer_with_gemini, 
    parse_user_query_with_gemini,
    format_conversation_history_for_prompt 
)

# --- Configuration ---
PRODUCTS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/products_final.csv'
REVIEWS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/customer_reviews.csv' 
ALL_DOCS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/all_documents.csv'   
IMAGE_BASE_PATH = '/Users/amankumar/Desktop/Aims/final data/images/' 
IMAGE_CAPTIONS_CSV_PATH = 'image_captions_multiple.csv' 

df_products_global = None
df_image_captions_global = None
MAX_CONVERSATION_HISTORY_TURNS = 3 # Number of past user/assistant turns to remember

def initial_setup():
    global df_products_global, df_image_captions_global
    print("Main: Initializing RAG system resources...")
    load_dotenv() 
    if not os.getenv("PINECONE_API_KEY"): print("Main CRITICAL ERROR: PINECONE_API_KEY not found."); return False
    if not os.getenv("GOOGLE_API_KEY"): print("Main CRITICAL ERROR: GOOGLE_API_KEY not found."); return False
    
    try:
        configure_gemini() 
        initialize_retriever_resources() 
        
        df_prods_temp, _, _ = load_and_clean_data(PRODUCTS_CSV_PATH, REVIEWS_CSV_PATH, ALL_DOCS_CSV_PATH)
        if df_prods_temp is None: raise FileNotFoundError(f"Main Error: Failed to load product data from {PRODUCTS_CSV_PATH}")
        df_products_global = df_prods_temp
        print(f"Main: Product metadata (df_products_global) loaded. Shape: {df_products_global.shape}")
        
        if 'price' in df_products_global.columns:
            df_products_global['price_numeric'] = pd.to_numeric(
                df_products_global['price'].astype(str).str.replace('₹', '').str.replace(',', ''), errors='coerce'
            )
        else: df_products_global['price_numeric'] = pd.NA

        try:
            df_image_captions_global = pd.read_csv(IMAGE_CAPTIONS_CSV_PATH)
            if 'full_image_path' not in df_image_captions_global.columns:
                raise ValueError("'full_image_path' column missing in image_captions_multiple.csv")

            if 'generated_captions_json' in df_image_captions_global.columns:
                 df_image_captions_global['all_captions_list'] = df_image_captions_global['generated_captions_json'].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else [])
            elif 'generated_caption' in df_image_captions_global.columns: 
                 df_image_captions_global['all_captions_list'] = df_image_captions_global['generated_caption'].apply(lambda x: [x] if pd.notna(x) else [])
            else:
                df_image_captions_global['all_captions_list'] = [[] for _ in range(len(df_image_captions_global))]

            if 'primary_caption' not in df_image_captions_global.columns:
                df_image_captions_global['primary_caption'] = df_image_captions_global['all_captions_list'].apply(lambda x: x[0] if x and isinstance(x, list) and len(x) > 0 else "Caption unavailable.")

            print(f"Main: Image captions (df_image_captions_global) loaded. Shape: {df_image_captions_global.shape}")
        except FileNotFoundError:
            print(f"Main Warning: Image captions CSV '{IMAGE_CAPTIONS_CSV_PATH}' not found. Image context will be limited.")
            df_image_captions_global = pd.DataFrame(columns=['full_image_path', 'primary_caption', 'all_captions_list'])
        except Exception as e_cap:
            print(f"Main Error loading or processing image captions: {e_cap}")
            df_image_captions_global = pd.DataFrame(columns=['full_image_path', 'primary_caption', 'all_captions_list'])

        print("Main: RAG system resources initialized successfully.")
        return True
    except Exception as e:
        print(f"Main Critical Error: During RAG system initialization: {e}"); return False

def map_entities_to_product_ids(parsed_query_info, df_all_products):
    if df_all_products is None or df_all_products.empty: return []
    product_entities_raw = parsed_query_info.get("product_entities", [])
    brand_entities_raw = parsed_query_info.get("brand_entities", [])
    product_entities = [str(e).lower().strip() for e in product_entities_raw if str(e).strip()]
    brand_entities = [str(b).lower().strip() for b in brand_entities_raw if str(b).strip()]
    candidate_pids = set()
    if 'title' not in df_all_products.columns: return []
    
    temp_df = df_all_products[['product_id', 'title']].copy()
    temp_df['title_lower'] = temp_df['title'].astype(str).str.lower()
    choices_titles_only = temp_df['title_lower'].tolist()
    product_id_list_for_index = temp_df['product_id'].tolist()

    if not choices_titles_only : return []
    if product_entities:
        for entity_model_name in product_entities:
            processed_entity = rapidfuzz_utils.default_process(entity_model_name)
            best_match = process.extractOne(processed_entity, choices_titles_only, scorer=fuzz.WRatio, score_cutoff=80) 
            if best_match:
                match_title_lower, score, index = best_match
                pid_candidate = product_id_list_for_index[index]
                entity_brands_lower = [b for b in brand_entities if b in entity_model_name]
                if not entity_brands_lower: entity_brands_lower = brand_entities
                brand_confirmed = True
                if entity_brands_lower: brand_confirmed = any(brand in match_title_lower for brand in entity_brands_lower)
                if brand_confirmed: candidate_pids.add(pid_candidate)
    if not candidate_pids and brand_entities: 
        for brand_name in brand_entities:
            for idx, row in temp_df.iterrows():
                if brand_name in row['title_lower']: candidate_pids.add(row['product_id'])
    return list(candidate_pids)

def parse_constraints_from_features(key_features_attributes):
    constraints = {"price_min": None, "price_max": None, "color": None, "type": None, "other_features_text": ""}
    other_features_list = []
    color_keywords = ["blue","black","red","white","grey","gray","green","silver","beige","aqua","crimson","purple","pink","yellow","orange"]
    type_keywords = ["over-ear","on-ear","in-ear","neckband","earbuds","tws"]
    for feature in key_features_attributes:
        feature_lower = str(feature).lower()
        price_max_match = re.search(r"(?:price under|less than|below|max price|upto|maximum|under rs)\s*₹?\s*(\d[\d,]*\.?\d*)", feature_lower)
        price_min_match = re.search(r"(?:price over|more than|above|min price|minimum|starting at|starting from)\s*₹?\s*(\d[\d,]*\.?\d*)", feature_lower)
        if price_max_match: constraints["price_max"] = float(price_max_match.group(1).replace(',', ''))
        elif price_min_match: constraints["price_min"] = float(price_min_match.group(1).replace(',', ''))
        elif feature_lower.startswith("color:"): constraints["color"] = feature_lower.split(":", 1)[1].strip()
        elif feature_lower.startswith("type:"): constraints["type"] = feature_lower.split(":", 1)[1].strip()
        else:
            found_color = next((c for c in color_keywords if c in feature_lower.split()), None)
            found_type = next((t for t in type_keywords if t in feature_lower), None)
            if found_color and not constraints["color"]: constraints["color"] = found_color
            elif found_type and not constraints["type"]: constraints["type"] = found_type
            else: other_features_list.append(feature)
    constraints["other_features_text"] = " ".join(other_features_list).strip()
    return constraints

def filter_products_by_constraints(df_prods, constraints_dict):
    if df_prods is None or df_prods.empty: return pd.DataFrame(columns=df_prods.columns if df_prods is not None else [])
    filtered_df = df_prods.copy()
    if 'price_numeric' not in filtered_df.columns: print("Main Warning: 'price_numeric' column not found for filtering by price.")
    else:
        if constraints_dict.get("price_min") is not None: filtered_df = filtered_df[filtered_df['price_numeric'].notna() & (filtered_df['price_numeric'] >= constraints_dict["price_min"])]
        if constraints_dict.get("price_max") is not None: filtered_df = filtered_df[filtered_df['price_numeric'].notna() & (filtered_df['price_numeric'] <= constraints_dict["price_max"])]
    if constraints_dict.get("color") and 'title' in filtered_df.columns: 
        filtered_df = filtered_df[filtered_df['title'].astype(str).str.lower().str.contains(constraints_dict["color"].lower(), na=False, regex=False)]
    if constraints_dict.get("type") and 'title' in filtered_df.columns: 
        type_regex = r'\b' + re.escape(constraints_dict["type"].lower()) + r'\b'
        type_condition = pd.Series([False] * len(filtered_df), index=filtered_df.index)
        if 'product_type' in filtered_df.columns and filtered_df['product_type'].notna().any(): 
            type_condition |= filtered_df['product_type'].astype(str).str.lower().str.contains(type_regex, na=False, regex=True)
        type_condition |= filtered_df['title'].astype(str).str.lower().str.contains(type_regex, na=False, regex=True)
        filtered_df = filtered_df[type_condition]
    return filtered_df

def get_captions_for_image_path(image_full_path):
    global df_image_captions_global
    default_primary = "Caption unavailable."
    default_all = []
    if df_image_captions_global is None or df_image_captions_global.empty or 'full_image_path' not in df_image_captions_global.columns:
        return default_primary, default_all
    caption_row_df = df_image_captions_global[df_image_captions_global['full_image_path'] == image_full_path]
    if not caption_row_df.empty:
        caption_row = caption_row_df.iloc[0]
        all_captions = caption_row.get('all_captions_list', [])
        if not isinstance(all_captions, list): all_captions = [str(all_captions)] if pd.notna(all_captions) else []
        primary_caption = caption_row.get('primary_caption', default_primary)
        if (primary_caption == default_primary or pd.isna(primary_caption)) and all_captions: primary_caption = all_captions[0]
        if (primary_caption == default_primary) and ('generated_caption' in caption_row and pd.notna(caption_row['generated_caption'])): # Fallback for older single caption CSV
            primary_caption = str(caption_row['generated_caption'])
            if not all_captions: all_captions = [primary_caption]
        return primary_caption, all_captions
    return default_primary, default_all

def get_product_details_and_associated_images(product_id, direct_image_matches_lookup):
    global df_products_global, IMAGE_BASE_PATH
    product_info_dict = None; images_info_list = []
    if not product_id or df_products_global is None: return product_info_dict, images_info_list
    product_row_df = df_products_global[df_products_global['product_id'] == product_id]
    if product_row_df.empty: return product_info_dict, images_info_list
    product_row = product_row_df.iloc[0]
    product_info_dict = {"title": product_row.get('title'), "price": product_row.get('price'), "product_type": product_row.get('product_type')}
    added_image_paths = set()
    for img_path_lookup, img_data_lookup in direct_image_matches_lookup.items():
        if len(images_info_list) >= 2: break
        if img_data_lookup.get('product_id') == product_id:
            images_info_list.append({"image_path": img_data_lookup['image_path'], "filename": os.path.basename(img_data_lookup['image_path']),
                                     "primary_caption": img_data_lookup.get('primary_caption', "N/A"),
                                     "all_captions": img_data_lookup.get('all_captions', [])})
            added_image_paths.add(img_data_lookup['image_path'])
    if len(images_info_list) < 2 and pd.notna(product_row.get('image_paths')):
        general_image_filenames_str = str(product_row.get('image_paths'))
        for rel_path in general_image_filenames_str.split(','):
            if len(images_info_list) >= 2: break
            img_filename = rel_path.strip()
            if not img_filename: continue
            full_path = os.path.join(IMAGE_BASE_PATH, img_filename)
            if full_path not in added_image_paths:
                primary_cap, all_caps = get_captions_for_image_path(full_path)
                images_info_list.append({"image_path": full_path, "filename": img_filename,
                                         "primary_caption": primary_cap, "all_captions": all_caps})
                added_image_paths.add(full_path)
    return product_info_dict, images_info_list

def assemble_llm_context(retrieved_texts, vilt_reranked_images, max_total_context_items=4):
    global df_products_global
    if df_products_global is None: print("Main Error: Product metadata not loaded."); return []
    final_context_items = []
    vilt_images_lookup = {img['image_path']: img for img in vilt_reranked_images if img.get('image_path')}
    processed_product_ids = set()
    if retrieved_texts:
        for text_chunk in retrieved_texts:
            if len(final_context_items) >= max_total_context_items: break
            pid = text_chunk['metadata'].get('product_id')
            if not pid: continue
            product_details, associated_images_for_text = get_product_details_and_associated_images(pid, vilt_images_lookup)
            final_context_items.append({"type": "text_derived_context", "text_content": text_chunk['text_content'], 
                                      "text_score": text_chunk.get('score', 0.0), "text_metadata_details": text_chunk['metadata'], 
                                      "associated_product_id": pid, "associated_product_info": product_details, 
                                      "associated_images": associated_images_for_text })
            processed_product_ids.add(pid)
    if vilt_reranked_images:
        for img_data in vilt_reranked_images: 
            if len(final_context_items) >= max_total_context_items: break
            pid = img_data.get('product_id'); img_path = img_data.get('image_path')
            if not pid or not img_path: continue
            if pid not in processed_product_ids:
                product_details, _ = get_product_details_and_associated_images(pid, {}) 
                final_context_items.append({"type": "image_derived_context", "image_path": img_path, 
                                          "primary_caption": img_data.get('primary_caption', "N/A"),
                                          "all_captions": img_data.get('all_captions', []),
                                          "image_score": img_data.get('score', 0.0), "vilt_score": img_data.get('vilt_score'), 
                                          "associated_product_id": pid, "associated_product_info": product_details,
                                          "associated_images": [{"image_path": img_path, "filename": os.path.basename(img_path),
                                                               "primary_caption": img_data.get('primary_caption', "N/A"),
                                                               "all_captions": img_data.get('all_captions', [])}]})
                processed_product_ids.add(pid)
    return final_context_items[:max_total_context_items]

if __name__ == '__main__':
    if not initial_setup(): exit()
    conversation_history = [] 

    while True:
        user_query_original = input("\n🛍️ Enter your query (or type 'quit' to exit): ").strip()
        if not user_query_original: continue
        if user_query_original.lower() == 'quit': break

        print(f"\n🔎 Main: Original User Query: '{user_query_original}'")
        history_for_prompt_str = format_conversation_history_for_prompt(conversation_history, MAX_CONVERSATION_HISTORY_TURNS)
        
        query_for_processing = user_query_original # No translation in this version

        parsed_query_info = parse_user_query_with_gemini(query_for_processing, history_for_prompt_str)
        # parsed_query_info["original_query_language"] = "en" # Assuming English
        
        intent = parsed_query_info.get("intent_type", "GENERAL_PRODUCT_SEARCH")
        key_features_attributes = parsed_query_info.get("key_features_attributes", [])
        comparison_entities = parsed_query_info.get("comparison_entities", [])
        product_entities = parsed_query_info.get("product_entities", [])
        brand_entities = parsed_query_info.get("brand_entities", [])
        
        retrieval_query_base = parsed_query_info.get("rewritten_query_for_retrieval", query_for_processing)
        if not retrieval_query_base or retrieval_query_base == "N/A":
            retrieval_query_base = query_for_processing
        # print(f"Main: Using base query for retrieval: '{retrieval_query_base}'")

        final_llm_context_list = []
        max_items_per_comp_product = 2 
        max_items_general_query = 3    

        if intent == "PRODUCT_COMPARISON" and len(comparison_entities) >= 2:
            # print(f"Main: Comparison detected. Comparing: '{', '.join(comparison_entities)}'")
            decomposition_hints = parsed_query_info.get("decomposition_hints", [])
            
            # Simplified: Use original comparison_entities if decomposition_hints is complex to implement fully now
            # For a more robust solution, the loop would iterate through decomposition_hints if present
            # or fallback to comparison_entities. For now, using comparison_entities.
            for entity_name_original in comparison_entities:
                entity_name = str(entity_name_original).strip()
                # print(f"\n--- Main: Retrieving info for comparison entity: '{entity_name}' ---")
                temp_parsed_info_for_entity = {"product_entities": [entity_name], "brand_entities": []}
                entity_name_parts = entity_name.split(); 
                if entity_name_parts: temp_parsed_info_for_entity["brand_entities"].append(entity_name_parts[0])
                pids_for_this_entity = map_entities_to_product_ids(temp_parsed_info_for_entity, df_products_global)
                
                parsed_constraints_for_query = parse_constraints_from_features(key_features_attributes)
                if pids_for_this_entity and df_products_global is not None and not df_products_global[df_products_global['product_id'].isin(pids_for_this_entity)].empty:
                    temp_df_for_entity = df_products_global[df_products_global['product_id'].isin(pids_for_this_entity)]
                    filtered_pids_by_constraint = filter_products_by_constraints(temp_df_for_entity, parsed_constraints_for_query)['product_id'].tolist()
                    if filtered_pids_by_constraint : pids_for_this_entity = filtered_pids_by_constraint
                
                entity_pinecone_filter = {"product_id": {"$in": pids_for_this_entity[:10]}} if pids_for_this_entity else None
                # Use a combined query for this entity
                retrieval_query_for_entity = f"{entity_name} {' '.join(parsed_query_info.get('key_features_attributes',[]))} {parsed_constraints_for_query.get('other_features_text','')}".strip().replace("  ", " ")
                if not retrieval_query_for_entity.replace(entity_name,"").strip() : retrieval_query_for_entity = f"{entity_name} details"
                
                texts_entity = retrieve_relevant_chunks(retrieval_query_for_entity, initial_top_k=5, final_top_k=max_items_per_comp_product, filter_dict=entity_pinecone_filter)
                images_clip_entity = retrieve_relevant_images_from_text_clip(retrieval_query_for_entity, top_k=3, filter_dict=entity_pinecone_filter) 
                images_vilt_entity = rerank_images_with_vilt(retrieval_query_for_entity, images_clip_entity, top_k=1) if images_clip_entity else []
                context_entity_items = assemble_llm_context(texts_entity, images_vilt_entity, max_total_context_items=max_items_per_comp_product)
                if context_entity_items:
                    final_llm_context_list.append({"type": "comparison_intro", "product_name": entity_name})
                    final_llm_context_list.extend(context_entity_items)
        
        else: 
            target_pids_from_entities = map_entities_to_product_ids(parsed_query_info, df_products_global)
            parsed_constraints = parse_constraints_from_features(key_features_attributes) 
            candidate_products_df_for_filter = df_products_global
            price_constraints_exist = parsed_constraints.get("price_min") is not None or parsed_constraints.get("price_max") is not None
            color_constraint_exists = parsed_constraints.get("color") is not None
            type_constraint_exists = parsed_constraints.get("type") is not None
            if price_constraints_exist or color_constraint_exists or type_constraint_exists:
                candidate_products_df_for_filter = filter_products_by_constraints(df_products_global, parsed_constraints)
            
            final_retrieval_pids = []
            if candidate_products_df_for_filter is not None and not candidate_products_df_for_filter.empty:
                if target_pids_from_entities: 
                    final_retrieval_pids = list(set(target_pids_from_entities) & set(candidate_products_df_for_filter['product_id'].tolist()))
                    if not final_retrieval_pids: final_retrieval_pids = candidate_products_df_for_filter['product_id'].tolist() if not candidate_products_df_for_filter.empty else target_pids_from_entities
                else: final_retrieval_pids = candidate_products_df_for_filter['product_id'].tolist()
            elif target_pids_from_entities: final_retrieval_pids = target_pids_from_entities
            
            pinecone_filter = {"product_id": {"$in": final_retrieval_pids[:20]}} if final_retrieval_pids else None
            
            semantic_query_parts = [retrieval_query_base]
            for pe_part in product_entities:
                if pe_part.lower() not in retrieval_query_base.lower(): semantic_query_parts.append(pe_part)
            for be_part in brand_entities:
                if be_part.lower() not in retrieval_query_base.lower(): semantic_query_parts.append(be_part)
            other_f_text = parsed_constraints.get('other_features_text','')
            if other_f_text and other_f_text.lower() not in retrieval_query_base.lower(): semantic_query_parts.append(other_f_text)
            retrieval_query_effective = " ".join(list(dict.fromkeys(filter(None, semantic_query_parts)))).strip()
            if not retrieval_query_effective: retrieval_query_effective = query_for_processing

            # print(f"Main: Effective SEMANTIC retrieval query: '{retrieval_query_effective}'")
            # print(f"Main: Using METADATA filter for Pinecone: {pinecone_filter}")
            texts_retrieved = retrieve_relevant_chunks(retrieval_query_effective, initial_top_k=10, final_top_k=max_items_general_query, filter_dict=pinecone_filter)
            image_retrieval_query_final = retrieval_query_effective
            if parsed_query_info.get("visual_aspects_queried"):
                image_retrieval_query_final += " " + " ".join(parsed_query_info["visual_aspects_queried"])
            images_clip_retrieved = retrieve_relevant_images_from_text_clip(image_retrieval_query_final.strip(), top_k=5, filter_dict=pinecone_filter) 
            images_vilt_reranked = rerank_images_with_vilt(image_retrieval_query_final.strip(), images_clip_retrieved, top_k=2) if images_clip_retrieved else []
            final_llm_context_list = assemble_llm_context(texts_retrieved, images_vilt_reranked, max_total_context_items=max_items_general_query)
        
        if final_llm_context_list or conversation_history:
            # (Your detailed print logic for context - can be uncommented for debugging)
            # print(f"\n📋 --- Main: Assembled Context for LLM (Query: '{query_for_processing}', Total Items: {len(final_llm_context_list)}) ---")
            # ...
            
            print("\n🤖 Main: Sending context and history to LLM for answer generation...")
            llm_answer_english = generate_answer_with_gemini(
                query_for_processing, 
                final_llm_context_list, 
                parsed_query_info,
                conversation_history_str=history_for_prompt_str
            )
            
            final_answer_to_user = llm_answer_english # No translation back in this version
            print("\n💡 --- Shopping Assistant's Answer ---")
            print(final_answer_to_user)
            conversation_history.append((user_query_original, final_answer_to_user))
            if len(conversation_history) > MAX_CONVERSATION_HISTORY_TURNS * 2: 
                conversation_history = conversation_history[-MAX_CONVERSATION_HISTORY_TURNS:]
        else:
            no_info_message = "I'm sorry, I couldn't find enough information to answer that right now."
            print(f"\n🤷 Main: {no_info_message}")
            conversation_history.append((user_query_original, no_info_message))
            
    print("\nExiting Visual Shopping Assistant. Goodbye!")