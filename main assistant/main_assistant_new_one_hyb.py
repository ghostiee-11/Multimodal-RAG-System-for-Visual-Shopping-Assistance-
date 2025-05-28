# main_assistant_new_one_hyb.py
import os
import pandas as pd
from dotenv import load_dotenv 
import re 
from rapidfuzz import process, fuzz, utils as rapidfuzz_utils
import json

# Custom module imports
from data_loader import load_and_clean_data 
from retriever_new_one_hyb import (
    initialize_retriever_resources as initialize_retriever_resources_v3, 
    retrieve_relevant_chunks as retrieve_relevant_chunks_v3, 
    retrieve_relevant_images_from_text_clip as retrieve_relevant_images_from_text_clip_v3,
    rerank_images_with_vilt as rerank_images_with_vilt_v3
)
from llm_handler_new_one import (
    configure_gemini as configure_gemini_v3, 
    generate_answer_with_gemini as generate_answer_with_gemini_v3, 
    parse_user_query_with_gemini as parse_user_query_with_gemini_v3,
    format_conversation_history_for_prompt as format_conversation_history_for_prompt_v3
)

# --- Configuration ---
PRODUCTS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/products_final.csv'
REVIEWS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/customer_reviews.csv' 
ALL_DOCS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/all_documents.csv'   
IMAGE_BASE_PATH = '/Users/amankumar/Desktop/Aims/final data/images/' 

# This should point to the CSV that has the BLIP captions / filtered texts from generate_image_textual_metadata.py
IMAGE_CAPTIONS_CSV_PATH = 'image_combined_blip_ocr_filtered_final.csv' 

df_products_global = None 
df_image_captions_global = None 
MAX_CONVERSATION_HISTORY_TURNS = 3

def initial_setup():
    global df_products_global, df_image_captions_global
    print("Main (V3): Initializing RAG system resources...")
    load_dotenv() 
    if not os.getenv("PINECONE_API_KEY"): 
        print("Main (V3) CRITICAL ERROR: PINECONE_API_KEY not found in .env file.")
        return False
    if not os.getenv("GOOGLE_API_KEY"): 
        print("Main (V3) CRITICAL ERROR: GOOGLE_API_KEY not found in .env file.")
        return False
    
    try:
        configure_gemini_v3() 
        initialize_retriever_resources_v3() 
        
        df_prods_temp, _, _ = load_and_clean_data(PRODUCTS_CSV_PATH, REVIEWS_CSV_PATH, ALL_DOCS_CSV_PATH)
        if df_prods_temp is None: 
            raise FileNotFoundError(f"Main (V3) Error: Failed to load product data from {PRODUCTS_CSV_PATH}")
        df_products_global = df_prods_temp # Assign to the module-level global
        print(f"Main (V3): Product metadata (df_products_global) loaded. Shape: {df_products_global.shape}")
        
        if 'price' in df_products_global.columns:
            df_products_global['price_numeric'] = pd.to_numeric(
                df_products_global['price'].astype(str).str.replace('₹', '').str.replace(',', ''), errors='coerce'
            )
        else: 
            df_products_global['price_numeric'] = pd.NA
            print("Main (V3) Warning: 'price' column not found in products CSV. Price filtering will be affected.")


        try:
            df_image_captions_global_temp = pd.read_csv(IMAGE_CAPTIONS_CSV_PATH)
            if 'full_image_path' not in df_image_captions_global_temp.columns:
                raise ValueError(f"'full_image_path' column missing in {IMAGE_CAPTIONS_CSV_PATH}")
            if 'generated_texts_json' not in df_image_captions_global_temp.columns: # This is the V3 column
                raise ValueError(f"'generated_texts_json' column missing in {IMAGE_CAPTIONS_CSV_PATH}. This column should contain a JSON list of text items (e.g., from BLIP/OCR).")

            # 'generated_texts_json' for V3 is a list of strings
            df_image_captions_global_temp['all_captions_list_parsed'] = df_image_captions_global_temp['generated_texts_json'].apply(
                lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else []
            )
            
            def get_primary_from_string_list_v3(text_list):
                if text_list and isinstance(text_list, list) and len(text_list) > 0 and isinstance(text_list[0], str):
                    return text_list[0]
                return "Caption unavailable."
            df_image_captions_global_temp['primary_caption_parsed'] = df_image_captions_global_temp['all_captions_list_parsed'].apply(get_primary_from_string_list_v3)
            df_image_captions_global = df_image_captions_global_temp # Assign to module-level global

            print(f"Main (V3): Image textual metadata (df_image_captions_global from {IMAGE_CAPTIONS_CSV_PATH}) loaded. Shape: {df_image_captions_global.shape}")
        except FileNotFoundError:
            print(f"Main (V3) Warning: Image textual metadata CSV '{IMAGE_CAPTIONS_CSV_PATH}' not found. Image context will be limited.")
            df_image_captions_global = pd.DataFrame(columns=['full_image_path', 'product_id', 'primary_caption_parsed', 'all_captions_list_parsed'])
        except Exception as e_cap:
            print(f"Main (V3) Error loading or processing image textual metadata from {IMAGE_CAPTIONS_CSV_PATH}: {e_cap}")
            df_image_captions_global = pd.DataFrame(columns=['full_image_path', 'product_id', 'primary_caption_parsed', 'all_captions_list_parsed'])

        print("Main (V3): RAG system resources initialized successfully.")
        return True
    except Exception as e:
        print(f"Main (V3) Critical Error: During RAG system initialization: {e}"); return False

def map_entities_to_product_ids(parsed_query_info, df_all_products):
    # This function remains the same as in main_assistant_new_one.py (provided previously)
    if df_all_products is None or df_all_products.empty: return []
    product_entities_raw = parsed_query_info.get("product_entities", [])
    brand_entities_raw = parsed_query_info.get("brand_entities", [])
    product_entities = [str(e).lower().strip() for e in product_entities_raw if str(e).strip()]
    brand_entities = [str(b).lower().strip() for b in brand_entities_raw if str(b).strip()]
    candidate_pids = set()
    if 'title' not in df_all_products.columns: 
        print("Main Warning: 'title' column not found in products for entity mapping.")
        return []
    
    temp_df = df_all_products[['product_id', 'title']].copy()
    temp_df['title_lower'] = temp_df['title'].astype(str).str.lower()
    choices_titles_only = temp_df['title_lower'].tolist()
    product_id_list_for_index = temp_df['product_id'].tolist()

    if not choices_titles_only : return []
    if product_entities:
        for entity_model_name in product_entities:
            processed_entity = rapidfuzz_utils.default_process(entity_model_name)
            best_match = process.extractOne(processed_entity, choices_titles_only, scorer=fuzz.WRatio, score_cutoff=85) 
            if best_match:
                match_title_lower, score, index = best_match
                pid_candidate = product_id_list_for_index[index]
                entity_brands_lower = [b for b in brand_entities if b in entity_model_name] 
                if not entity_brands_lower: entity_brands_lower = brand_entities 

                brand_confirmed = True 
                if entity_brands_lower: 
                    brand_confirmed = any(brand in match_title_lower for brand in entity_brands_lower)
                
                if brand_confirmed: 
                    candidate_pids.add(pid_candidate)
    if not candidate_pids and brand_entities: 
        for brand_name in brand_entities:
            brand_matches = temp_df[temp_df['title_lower'].str.contains(brand_name, na=False)]['product_id'].tolist()
            candidate_pids.update(brand_matches[:5]) 
    return list(candidate_pids)

def parse_constraints_from_features(key_features_attributes):
    # This function remains the same as in main_assistant_new_one.py (provided previously)
    constraints = {"price_min": None, "price_max": None, "color": None, "type": None, "other_features_text": ""}
    other_features_list = []
    color_keywords = ["blue","black","red","white","grey","gray","green","silver","beige","aqua","crimson","purple","pink","yellow","orange", "gold", "brown", "cream"]
    type_keywords = ["over-ear","on-ear","in-ear","neckband","earbuds","tws", "true wireless", "wired", "wireless"]
    for feature in key_features_attributes:
        feature_lower = str(feature).lower().strip()
        if not feature_lower: continue
        price_max_match = re.search(r"(?:price\s*(?:under|less than|below|max|upto|maximum)|under\s*(?:rs|₹))\s*₹?\s*(\d[\d,]*\.?\d*)", feature_lower)
        price_min_match = re.search(r"(?:price\s*(?:over|more than|above|min|minimum)|(?:starting|from)\s*(?:rs|₹))\s*₹?\s*(\d[\d,]*\.?\d*)", feature_lower)
        if price_max_match: constraints["price_max"] = float(price_max_match.group(1).replace(',', ''))
        elif price_min_match: constraints["price_min"] = float(price_min_match.group(1).replace(',', ''))
        elif feature_lower.startswith("color:"): constraints["color"] = feature_lower.split(":", 1)[1].strip()
        elif feature_lower.startswith("type:"): constraints["type"] = feature_lower.split(":", 1)[1].strip()
        else:
            found_color = next((c for c in color_keywords if re.search(r'\b' + re.escape(c) + r'\b', feature_lower)), None)
            found_type = next((t for t in type_keywords if re.search(r'\b' + re.escape(t) + r'\b', feature_lower)), None)
            if found_color and not constraints["color"]: constraints["color"] = found_color
            elif found_type and not constraints["type"]: constraints["type"] = found_type
            else: other_features_list.append(feature.strip())
    constraints["other_features_text"] = " ".join(other_features_list).strip()
    return constraints

def filter_products_by_constraints(df_prods, constraints_dict):
    # This function remains the same as in main_assistant_new_one.py (provided previously)
    if df_prods is None or df_prods.empty: 
        return pd.DataFrame(columns=df_prods.columns if df_prods is not None else [])
    filtered_df = df_prods.copy()
    if 'price_numeric' not in filtered_df.columns: 
        print("Main Warning: 'price_numeric' column not found for filtering by price.")
    else:
        if constraints_dict.get("price_min") is not None: 
            filtered_df = filtered_df[filtered_df['price_numeric'].notna() & (filtered_df['price_numeric'] >= constraints_dict["price_min"])]
        if constraints_dict.get("price_max") is not None: 
            filtered_df = filtered_df[filtered_df['price_numeric'].notna() & (filtered_df['price_numeric'] <= constraints_dict["price_max"])]
    if constraints_dict.get("color") and 'title' in filtered_df.columns: 
        color_regex = r'\b' + re.escape(constraints_dict["color"].lower()) + r'\b'
        filtered_df = filtered_df[filtered_df['title'].astype(str).str.lower().str.contains(color_regex, na=False, regex=True)]
    if constraints_dict.get("type") and 'title' in filtered_df.columns: 
        type_regex = r'\b' + re.escape(constraints_dict["type"].lower()) + r'\b'
        type_condition = pd.Series([False] * len(filtered_df), index=filtered_df.index)
        if 'product_type' in filtered_df.columns and filtered_df['product_type'].notna().any(): 
            type_condition |= filtered_df['product_type'].astype(str).str.lower().str.contains(type_regex, na=False, regex=True)
        type_condition |= filtered_df['title'].astype(str).str.lower().str.contains(type_regex, na=False, regex=True)
        filtered_df = filtered_df[type_condition]
    return filtered_df

def get_captions_for_image_path(image_full_path): # V3 specific way to use df_image_captions_global
    global df_image_captions_global # This is df_image_captions_global_v3
    default_primary = "Caption unavailable."
    default_all_text_items_list = [] 

    if df_image_captions_global is None or df_image_captions_global.empty or 'full_image_path' not in df_image_captions_global.columns:
        return default_primary, default_all_text_items_list
        
    caption_row_df = df_image_captions_global[df_image_captions_global['full_image_path'] == image_full_path]
    
    if not caption_row_df.empty:
        caption_row = caption_row_df.iloc[0]
        # 'all_captions_list_parsed' from V3 init is already a list of strings
        all_text_items_list = caption_row.get('all_captions_list_parsed', default_all_text_items_list) 
        primary_caption = caption_row.get('primary_caption_parsed', default_primary)

        # Convert list of strings to list of dicts {'source': 'BLIP/OCR_Combined', 'text': ...}
        # This is what assemble_llm_context expects for 'all_captions_with_source'
        all_text_dicts_for_img = [{'source': 'BLIP/OCR_Combined', 'text': str(s)} for s in all_text_items_list if isinstance(s, str)]
        
        return primary_caption, all_text_dicts_for_img
        
    return default_primary, default_all_text_items_list


def get_product_details_and_associated_images(product_id, direct_image_matches_lookup):
    # This function remains the same as in main_assistant_new_one.py (provided previously)
    global df_products_global, IMAGE_BASE_PATH
    product_info_dict = None; images_info_list = []
    if not product_id or df_products_global is None: return product_info_dict, images_info_list
    
    product_row_df = df_products_global[df_products_global['product_id'] == product_id]
    if product_row_df.empty: return product_info_dict, images_info_list
    
    product_row = product_row_df.iloc[0]
    product_info_dict = {
        "title": product_row.get('title'), 
        "price": product_row.get('price'), 
        "product_type": product_row.get('product_type')
    }
    added_image_paths = set()

    for img_path_lookup, img_data_lookup in direct_image_matches_lookup.items():
        if len(images_info_list) >= 1: break 
        if img_data_lookup.get('product_id') == product_id:
            primary_cap_val = img_data_lookup.get('primary_caption', "N/A")
            all_caps_with_source_val = img_data_lookup.get('all_captions_with_source', [])
            
            images_info_list.append({
                "image_path": img_path_lookup, 
                "filename": os.path.basename(img_path_lookup),
                "primary_caption": primary_cap_val,
                "all_captions_with_source": all_caps_with_source_val 
            })
            added_image_paths.add(img_path_lookup)

    if not images_info_list and pd.notna(product_row.get('image_paths')):
        general_image_filenames_str = str(product_row.get('image_paths'))
        first_img_filename = general_image_filenames_str.split(',')[0].strip()
        if first_img_filename:
            full_path = os.path.join(IMAGE_BASE_PATH, first_img_filename) 
            if full_path not in added_image_paths and os.path.exists(full_path):
                primary_cap, all_text_dicts_for_img = get_captions_for_image_path(full_path)
                images_info_list.append({
                    "image_path": full_path, 
                    "filename": first_img_filename,
                    "primary_caption": primary_cap, 
                    "all_captions_with_source": all_text_dicts_for_img
                })
                added_image_paths.add(full_path)
    return product_info_dict, images_info_list


def assemble_llm_context(retrieved_texts, vilt_reranked_images, max_total_context_items=4):
    # This function remains the same as in main_assistant_new_one.py (provided previously)
    global df_products_global
    if df_products_global is None: 
        print("Main Error: Product metadata (df_products_global) not loaded for context assembly.")
        return []
    final_context_items = []
    vilt_images_lookup_processed = {}
    for img_data_vilt in vilt_reranked_images:
        if img_data_vilt.get('image_path'):
            img_path_vilt = img_data_vilt['image_path']
            primary_cap_vilt, all_text_dicts_vilt = get_captions_for_image_path(img_path_vilt)
            vilt_images_lookup_processed[img_path_vilt] = {
                **img_data_vilt, 
                'primary_caption': primary_cap_vilt, 
                'all_captions_with_source': all_text_dicts_vilt 
            }
    processed_entity_ids_in_context = set()
    if retrieved_texts:
        for text_chunk in retrieved_texts:
            if len(final_context_items) >= max_total_context_items: break
            pid = text_chunk['metadata'].get('product_id')
            if not pid: continue
            text_chunk_metadata = text_chunk.get('metadata', {})
            current_chunk_text_type = text_chunk_metadata.get('text_type', 'unknown_text_type')
            specific_image_for_this_text_chunk = {}
            is_image_derived_text = current_chunk_text_type.startswith('image_')
            if is_image_derived_text:
                image_source_path_for_text = text_chunk_metadata.get('original_doc_id')
                if image_source_path_for_text:
                    if image_source_path_for_text in vilt_images_lookup_processed:
                        specific_image_for_this_text_chunk = {image_source_path_for_text: vilt_images_lookup_processed[image_source_path_for_text]}
                    else: 
                         primary_cap_img, all_text_dicts_img = get_captions_for_image_path(image_source_path_for_text)
                         specific_image_for_this_text_chunk = {
                            image_source_path_for_text: {
                                'image_path': image_source_path_for_text, 'product_id': pid,
                                'primary_caption': primary_cap_img, 
                                'all_captions_with_source': all_text_dicts_img
                            }
                        }
            lookup_for_assoc_images = specific_image_for_this_text_chunk if specific_image_for_this_text_chunk else vilt_images_lookup_processed
            product_details, associated_images_for_text_item = get_product_details_and_associated_images(
                pid, lookup_for_assoc_images
            )
            context_item_dict = {
                "type": "text_derived_context", "text_content": text_chunk['text_content'],
                "text_score": text_chunk.get('score', 0.0), "text_metadata_details": text_chunk_metadata,
                "associated_product_id": pid, "associated_product_info": product_details,
                "associated_images": associated_images_for_text_item
            }
            final_context_items.append(context_item_dict)
            processed_entity_ids_in_context.add(pid)
    if vilt_reranked_images:
        for img_path, img_data_lookup in vilt_images_lookup_processed.items(): 
            if len(final_context_items) >= max_total_context_items: break
            pid = img_data_lookup.get('product_id')
            if not pid or not img_path: continue
            is_already_associated = any(
                any(assoc_img.get('image_path') == img_path for assoc_img in item_ctx.get('associated_images', []))
                for item_ctx in final_context_items if item_ctx['type'] == 'text_derived_context'
            )
            if not is_already_associated:
                product_details, _ = get_product_details_and_associated_images(pid, {}) 
                final_context_items.append({
                    "type": "image_derived_context", "image_path": img_path,
                    "primary_caption": img_data_lookup.get('primary_caption', "N/A"),
                    "all_captions_with_source": img_data_lookup.get('all_captions_with_source', []),
                    "image_score": img_data_lookup.get('score', 0.0), 
                    "vilt_score": img_data_lookup.get('vilt_score'), 
                    "associated_product_id": pid, "associated_product_info": product_details,
                    "associated_images": [{"image_path": img_path, "filename": os.path.basename(img_path),
                                           "primary_caption": img_data_lookup.get('primary_caption', "N/A"),
                                           "all_captions_with_source": img_data_lookup.get('all_captions_with_source', [])}]
                })
    return final_context_items[:max_total_context_items]


if __name__ == '__main__':
    if not initial_setup(): 
        print("Exiting: Initial setup failed.")
        exit()
        
    conversation_history = [] 

    while True:
        user_query_original = input("\n🛍️ Enter your query (or type 'quit' to exit): ").strip()
        if not user_query_original: continue
        if user_query_original.lower() == 'quit': break

        print(f"\n🔎 Main: Original User Query: '{user_query_original}'")
        history_for_prompt_str = format_conversation_history_for_prompt_v3(conversation_history, MAX_CONVERSATION_HISTORY_TURNS)
        
        query_for_processing = user_query_original

        parsed_query_info = parse_user_query_with_gemini_v3(query_for_processing, history_for_prompt_str)
        
        intent = parsed_query_info.get("intent_type", "GENERAL_PRODUCT_SEARCH")
        key_features_attributes = parsed_query_info.get("key_features_attributes", [])
        comparison_entities = parsed_query_info.get("comparison_entities", [])
        product_entities_from_llm = parsed_query_info.get("product_entities", [])
        brand_entities_from_llm = parsed_query_info.get("brand_entities", [])
        
        retrieval_query_base = parsed_query_info.get("rewritten_query_for_retrieval", query_for_processing)
        if not retrieval_query_base or retrieval_query_base == "N/A":
            retrieval_query_base = query_for_processing
        
        print(f"  LLM Parsed Info: Intent: {intent}, Products: {product_entities_from_llm}, Brands: {brand_entities_from_llm}, Features: {key_features_attributes}")
        print(f"  Using base query for retrieval: '{retrieval_query_base}'")


        final_llm_context_list = []
        max_items_per_comp_product = 2 
        max_items_general_query = 3    

        if intent == "PRODUCT_COMPARISON" and len(comparison_entities) >= 2:
            print(f"Main: Comparison detected. Entities: {comparison_entities}")
            for entity_name_original in comparison_entities:
                entity_name = str(entity_name_original).strip()
                print(f"\n  --- Main: Retrieving for comparison entity: '{entity_name}' ---")
                
                temp_parsed_for_entity_mapping = {"product_entities": [entity_name], "brand_entities": brand_entities_from_llm}
                pids_for_this_entity = map_entities_to_product_ids(temp_parsed_for_entity_mapping, df_products_global)
                
                parsed_constraints_for_query = parse_constraints_from_features(key_features_attributes)
                
                if pids_for_this_entity and df_products_global is not None:
                    temp_df_for_entity_constraints = df_products_global[df_products_global['product_id'].isin(pids_for_this_entity)]
                    if not temp_df_for_entity_constraints.empty:
                        filtered_pids_by_constraint = filter_products_by_constraints(temp_df_for_entity_constraints, parsed_constraints_for_query)['product_id'].tolist()
                        if filtered_pids_by_constraint:
                            pids_for_this_entity = filtered_pids_by_constraint
                
                entity_pinecone_filter = {"product_id": {"$in": pids_for_this_entity[:10]}} if pids_for_this_entity else None
                
                retrieval_query_for_entity_parts = [entity_name] + key_features_attributes + [parsed_constraints_for_query.get('other_features_text','')]
                retrieval_query_for_entity = " ".join(filter(None,retrieval_query_for_entity_parts)).strip().replace("  ", " ")
                if not retrieval_query_for_entity.replace(entity_name,"").strip() : retrieval_query_for_entity = f"{entity_name} details features"
                print(f"    Retrieval query for '{entity_name}': {retrieval_query_for_entity}, Filter: {entity_pinecone_filter}")

                
                texts_entity = retrieve_relevant_chunks_v3(
                    retrieval_query_for_entity, 
                    initial_top_k_dense=5, 
                    initial_top_k_sparse=5,
                    rerank_top_p_hybrid=7,
                    final_top_k=max_items_per_comp_product, 
                    filter_dict=entity_pinecone_filter
                )
                images_clip_entity = retrieve_relevant_images_from_text_clip_v3(retrieval_query_for_entity, top_k=3, filter_dict=entity_pinecone_filter) 
                images_vilt_entity = rerank_images_with_vilt_v3(retrieval_query_for_entity, images_clip_entity, top_k=1) if images_clip_entity else []
                
                context_entity_items = assemble_llm_context(texts_entity, images_vilt_entity, max_total_context_items=max_items_per_comp_product)
                if context_entity_items:
                    final_llm_context_list.append({"type": "comparison_intro", "product_name": entity_name})
                    final_llm_context_list.extend(context_entity_items)
                else:
                    print(f"    No context items assembled for comparison entity: {entity_name}")
        
        else: # General search, feature specific, follow-up etc.
            target_pids_from_entities = map_entities_to_product_ids(parsed_query_info, df_products_global)
            parsed_constraints = parse_constraints_from_features(key_features_attributes) 
            
            candidate_products_df_for_filter = df_products_global
            if target_pids_from_entities:
                candidate_products_df_for_filter = df_products_global[df_products_global['product_id'].isin(target_pids_from_entities)]
            
            if candidate_products_df_for_filter is not None and not candidate_products_df_for_filter.empty:
                 candidate_products_df_for_filter = filter_products_by_constraints(candidate_products_df_for_filter, parsed_constraints)

            final_retrieval_pids = []
            if candidate_products_df_for_filter is not None and not candidate_products_df_for_filter.empty:
                final_retrieval_pids = candidate_products_df_for_filter['product_id'].tolist()
            elif not target_pids_from_entities and (parsed_constraints["price_min"] or parsed_constraints["price_max"] or parsed_constraints["color"] or parsed_constraints["type"]):
                 pass 
            elif target_pids_from_entities: 
                 final_retrieval_pids = target_pids_from_entities
            
            pinecone_filter = {"product_id": {"$in": final_retrieval_pids[:20]}} if final_retrieval_pids else None
            
            semantic_query_parts = [retrieval_query_base]
            for pe_part in product_entities_from_llm:
                if pe_part.lower() not in retrieval_query_base.lower(): semantic_query_parts.append(pe_part)
            for be_part in brand_entities_from_llm:
                if be_part.lower() not in retrieval_query_base.lower(): semantic_query_parts.append(be_part)
            other_f_text = parsed_constraints.get('other_features_text','')
            if other_f_text and other_f_text.lower() not in retrieval_query_base.lower(): 
                semantic_query_parts.append(other_f_text)
            retrieval_query_effective = " ".join(list(dict.fromkeys(filter(None, semantic_query_parts)))).strip()
            if not retrieval_query_effective: retrieval_query_effective = query_for_processing

            print(f"  Effective SEMANTIC retrieval query: '{retrieval_query_effective}'")
            print(f"  Using METADATA filter for Pinecone: {pinecone_filter}")
            
            # CORRECTED CALL with new parameters for hybrid search
            texts_retrieved = retrieve_relevant_chunks_v3(
                retrieval_query_effective, 
                initial_top_k_dense=10,      # Example value
                initial_top_k_sparse=10,     # Example value
                rerank_top_p_hybrid=15,      # Example value (how many to pass to cross-encoder)
                final_top_k=max_items_general_query, # Your existing variable
                filter_dict=pinecone_filter
            )
            
            image_retrieval_query_final = retrieval_query_effective
            if parsed_query_info.get("visual_aspects_queried"):
                image_retrieval_query_final += " " + " ".join(parsed_query_info["visual_aspects_queried"])
            
            images_clip_retrieved = retrieve_relevant_images_from_text_clip_v3(image_retrieval_query_final.strip(), top_k=5, filter_dict=pinecone_filter) 
            images_vilt_reranked = rerank_images_with_vilt_v3(image_retrieval_query_final.strip(), images_clip_retrieved, top_k=2) if images_clip_retrieved else []
            
            final_llm_context_list = assemble_llm_context(texts_retrieved, images_vilt_reranked, max_total_context_items=max_items_general_query)
        
        if final_llm_context_list or (intent == "FOLLOW_UP_QUESTION" and conversation_history):
            print(f"\n📋 --- Main: Assembled Context for LLM (Query: '{query_for_processing}', Total Items: {len(final_llm_context_list)}) ---")
            # (Your existing print logic for context items can go here if needed for debugging)
            
            print("\n🤖 Main: Sending context and history to LLM for answer generation...")
            llm_answer_english = generate_answer_with_gemini_v3(
                query_for_processing, 
                final_llm_context_list, 
                parsed_query_info,
                conversation_history_str=history_for_prompt_str
            )
            
            final_answer_to_user = llm_answer_english
            print("\n💡 --- Shopping Assistant's Answer ---")
            print(final_answer_to_user)
            conversation_history.append((user_query_original, final_answer_to_user))
            if len(conversation_history) > MAX_CONVERSATION_HISTORY_TURNS * 2: 
                conversation_history = conversation_history[-MAX_CONVERSATION_HISTORY_TURNS:]
        else:
            no_info_message = "I'm sorry, I couldn't find enough specific information to answer that right now. Could you try rephrasing or ask about something else?"
            print(f"\n🤷 Main: {no_info_message}")
            conversation_history.append((user_query_original, no_info_message))
            
    print("\nExiting Visual Shopping Assistant. Goodbye!")