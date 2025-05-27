# main_assistant.py
import os
import pandas as pd
from dotenv import load_dotenv 
import re 


# Custom module imports
from data_loader import load_and_clean_data
from retriever import (
    initialize_retriever_resources, 
    retrieve_relevant_chunks, 
    retrieve_relevant_images_from_text
)
from llm_handler import configure_gemini, generate_answer_with_gemini, parse_user_query_with_gemini

# --- Configuration ---
PRODUCTS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/products_final.csv'
REVIEWS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/customer_reviews.csv' 
ALL_DOCS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/all_documents.csv'   
IMAGE_BASE_PATH = '/Users/amankumar/Desktop/Aims/final data/images/' 

df_products_global = None


def initial_setup():
    """Loads API keys and initializes all necessary resources."""
    print("Initializing RAG system resources...")
    load_dotenv() 
    if not os.getenv("PINECONE_API_KEY"): print("CRITICAL ERROR: PINECONE_API_KEY not found."); return False
    if not os.getenv("GOOGLE_API_KEY"): print("CRITICAL ERROR: GOOGLE_API_KEY not found."); return False
    try:
        configure_gemini() 
        initialize_retriever_resources() 
        global df_products_global
        df_prods_temp, _, _ = load_and_clean_data(PRODUCTS_CSV_PATH, REVIEWS_CSV_PATH, ALL_DOCS_CSV_PATH)
        if df_prods_temp is None: raise FileNotFoundError(f"Failed to load product data from {PRODUCTS_CSV_PATH}")
        df_products_global = df_prods_temp
        print(f"Product metadata (df_products_global) loaded. Shape: {df_products_global.shape}")
        if df_products_global.empty: print(f"WARNING: {PRODUCTS_CSV_PATH} loaded empty.")
        if 'price' in df_products_global.columns:
            df_products_global['price_numeric'] = pd.to_numeric(
                df_products_global['price'].astype(str).str.replace('₹', '').str.replace(',', ''), errors='coerce'
            )
            print("Processed 'price' column into 'price_numeric'.")
        else: df_products_global['price_numeric'] = pd.NA
        print("RAG system resources initialized successfully.")
        return True
    except Exception as e: print(f"Critical error during RAG system initialization: {e}"); return False

def map_entities_to_product_ids(parsed_query_info, df_all_products):
    if df_all_products is None or df_all_products.empty: return []
    product_entities = [str(e).lower().strip() for e in parsed_query_info.get("product_entities", []) if str(e).strip()]
    brand_entities = [str(b).lower().strip() for b in parsed_query_info.get("brand_entities", []) if str(b).strip()]
    candidate_pids = set()
    df_all_products['title_lower'] = df_all_products['title'].str.lower()
    if product_entities:
        for entity_model_name in product_entities:
            search_terms = [term for term in re.split(r'\s+|-', entity_model_name) if len(term) > 1] 
            if not search_terms: continue
            condition = pd.Series([True] * len(df_all_products))
            for term in search_terms: condition &= df_all_products['title_lower'].str.contains(term, na=False, regex=False)
            if brand_entities:
                brand_match_condition = pd.Series([False] * len(df_all_products))
                for brand in brand_entities: 
                    if brand in entity_model_name: brand_match_condition |= df_all_products['title_lower'].str.contains(brand, na=False, regex=False)
                if brand_match_condition.any(): condition &= brand_match_condition
            matches = df_all_products[condition]
            for pid in matches['product_id'].tolist(): candidate_pids.add(pid)
    elif brand_entities and not product_entities: 
        for brand in brand_entities:
            matches = df_all_products[df_all_products['title_lower'].str.contains(brand, na=False, regex=False)]
            for pid in matches['product_id'].tolist(): candidate_pids.add(pid)
    df_all_products.drop(columns=['title_lower'], inplace=True, errors='ignore') 
    found_pids = list(candidate_pids)
    if found_pids: print(f"DEBUG: Mapped PIDs: {found_pids} for P_Entities={product_entities}, B_Entities={brand_entities}")
    else: print(f"DEBUG: No PIDs mapped for P_Entities={product_entities}, B_Entities={brand_entities}")
    return found_pids

def parse_constraints_from_features(key_features_attributes):
    constraints = {"price_min": None, "price_max": None, "color": None, "type": None}
    other_features = []
    color_keywords = ["blue", "black", "red", "white", "grey", "gray", "green", "silver", "beige", "aqua", "crimson"]
    type_keywords = ["over-ear", "on-ear", "in-ear", "neckband", "earbuds"]
    for feature in key_features_attributes:
        feature_lower = str(feature).lower()
        price_max_match = re.search(r"(?:price under|less than|below)\s*₹?\s*(\d[\d,]*\.?\d*)", feature_lower)
        price_min_match = re.search(r"(?:price over|more than|above)\s*₹?\s*(\d[\d,]*\.?\d*)", feature_lower)
        if price_max_match: constraints["price_max"] = float(price_max_match.group(1).replace(',', ''))
        elif price_min_match: constraints["price_min"] = float(price_min_match.group(1).replace(',', ''))
        elif feature_lower.startswith("color:"): constraints["color"] = feature_lower.split(":", 1)[1].strip()
        elif feature_lower.startswith("type:"): constraints["type"] = feature_lower.split(":", 1)[1].strip()
        else:
            found_color = next((color for color in color_keywords if color in feature_lower), None)
            found_type = next((ptype for ptype in type_keywords if ptype in feature_lower), None)
            if found_color and not constraints["color"]: constraints["color"] = found_color
            elif found_type and not constraints["type"]: constraints["type"] = found_type
            else: other_features.append(feature)
    constraints["other_features_text"] = " ".join(other_features).strip()
    print(f"DEBUG: Parsed constraints: {constraints}")
    return constraints

def filter_products_by_constraints(df_prods, constraints):
    if df_prods is None or df_prods.empty: return pd.DataFrame()
    filtered_df = df_prods.copy()
    if 'price_numeric' not in filtered_df.columns: return filtered_df 
    if constraints.get("price_min") is not None: filtered_df = filtered_df[filtered_df['price_numeric'] >= constraints["price_min"]]
    if constraints.get("price_max") is not None: filtered_df = filtered_df[filtered_df['price_numeric'] <= constraints["price_max"]]
    if constraints.get("color"): filtered_df = filtered_df[filtered_df['title'].str.lower().str.contains(constraints["color"], na=False, regex=False)]
    if constraints.get("type"):
        type_condition = pd.Series([False] * len(filtered_df))
        if 'product_type' in filtered_df.columns: type_condition |= filtered_df['product_type'].str.lower().str.contains(constraints["type"], na=False, regex=False)
        type_condition |= filtered_df['title'].str.lower().str.contains(constraints["type"], na=False, regex=False)
        filtered_df = filtered_df[type_condition]
    print(f"DEBUG: Products after metadata constraint filtering: {len(filtered_df)} rows")
    return filtered_df

def get_product_details_and_images(product_id, direct_image_matches_lookup):
    global df_products_global, IMAGE_BASE_PATH
    product_info_dict = None; images_info_list = []
    if product_id and df_products_global is not None:
        product_row_df = df_products_global[df_products_global['product_id'] == product_id]
        if not product_row_df.empty:
            product_row = product_row_df.iloc[0]
            product_info_dict = {"title": product_row.get('title'), "price": product_row.get('price'), "product_type": product_row.get('product_type')}
            if pd.notna(product_row.get('image_paths')):
                images_processed_for_this_product = 0
                for rel_path in str(product_row.get('image_paths')).split(','): 
                    if images_processed_for_this_product >= 2: break
                    img_filename = rel_path.strip()
                    if not img_filename: continue
                    full_path = os.path.join(IMAGE_BASE_PATH, img_filename)
                    caption = direct_image_matches_lookup.get(full_path, {}).get('caption', "Caption not available for this associated image.")
                    images_info_list.append({"image_path": full_path, "caption": caption})
                    images_processed_for_this_product +=1
    return product_info_dict, images_info_list

def assemble_llm_context(retrieved_texts, retrieved_images, max_total_context_items=3):
    global df_products_global
    if df_products_global is None: print("Error: Product metadata not loaded."); return []
    final_context_items = []
    direct_image_matches_lookup = {img['image_path']: img for img in retrieved_images if img.get('image_path')}
    processed_product_ids_in_final_context = set()
    num_text_items_to_process = max_total_context_items 
    num_image_items_to_process = max_total_context_items 
    if retrieved_texts:
        print("\nProcessing text-based results for LLM context...")
        for text_chunk in retrieved_texts[:num_text_items_to_process]: 
            if len(final_context_items) >= max_total_context_items: break
            pid = text_chunk['metadata'].get('product_id')
            if not pid: continue
            info, images = get_product_details_and_images(pid, direct_image_matches_lookup)
            final_context_items.append({"type": "text_derived_context", "text_content": text_chunk['text_content'], 
                                      "text_score": text_chunk['score'], "text_metadata_details": text_chunk['metadata'], 
                                      "associated_product_id": pid, "associated_product_info": info, 
                                      "associated_images": images})
            processed_product_ids_in_final_context.add(pid)
    if retrieved_images:
        print("\nProcessing image-based results for LLM context...")
        added_image_specific_items = 0
        for img_data in sorted(retrieved_images, key=lambda x: x.get('score', 0.0), reverse=True):
            if len(final_context_items) >= max_total_context_items: break
            if added_image_specific_items >= num_image_items_to_process : break
            pid = img_data['product_id']; img_path = img_data['image_path']
            product_already_has_text_context = False
            if pid in processed_product_ids_in_final_context:
                product_already_has_text_context = True
                for item in final_context_items:
                    if item.get("associated_product_id") == pid and item.get("type") == "text_derived_context":
                        image_already_in_item_visuals = False
                        for assoc_img in item.get("associated_images", []):
                            if assoc_img["image_path"] == img_path:
                                assoc_img["caption"] = img_data['caption']; image_already_in_item_visuals = True; break
                        if not image_already_in_item_visuals and len(item.get("associated_images",[])) < 2 : 
                             item["associated_images"].append({"image_path": img_path, "caption": img_data['caption']})
                        break 
            if not product_already_has_text_context: 
                info, _ = get_product_details_and_images(pid, {})
                final_context_items.append({"type": "image_derived_context", "image_path": img_path, 
                                          "image_caption": img_data['caption'], "image_score": img_data['score'], 
                                          "associated_product_id": pid, "associated_product_info": info, 
                                          "associated_images": [{"image_path": img_path, "caption": img_data['caption']}]})
                if pid: processed_product_ids_in_final_context.add(pid)
                added_image_specific_items +=1
    def sort_key(item):
        priority = 0; score = 0.0
        if item['type'] == 'text_derived_context': priority = 0; score = item.get('text_score', 0.0)
        elif item['type'] == 'image_derived_context': priority = 1; score = item.get('image_score', 0.0)
        else: priority = -1 
        return (priority, -score) 
    final_context_items.sort(key=sort_key)
    return final_context_items[:max_total_context_items]

# --- END: Pasted functions ---


if __name__ == '__main__':
    if not initial_setup(): 
        exit()

    # --- Interactive Loop ---
    while True:
        user_query_original = input("\n🛍️ Enter your query (or type 'quit' to exit): ")
        if user_query_original.lower() == 'quit':
            break
        if not user_query_original.strip():
            continue

        print(f"\n🔎 Original User Query: '{user_query_original}'")
        parsed_query_info = parse_user_query_with_gemini(user_query_original)
        
        intent = parsed_query_info.get("intent_type", "general_search")
        key_features_attributes = parsed_query_info.get("key_features_attributes", [])
        comparison_entities = parsed_query_info.get("comparison_entities", [])

        final_llm_context_list = []
        # llm_query_to_use_for_answer = user_query_original # LLM always answers original query

        if intent == "product_comparison" and len(comparison_entities) == 2:
            print(f"Comparison detected. Comparing: '{comparison_entities[0]}' WITH '{comparison_entities[1]}'")
            max_items_per_comp_product = 2 
            
            for i, entity_raw_name in enumerate(comparison_entities):
                entity_name = str(entity_raw_name).strip()
                print(f"\n--- Retrieving info for comparison entity: '{entity_name}' ---")
                
                parsed_constraints_for_entity = parse_constraints_from_features(key_features_attributes)
                pids_for_this_entity = map_entities_to_product_ids({"product_entities": [entity_name], "brand_entities": [entity_name.split(' ')[0]]}, df_products_global)
                
                if pids_for_this_entity and not df_products_global[df_products_global['product_id'].isin(pids_for_this_entity)].empty:
                    temp_df_for_entity = df_products_global[df_products_global['product_id'].isin(pids_for_this_entity)]
                    # Apply general constraints from query to this entity's product list
                    filtered_pids_by_constraint = filter_products_by_constraints(temp_df_for_entity, parsed_constraints_for_entity)['product_id'].tolist()
                    if filtered_pids_by_constraint : pids_for_this_entity = filtered_pids_by_constraint
                
                entity_filter = {"product_id": {"$in": pids_for_this_entity[:20]}} if pids_for_this_entity else None
                # Construct a retrieval query focused on the entity and any general features from original query
                retrieval_query_for_entity = f"{entity_name} {parsed_constraints_for_entity.get('other_features_text','')} {' '.join(key_features_attributes)}".strip()
                if not retrieval_query_for_entity.replace(entity_name, "").strip(): # If only entity name, add generic terms
                    retrieval_query_for_entity = f"{entity_name} features specifications reviews"
                
                print(f"Effective retrieval query for '{entity_name}': '{retrieval_query_for_entity}' with filter: {entity_filter}")
                texts_entity = retrieve_relevant_chunks(retrieval_query_for_entity, initial_top_k=5, final_top_k=max_items_per_comp_product, filter_dict=entity_filter)
                images_entity = retrieve_relevant_images_from_text(retrieval_query_for_entity, top_k=1, filter_dict=entity_filter) 
                context_entity_items = assemble_llm_context(texts_entity, images_entity, max_total_context_items=max_items_per_comp_product)
                
                if context_entity_items:
                    final_llm_context_list.append({"type": "comparison_intro", "product_name": entity_name})
                    final_llm_context_list.extend(context_entity_items)
        
        else: # Standard or Multi-Constraint Search (non-comparison)
            target_pids_from_entities = map_entities_to_product_ids(parsed_query_info, df_products_global)
            parsed_constraints = parse_constraints_from_features(key_features_attributes)
            
            candidate_products_df = df_products_global
            if any(v is not None for k,v in parsed_constraints.items() if k != 'other_features_text') : 
                candidate_products_df = filter_products_by_constraints(df_products_global, parsed_constraints)
            
            final_retrieval_pids = []
            if not candidate_products_df.empty: # If metadata filter yielded results
                if target_pids_from_entities: 
                    final_retrieval_pids = list(set(target_pids_from_entities) & set(candidate_products_df['product_id'].tolist()))
                    if not final_retrieval_pids: 
                        print("DEBUG: No intersection of PIDs. Using PIDs from metadata filter if available, else from entities.")
                        final_retrieval_pids = candidate_products_df['product_id'].tolist() if not candidate_products_df.empty else target_pids_from_entities
                else: 
                    final_retrieval_pids = candidate_products_df['product_id'].tolist()
            elif target_pids_from_entities: # No results from metadata filter, but entities found PIDs
                 final_retrieval_pids = target_pids_from_entities
            
            print(f"DEBUG: Final target PIDs for retrieval: {final_retrieval_pids[:5]} (up to 5 shown)")
            pinecone_filter = {"product_id": {"$in": final_retrieval_pids[:20]}} if final_retrieval_pids else None
            
            # Construct effective semantic query
            semantic_query_parts = []
            if parsed_constraints.get('other_features_text'): # Unstructured features first
                semantic_query_parts.append(parsed_constraints['other_features_text'])
            # Add entities/brands if no PID filter is active OR to reinforce semantic search for those entities
            if not pinecone_filter or (pinecone_filter and (parsed_query_info.get("product_entities") or parsed_query_info.get("brand_entities"))):
                 semantic_query_parts.extend(parsed_query_info.get("product_entities", []))
                 semantic_query_parts.extend(parsed_query_info.get("brand_entities", []))
            
            if not semantic_query_parts: # Fallback if no specific parts parsed
                semantic_query_parts.append(user_query_original)
            # Ensure original query is part of semantic search if not fully covered by parsed parts
            elif user_query_original.lower() not in " ".join(semantic_query_parts).lower():
                 semantic_query_parts.append(user_query_original)


            retrieval_query_effective = " ".join(list(dict.fromkeys(filter(None, semantic_query_parts)))).strip() # Unique, non-empty
            if not retrieval_query_effective: retrieval_query_effective = user_query_original

            print(f"Effective SEMANTIC retrieval query: '{retrieval_query_effective}'")
            print(f"Using METADATA filter for Pinecone: {pinecone_filter}")
            
            texts = retrieve_relevant_chunks(retrieval_query_effective, initial_top_k=10, final_top_k=3, filter_dict=pinecone_filter)
            images = retrieve_relevant_images_from_text(retrieval_query_effective, top_k=2, filter_dict=pinecone_filter) 
            final_llm_context_list = assemble_llm_context(texts, images, max_total_context_items=3)
        
        if final_llm_context_list:
            print(f"\n📋 --- Assembled Context for LLM (Query: '{user_query_original}', Total Items for LLM: {len(final_llm_context_list)}) ---")
            current_product_header = "GENERAL CONTEXT"
            processed_headers_count = 0
            for i_ctx, item_ctx in enumerate(final_llm_context_list):
                if item_ctx['type'] == "comparison_intro":
                    current_product_header = item_ctx['product_name']
                    print(f"\n--- CONTEXT FOR PRODUCT: {current_product_header.upper()} ---")
                    processed_headers_count += 1; continue
                print_item_index = i_ctx + 1 - processed_headers_count
                print(f"\nContext Snippet #{print_item_index} (Source: {item_ctx['type']}):")
                if item_ctx.get('associated_product_info'): print(f"  Product: {item_ctx['associated_product_info'].get('title', 'N/A')} (ID: {item_ctx.get('associated_product_id', 'N/A')})")
                elif item_ctx.get('associated_product_id'): print(f"  Product ID: {item_ctx.get('associated_product_id', 'N/A')}")
                if item_ctx['type'] == 'text_derived_context':
                    print(f"  Text (Rerank Score: {item_ctx.get('text_score', 0.0):.4f}): \"{item_ctx.get('text_content', '')[:100]}...\"")
                    if item_ctx.get('text_metadata_details'): print(f"  Text Aspect: {item_ctx['text_metadata_details'].get('aspect', 'N/A')}")
                elif item_ctx['type'] == 'image_derived_context':
                    print(f"  Image (CLIP Score: {item_ctx.get('image_score', 0.0):.4f}): {item_ctx.get('image_path', 'N/A')}")
                    print(f"  Caption: {item_ctx.get('image_caption', 'N/A')}") 
                if item_ctx.get('associated_images'): 
                    print("  Visuals:")
                    for img_info in item_ctx['associated_images'][:1]: 
                        print(f"    - Image: {img_info.get('image_path', 'N/A')}")
                        print(f"      Caption: {img_info.get('caption', 'N/A')}")
                print("-" * 20)

            print("\n🤖 Sending context to LLM for answer generation...")
            llm_answer = generate_answer_with_gemini(user_query_original, final_llm_context_list, parsed_query_info)
            print("\n💡 --- Shopping Assistant's Answer ---")
            print(llm_answer)
        else:
            print("\n🤷 No sufficient context assembled from text or image search to send to the LLM.")
            
    print("\nExiting Visual Shopping Assistant. Goodbye!")