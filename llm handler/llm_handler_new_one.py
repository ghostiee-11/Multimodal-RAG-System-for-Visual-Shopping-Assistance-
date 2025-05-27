# llm_handler_new.py
import os
import pandas as pd 
import google.generativeai as genai
from PIL import Image 
from dotenv import load_dotenv
import json 
import re

# --- Configuration ---
GEMINI_ANSWER_MODEL_NAME = "gemini-1.5-flash-latest" 
QUERY_PARSER_MODEL_NAME = "gemini-1.5-flash-latest" 

def configure_gemini():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("LLM Handler Error: GOOGLE_API_KEY not found in .env file.")
    try:
        genai.configure(api_key=api_key)
        print("LLM Handler: Gemini API configured successfully.")
    except Exception as e:
        print(f"LLM Handler Error: Configuring Gemini API: {e}"); raise

def format_conversation_history_for_prompt(history_tuples, max_turns=3):
    if not history_tuples:
        return ""
    formatted_history = "\n\n--- Previous Conversation Turn(s) ---\n"
    recent_history = history_tuples[-max_turns:]
    for i, (user_turn, assistant_turn) in enumerate(recent_history):
        turn_number = -len(recent_history) + i 
        formatted_history += f"User (Turn {turn_number}): {user_turn}\n"
        formatted_history += f"Assistant (Turn {turn_number}): {assistant_turn}\n---\n"
    return formatted_history.strip()


def parse_user_query_with_gemini(user_query, conversation_history_str=""):
    print(f"\n🤖 LLM Handler: Parsing user query (history considered): '{user_query}'")
    try:
        parser_model = genai.GenerativeModel(QUERY_PARSER_MODEL_NAME)
        
        parsing_prompt = f"""Your task is to deeply analyze the user's current shopping query for headphones and extract structured information.
Consider the preceding conversation turns, if provided, to understand context and resolve ambiguities.
{conversation_history_str}

Current User Query: "{user_query}"

Strictly return your answer ONLY as a JSON object with the following keys. If a category has no information, use an empty list [] or "N/A" for strings.
The extracted information should primarily focus on the *Current User Query*.

1.  "original_query": (String) The user's current query.
2.  "intent_type": (String) Classify the CURRENT query's primary intent. Choose ONE from: 
    "PRODUCT_FEATURE_SPECIFIC", "PRODUCT_COMPARISON", "GENERAL_PRODUCT_SEARCH", 
    "OPINION_REQUEST", "VISUAL_REQUEST", "AVAILABILITY_CHECK", "FOLLOW_UP_QUESTION", "OTHER_UNRELATED".
3.  "product_entities": (List of Strings) Specific product model names or series from CURRENT query, or clearly referenced from history.
4.  "brand_entities": (List of Strings) Distinct brand names from CURRENT query or history.
5.  "key_features_attributes": (List of Strings) Explicit features/attributes/constraints from CURRENT query, or implied by follow-up. Normalize where possible (e.g., "color:blue", "price_max:2000", "type:over-ear", "feature:noise cancelling").
6.  "comparison_entities": (List of Strings) If intent_type is "PRODUCT_COMPARISON", list distinct product/brand names being compared (at least two). Else, empty list.
7.  "visual_aspects_queried": (List of Strings) Visual aspects user is interested in (e.g., ["look", "design", "color of earcup"]).
8.  "rewritten_query_for_retrieval": (String) If current query is vague, provide a more explicit version for semantic search, incorporating history context. If no rewrite needed, repeat current query or use "N/A".
9.  "decomposition_hints": (List of Dictionaries) For complex comparisons in CURRENT query. Each dict: {{"entity": "Product Name", "features_to_check": ["feature1", "feature2"]}}. Else, empty list.

Example (with history):
--- Previous Conversation Turn(s) ---
User (Turn -1): Tell me about Sony WH-1000XM5.
Assistant (Turn -1): The Sony WH-1000XM5 has great noise cancellation and up to 40 hours battery.
---
Current User Query: "What about its price and does it come in silver?"
Output JSON: {{
    "original_query": "What about its price and does it come in silver?",
    "intent_type": "FOLLOW_UP_QUESTION", 
    "product_entities": ["Sony WH-1000XM5"], 
    "brand_entities": ["Sony"], 
    "key_features_attributes": ["price", "color:silver"],
    "comparison_entities": [],
    "visual_aspects_queried": ["color:silver"],
    "rewritten_query_for_retrieval": "price and silver color availability of Sony WH-1000XM5",
    "decomposition_hints": []
}}

Now, parse the Current User Query. Output ONLY the JSON object.
"""
        response = parser_model.generate_content(parsing_prompt)
        cleaned_response_text = response.text.strip()
        
        match = re.search(r"```json\s*([\s\S]+?)\s*```", cleaned_response_text)
        if match:
            json_str = match.group(1)
        elif cleaned_response_text.startswith("{") and cleaned_response_text.endswith("}"):
            json_str = cleaned_response_text
        else:
            json_start_index = cleaned_response_text.find('{')
            json_end_index = cleaned_response_text.rfind('}')
            if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
                json_str = cleaned_response_text[json_start_index : json_end_index+1]
            else:
                print(f"LLM Handler Warning: Could not robustly extract JSON from parser response: {cleaned_response_text}")
                # Fallback: Try to find any JSON-like structure, less strict
                try:
                    # This is a more desperate attempt, might grab partial or incorrect JSON
                    potential_json_match = re.search(r"(\{[\s\S]*\})", cleaned_response_text)
                    if potential_json_match:
                        json_str = potential_json_match.group(1)
                        # Attempt to parse to see if it's valid enough
                        json.loads(json_str) 
                    else:
                        raise json.JSONDecodeError("No JSON object found in response", cleaned_response_text, 0)
                except json.JSONDecodeError: # If even the desperate attempt fails
                     raise json.JSONDecodeError("No JSON object found in response after multiple attempts", cleaned_response_text, 0)

        parsed_info = json.loads(json_str)
        defaults = {
            "original_query": user_query, "intent_type": "GENERAL_PRODUCT_SEARCH", 
            "product_entities": [], "brand_entities": [], "key_features_attributes": [],
            "comparison_entities": [], "visual_aspects_queried": [],
            "rewritten_query_for_retrieval": user_query, "decomposition_hints": []
        }
        for key, default_val in defaults.items():
            parsed_info.setdefault(key, default_val)

        print(f"  LLM Handler: Parsed query info: {json.dumps(parsed_info, indent=2)}")
        return parsed_info

    except Exception as e:
        print(f"LLM Handler Error: During query parsing with Gemini: {e}. Response text: '{cleaned_response_text if 'cleaned_response_text' in locals() else 'N/A'}'")
        # Return a default structure on error to prevent crashes downstream
        return {
            "original_query": user_query, "intent_type": "GENERAL_PRODUCT_SEARCH", 
            "product_entities": [], "brand_entities": [],
            "key_features_attributes": [user_query], # Use original query as a feature if parsing fails
            "comparison_entities": [], "visual_aspects_queried": [], 
            "rewritten_query_for_retrieval": user_query, "decomposition_hints": []
        }


def generate_answer_with_gemini(user_query, multimodal_contexts_list, parsed_query_info=None, conversation_history_str=""):
    if not multimodal_contexts_list and not user_query:
        if parsed_query_info and parsed_query_info.get("intent_type") == "FOLLOW_UP_QUESTION" and conversation_history_str:
            pass 
        else:
            return "I need more information or a query to help you."
            
    if not multimodal_contexts_list and not conversation_history_str: # No new context and no history
        return "I couldn't find specific product information related to your current query, and there's no prior conversation to draw from."

    try:
        model = genai.GenerativeModel(GEMINI_ANSWER_MODEL_NAME)
    except Exception as e:
        return f"LLM Handler Error: Initializing Gemini model '{GEMINI_ANSWER_MODEL_NAME}': {e}"

    prompt_parts = []
    image_pil_objects_for_llm = {} 
    
    is_comparison_task = parsed_query_info and parsed_query_info.get("intent_type") == "PRODUCT_COMPARISON" and \
                         len(parsed_query_info.get("comparison_entities", [])) >= 2
    product_names_for_comparison = parsed_query_info.get("comparison_entities", []) if is_comparison_task else []

    if conversation_history_str:
        prompt_parts.append(conversation_history_str)
        prompt_parts.append("\n--- Current Interaction & Provided Context for Grounding ---")

    base_instructions = f"""You are a helpful and precise **visual shopping assistant**.
Your role is to directly answer the user's current question about products, drawing knowledge *only* from the "Provided Context for Grounding" below and relevant "Previous Conversation Turns" (if any).

**IMPORTANT: Your answer should be a direct response to the user. Do NOT say "Based on the provided information/context..." or "The context shows...".**
Instead, synthesize the information from the context and present it as your own knowledge.

When information comes from text extracted from an image (like OCR or an AI-generated caption for an image), you can subtly indicate this if it adds clarity, for example:
- "The product image for [Product Name] shows text indicating 'Feature X'."
- "An AI description of the [Product Name] image mentions it is '[color and type]'."
- "Yes, the [Product Name] has [Feature X], which is also visible on its packaging/image."

If visual information from an image itself or its direct textual descriptions (captions/OCR) confirms or elaborates on other text, integrate this smoothly.

⚠️ **Key Directives for Your Response:**
1.  **Grounding:** All facts in your answer MUST be traceable to the "Provided Context for Grounding" or "Previous Conversation Turns." Do not use external knowledge.
2.  **Direct Answer:** Formulate a direct, helpful, and natural sounding answer to the user's current question.
3.  **Synthesize, Don't Recite Context:** Do not just list context items. Understand them and provide a coherent answer.
4.  **Cannot Answer:** If the necessary information is NOT in the provided context or relevant history, state clearly: "I'm sorry, I don't have that specific information." or "I couldn't find details about that." Do not guess.
5.  **Conciseness:** Be informative but avoid unnecessary verbosity.
"""

    if is_comparison_task:
        prompt_parts.append(f"""{base_instructions}
Specifically, your task is to **compare products: { ' and '.join([f"'{name.strip().title()}'" for name in product_names_for_comparison]) }** based on the user's current question, using ONLY the provided context.
Highlight key differences and similarities relevant to the user's query.
---
**User’s Current Question:** "{user_query}"
---
**Provided Context for Grounding (organized by product if applicable):**
""")
    else: 
        prompt_parts.append(f"""{base_instructions}
---
**User’s Current Question:** "{user_query}"
---
**Provided Context for Grounding:**
""")
    
    context_text_blob_parts = [] 
    current_product_header = None
    max_llm_context_items = 6 

    for i, context_item in enumerate(multimodal_contexts_list[:max_llm_context_items]):
        item_type = context_item.get("type")
        temp_text_parts = []

        if item_type == "comparison_intro": 
            product_name_intro = context_item.get("product_name", "Unnamed Product").strip().title()
            if current_product_header and current_product_header.lower() != product_name_intro.lower():
                 temp_text_parts.append("\n--- End of Details for Previous Product ---") 
            current_product_header = product_name_intro
            temp_text_parts.append(f"\n\n**CONTEXT BLOCK FOR PRODUCT: {current_product_header.upper()}**")
        else: 
            temp_text_parts.append(f"\nContext Snippet (Source Type: {item_type}):")
            prod_info = context_item.get('associated_product_info')
            prod_id = context_item.get('associated_product_id')
            
            if prod_info and prod_info.get('title'):
                temp_text_parts.append(f"  Product: {prod_info['title']} (ID: {prod_id})")
            elif prod_id:
                temp_text_parts.append(f"  Product ID: {prod_id}")

            if item_type == 'text_derived_context':
                text_meta = context_item.get('text_metadata_details', {})
                text_content_str = str(context_item.get('text_content', ''))
                
                image_filename_source_display = "N/A"
                if text_meta.get('text_type', '').startswith('image_'):
                    img_src_path_meta = text_meta.get('original_doc_id') 
                    if img_src_path_meta and isinstance(img_src_path_meta, str):
                        try: image_filename_source_display = os.path.basename(img_src_path_meta)
                        except: image_filename_source_display = str(img_src_path_meta) # If not a path
                    elif text_meta.get('image_filename_source'):
                         image_filename_source_display = text_meta.get('image_filename_source')
                    elif text_meta.get('image_filename'): 
                         image_filename_source_display = text_meta.get('image_filename')

                current_chunk_text_type_meta = text_meta.get('text_type', 'unknown_text_type')
                if current_chunk_text_type_meta == 'image_ocr_text':
                    temp_text_parts.append(f"  Text Source: OCR from image '{image_filename_source_display}'")
                    temp_text_parts.append(f"  Text Content: \"{text_content_str}\"")
                elif current_chunk_text_type_meta == 'image_blip_text':
                    temp_text_parts.append(f"  Text Source: AI-caption (BLIP) for image '{image_filename_source_display}'")
                    temp_text_parts.append(f"  Text Content: \"{text_content_str}\"")
                elif current_chunk_text_type_meta == 'image_derived_filtered_text': 
                    temp_text_parts.append(f"  Text Source: Filtered text from image '{image_filename_source_display}'")
                    temp_text_parts.append(f"  Text Content: \"{text_content_str}\"")
                else: 
                    temp_text_parts.append(f"  Text Source Detail: Type: {text_meta.get('text_type', 'N/A')}, Aspect: {text_meta.get('aspect', 'N/A')} (Sentiment: {text_meta.get('sentiment', 'N/A')})")
                    temp_text_parts.append(f"  Text Content: \"{text_content_str}\"")
            
            elif item_type == 'image_derived_context': 
                img_path_display_direct = context_item.get('image_path', 'N/A')
                if img_path_display_direct != 'N/A': img_path_display_direct = os.path.basename(img_path_display_direct)
                temp_text_parts.append(f"  Primary Image Item: '{img_path_display_direct}'")
                
                all_text_dicts_for_primary_img = context_item.get('all_captions_with_source', [])
                if all_text_dicts_for_primary_img:
                    temp_text_parts.append(f"    Texts describing this image '{img_path_display_direct}':")
                    for text_dict in all_text_dicts_for_primary_img[:3]: 
                        source_label = str(text_dict.get('source', 'Unknown')).upper().replace("_TEXT", "").replace("IMAGE_","")
                        text_val = str(text_dict.get('text', 'N/A'))
                        temp_text_parts.append(f"      - ({source_label}): \"{text_val[:100]}...\"")
                elif context_item.get('primary_caption') and context_item.get('primary_caption') not in ["N/A", "Caption unavailable."]:
                     temp_text_parts.append(f"    Primary AI Caption: \"{context_item.get('primary_caption')}\"")

            if prod_info and prod_info.get('price'): temp_text_parts.append(f"  Price: {prod_info.get('price')}")

            associated_imgs_for_llm_prompt = context_item.get('associated_images', [])
            if associated_imgs_for_llm_prompt:
                temp_text_parts.append("  Visually Linked Image(s) (content may be attached to prompt if different from primary image item):")
                for img_detail in associated_imgs_for_llm_prompt[:1]: 
                    img_path_assoc_llm = img_detail.get('image_path')
                    img_filename_assoc_llm = os.path.basename(img_path_assoc_llm) if img_path_assoc_llm and isinstance(img_path_assoc_llm, str) else "N/A"
                    cap_primary_assoc_llm = str(img_detail.get('primary_caption', 'N/A')) 
                    temp_text_parts.append(f"    - File: '{img_filename_assoc_llm}', Primary AI Caption: \"{cap_primary_assoc_llm}\"")
                    
                    if img_path_assoc_llm and isinstance(img_path_assoc_llm, str) and os.path.exists(img_path_assoc_llm) and img_path_assoc_llm not in image_pil_objects_for_llm:
                        try: 
                            image_pil_objects_for_llm[img_path_assoc_llm] = Image.open(img_path_assoc_llm).convert("RGB")
                        except Exception as e_img_load: 
                            print(f"LLM Warn: Could not load image {img_path_assoc_llm} for LLM prompt: {e_img_load}")
        
        context_text_blob_parts.append("\n".join(temp_text_parts))
    
    prompt_parts.append("\n".join(context_text_blob_parts))

    if image_pil_objects_for_llm:
        prompt_parts.append("\n\n--- Attached Image Data (filenames referenced in context) ---")
        for img_path_key_llm, pil_image_obj_llm in image_pil_objects_for_llm.items():
            prompt_parts.append(f"\nImage Content for '{os.path.basename(img_path_key_llm)}':")
            prompt_parts.append(pil_image_obj_llm)

    prompt_parts.append(f"""
---
Now, directly answer the user's current question: "{user_query}"
Remember all directives: provide a direct answer, synthesize information, do not recite context, and only use the provided grounding information.
If the information isn't present, clearly state that.
Your Answer:
""")
    
    try:
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=1024, 
            temperature=0.15, 
            top_p=0.9,      
            top_k=30        
        )
        safety_settings = [ 
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = model.generate_content(
            prompt_parts, 
            generation_config=generation_config, 
            safety_settings=safety_settings,
            request_options={'timeout': 120}
        )
        
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            print(f"LLM Handler Warning: Prompt was blocked. Reason: {response.prompt_feedback.block_reason}")
            if response.prompt_feedback.safety_ratings:
                for rating in response.prompt_feedback.safety_ratings: print(f"  Safety Rating: Category '{rating.category}', Probability '{rating.probability}'")
            return "I'm sorry, I can't provide an answer to that query due to content restrictions."

        if not response.candidates or not response.candidates[0].content.parts:
             print("LLM Handler Warning: No content parts in Gemini response (candidate might have been filtered).")
             if response.candidates and response.candidates[0].finish_reason:
                 print(f"  Candidate Finish Reason: {response.candidates[0].finish_reason.name}")
                 if response.candidates[0].safety_ratings:
                     for rating in response.candidates[0].safety_ratings: print(f"  Candidate Safety Rating: Category '{rating.category}', Probability '{rating.probability}'")
             return "I received an empty or filtered response. Please try rephrasing your query."
        
        final_answer_text = response.text.strip()
        return final_answer_text

    except Exception as e:
        print(f"LLM Handler Error: During Gemini API call: {type(e).__name__} - {e}")
        return "Sorry, I encountered an error while generating a response for your query. Please try again later."

if __name__ == '__main__':
    try:
        configure_gemini()
    except Exception as e:
        print(f"LLM Handler Test: Failed to configure Gemini: {e}"); exit()

    print("\n--- LLM Handler Test: Direct Answer Style ---")
    test_q_direct = "Do the Sony WH-1000XM4 headphones have good battery life?"
    parsed_direct = parse_user_query_with_gemini(test_q_direct)
    
    dummy_image_dir_llm = "dummy_test_images_llm_handler_main" # Unique name for this test
    os.makedirs(dummy_image_dir_llm, exist_ok=True)
    dummy_img_path_llm_test = os.path.join(dummy_image_dir_llm, "B0863FR3S9_box_image.jpg")
    if not os.path.exists(dummy_img_path_llm_test): 
        try: Image.new('RGB', (60, 30), color = 'green').save(dummy_img_path_llm_test)
        except Exception as e_img_save: print(f"Could not save dummy image for LLM test: {e_img_save}")


    dummy_context_direct = [
        {
            "type": "text_derived_context", 
            "text_content": "Sony WH-1000XM4: A single charge provides up to 30 hrs of playtime for reliable all day listening.",
            "associated_product_id": "B0863FR3S9", 
            "associated_product_info": {"title": "Sony WH-1000XM4 Wireless Headphones", "price": "₹22990"},
            "text_metadata_details": {
                "text_type": "specification", "aspect": "Battery life", "sentiment": "N/A",
                "original_doc_id": "spec_doc_id_xm4_battery"
            },
            "associated_images": [] # No image directly from this spec text
        },
        {
            "type": "text_derived_context",
            "text_content": "Up to 30 HR", # Simulating OCR text more realistically
            "associated_product_id": "B0863FR3S9",
            "associated_product_info": {"title": "Sony WH-1000XM4 Wireless Headphones"},
            "text_metadata_details": {
                "text_type": "image_ocr_text", "aspect": "image_content_textual",
                "original_doc_id": dummy_img_path_llm_test, # Actual path to the dummy image
                "image_filename_source": "B0863FR3S9_box_image.jpg"
            },
            "associated_images": [{ # The image from which OCR was extracted
                "image_path": dummy_img_path_llm_test, 
                "filename": "B0863FR3S9_box_image.jpg",
                "primary_caption": "Sony WH-1000XM4 box showing battery life detail",
                "all_captions_with_source": [
                    {'source': 'BLIP', 'text': 'a box for sony headphones'},
                    {'source': 'OCR', 'text': 'Up to 30 HR'}
                ]
            }]
        }
    ]
    answer_direct = generate_answer_with_gemini(test_q_direct, dummy_context_direct, parsed_direct)
    print(f"User Query: {test_q_direct}")
    print(f"LLM Direct Answer Style:\n{answer_direct}")
    print("-" * 20)