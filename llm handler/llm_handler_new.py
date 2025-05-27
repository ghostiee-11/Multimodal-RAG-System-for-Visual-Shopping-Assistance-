# llm_handler.py
import os
import google.generativeai as genai
from PIL import Image # For sending image data to Gemini Pro Vision
from dotenv import load_dotenv
import json 
import re

# --- Configuration ---
GEMINI_ANSWER_MODEL_NAME = "gemini-1.5-flash-latest" # Or "gemini-pro-vision" for vision capabilities
QUERY_PARSER_MODEL_NAME = "gemini-1.5-flash-latest" # Or "gemini-pro" for robust parsing

def configure_gemini():
    """Configures the Gemini API with the API key from environment variables."""
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
    """
    Formats a list of (user_query, assistant_answer) tuples into a string 
    for inclusion in LLM prompts.
    """
    if not history_tuples:
        return ""
    
    formatted_history = "\n\n--- Previous Conversation Turn(s) ---\n"
    # Take the last 'max_turns' from the history
    recent_history = history_tuples[-max_turns:]
    for i, (user_turn, assistant_turn) in enumerate(recent_history):
        # Use a numbering that indicates recency, e.g., Turn -N, Turn -N+1 ... Turn -1
        turn_number = -len(recent_history) + i 
        formatted_history += f"User (Turn {turn_number}): {user_turn}\n"
        formatted_history += f"Assistant (Turn {turn_number}): {assistant_turn}\n---\n"
    return formatted_history.strip()


def parse_user_query_with_gemini(user_query, conversation_history_str=""):
    """
    Uses Gemini to parse the user's current query, considering conversation history.
    Returns a dictionary with extracted information.
    """
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
5.  "key_features_attributes": (List of Strings) Explicit features/attributes/constraints from CURRENT query, or implied by follow-up. Normalize where possible (e.g., "color:blue", "price_max:2000").
6.  "comparison_entities": (List of Strings) If intent_type is "PRODUCT_COMPARISON", list distinct product/brand names being compared (at least two). Else, empty list.
7.  "visual_aspects_queried": (List of Strings) Visual aspects user is interested in (e.g., ["look", "design", "color"]).
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
        match = re.search(r"```json\s*([\s\S]*?)\s*```", cleaned_response_text, re.IGNORECASE) # Added re.IGNORECASE
        if match:
            cleaned_response_text = match.group(1).strip()
        else: 
            if cleaned_response_text.startswith("```"): cleaned_response_text = cleaned_response_text[3:]
            if cleaned_response_text.endswith("```"): cleaned_response_text = cleaned_response_text[:-3]
            cleaned_response_text = cleaned_response_text.strip()
            
        parsed_info = json.loads(cleaned_response_text)
        defaults = {
            "original_query": user_query, "intent_type": "GENERAL_PRODUCT_SEARCH", 
            "product_entities": [], "brand_entities": [], "key_features_attributes": [],
            "comparison_entities": [], "visual_aspects_queried": [],
            "rewritten_query_for_retrieval": user_query, "decomposition_hints": []
        }
        for key, default_val in defaults.items():
            parsed_info.setdefault(key, default_val)

        # print(f"LLM Handler: Parsed query info: {json.dumps(parsed_info, indent=2)}")
        return parsed_info

    except Exception as e:
        print(f"LLM Handler Error: During query parsing with Gemini: {e}")
        return {
            "original_query": user_query, "intent_type": "GENERAL_PRODUCT_SEARCH", 
            "product_entities": [], "brand_entities": [],
            "key_features_attributes": [user_query], "comparison_entities": [],
            "visual_aspects_queried": [], "rewritten_query_for_retrieval": user_query,
            "decomposition_hints": []
        }

def generate_answer_with_gemini(user_query, multimodal_contexts_list, parsed_query_info=None, conversation_history_str=""):
    if not multimodal_contexts_list and not user_query:
        return "I need more information or a query to help you."
    if not multimodal_contexts_list and not conversation_history_str: # Allow if only history exists for pure conversational follow-up
        return "I couldn't find specific product information related to your current query."

    try:
        model = genai.GenerativeModel(GEMINI_ANSWER_MODEL_NAME)
    except Exception as e:
        return f"LLM Handler Error: Initializing Gemini model '{GEMINI_ANSWER_MODEL_NAME}': {e}"

    prompt_parts = []
    image_pil_objects_for_llm = {} 
    
    is_comparison_task = parsed_query_info and parsed_query_info.get("intent_type") == "PRODUCT_COMPARISON" and \
                         len(parsed_query_info.get("comparison_entities", [])) >= 2
    product_names_for_comparison = parsed_query_info.get("comparison_entities", []) if is_comparison_task else []

    # Add conversation history first if it exists
    if conversation_history_str:
        prompt_parts.append(conversation_history_str) # Already formatted with headers
        prompt_parts.append("\n--- Current Interaction & Provided Context ---")


    if is_comparison_task:
        prompt_parts.append(f"""You are a helpful and precise **visual shopping assistant** continuing a conversation.
Your task is to **compare products: { ' and '.join([f"'{name.strip().title()}'" for name in product_names_for_comparison]) }** based on the user's current question and the specific context provided below for each product. If relevant, refer to the previous conversation turns.
Base your comparison STRICTLY on the provided context (text, image captions, and image content itself).
**Refer to visual details from the images and their captions to confirm or elaborate on textual information when relevant to the comparison.**
Highlight key differences and similarities relevant to the user's current query: "{user_query}".
If specific information for comparison is missing, state that clearly. Do NOT use external knowledge. Format your answer clearly.

---
**User’s Current Question:** "{user_query}"
---
**Provided Context (for current question, organized by product if applicable):**
""")
    else: 
        prompt_parts.append(f"""You are a helpful and precise **visual shopping assistant** continuing a conversation.
Your task is to answer the user's current question about products. If relevant, refer to the previous conversation turns.
Base your answer STRICTLY on the provided new context below. The context includes textual information and associated product images with captions.
**Crucially, use the visual information from the images and their captions to confirm, illustrate, or add detail to the textual information whenever possible.**

⚠️ **Key Instructions:**
1.  **Adhere to Context & History:** Use ONLY the provided new context and relevant information from previous turns. Do not use external knowledge.
2.  **Integrate Visuals:** Explicitly mention if you are using information from an image or its caption.
3.  **Cannot Answer:** If the information is not in the new context or relevant history, state: "I cannot answer this based on the information I have."
4.  **Be Factual and Relevant to the CURRENT question.**

---
**User’s Current Question:** "{user_query}"
---
**Provided Context (for current question):**
""")
    
    # Assemble textual context from multimodal_contexts_list
    context_text_blob_parts = [] 
    current_product_header = None # For comparison task, to group context by product
    max_llm_context_items = 6 # Cap on number of distinct context snippets (text or image-group)

    for i, context_item in enumerate(multimodal_contexts_list[:max_llm_context_items]):
        item_type = context_item.get("type")
        temp_text_parts = []

        if item_type == "comparison_intro": 
            product_name_intro = context_item.get("product_name", "Unnamed Product").strip().title()
            if current_product_header and current_product_header.lower() != product_name_intro.lower():
                 temp_text_parts.append("\n--- End of Details for Previous Product ---") 
            current_product_header = product_name_intro
            temp_text_parts.append(f"\n\n**CONTEXT BLOCK FOR PRODUCT: {current_product_header.upper()}**")
        else: # For text_derived_context or image_derived_context
            temp_text_parts.append(f"\nContext Snippet (Source: {item_type}):")
            prod_info = context_item.get('associated_product_info')
            prod_id = context_item.get('associated_product_id')
            if prod_info and prod_info.get('title'):
                temp_text_parts.append(f"  Product: {prod_info['title']} (ID: {prod_id})")
            elif prod_id:
                temp_text_parts.append(f"  Product ID: {prod_id}")

            if item_type == 'text_derived_context':
                temp_text_parts.append(f"  Relevance Score (Text): {context_item.get('text_score', 0):.3f}")
                if context_item.get('text_metadata_details'):
                    temp_text_parts.append(f"  Source Detail: {context_item['text_metadata_details'].get('aspect', 'N/A')} ({context_item['text_metadata_details'].get('sentiment', 'N/A')})")
                temp_text_parts.append(f"  Text: \"{context_item.get('text_content', '')}\"")
            
            elif item_type == 'image_derived_context':
                score_disp = f"ViLT: {context_item.get('vilt_score', 'N/A'):.3f}" if context_item.get('vilt_score') is not None else f"CLIP: {context_item.get('image_score', 0):.3f}"
                temp_text_parts.append(f"  Relevance (Image - {score_disp})")
                temp_text_parts.append(f"  Image File Ref: '{os.path.basename(context_item.get('image_path', 'N/A'))}'")
                # Include multiple captions
                primary_c = context_item.get('primary_caption', 'N/A')
                all_c = context_item.get('all_captions', [])
                distinct_caps = list(set([c for c in [primary_c] + all_c if c and c.strip() and c != 'N/A']))[:3]
                if distinct_caps:
                    for cap_idx, cap_text in enumerate(distinct_caps):
                        temp_text_parts.append(f"    Caption Variant {cap_idx + 1}: \"{cap_text}\"")
                else: temp_text_parts.append("    Caption: Not available.")
            
            if prod_info and prod_info.get('price'): temp_text_parts.append(f"  Price: {prod_info.get('price')}")

            # Associated images for this context item (text or image derived)
            associated_imgs = context_item.get('associated_images', [])
            if associated_imgs:
                temp_text_parts.append("  Associated Visual(s):")
                for img_detail in associated_imgs[:1]: # Take first associated image
                    img_path = img_detail.get('image_path')
                    img_filename = os.path.basename(img_path) if img_path else "N/A"
                    cap_primary_assoc = img_detail.get('primary_caption', 'N/A')
                    temp_text_parts.append(f"    - File: '{img_filename}', Caption: \"{cap_primary_assoc}\" (Content provided separately)")
                    if img_path and os.path.exists(img_path) and img_path not in image_pil_objects_for_llm:
                        try: image_pil_objects_for_llm[img_path] = Image.open(img_path).convert("RGB")
                        except Exception as e_img: print(f"LLM Warn: Could not load image {img_path}: {e_img}")
        
        context_text_blob_parts.append("\n".join(temp_text_parts))
    
    prompt_parts.append("\n".join(context_text_blob_parts))

    if image_pil_objects_for_llm:
        prompt_parts.append("\n\n--- Attached Image Data (filenames referenced in context) ---")
        for img_path_key, pil_image_obj in image_pil_objects_for_llm.items():
            prompt_parts.append(f"\nImage Content for '{os.path.basename(img_path_key)}':")
            prompt_parts.append(pil_image_obj)

    prompt_parts.append(f"""
---
Based ONLY on the provided new context and relevant information from previous conversation turns (if any), please now address the user's current question: "{user_query}"
Remember to adhere to all instructions, especially regarding context adherence and visual integration.
""")

    try:
        # print(f"\n🤖 LLM Handler: Sending request to Gemini ({GEMINI_ANSWER_MODEL_NAME})...")
        # for i, p_part in enumerate(prompt_parts): # DEBUG: Print prompt parts
        #     if isinstance(p_part, str): print(f"PROMPT PART {i} (text):\n{p_part[:500]}...\n")
        #     else: print(f"PROMPT PART {i} (image): {type(p_part)}\n")

        generation_config = genai.types.GenerationConfig(max_output_tokens=1024, temperature=0.25)
        response = model.generate_content(prompt_parts, generation_config=generation_config, request_options={'timeout': 120}) # Added timeout
        
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            print(f"LLM Handler Warning: Prompt was blocked. Reason: {response.prompt_feedback.block_reason}")
            return "I'm sorry, I can't provide an answer to that query due to content restrictions."
        if not response.candidates or not response.candidates[0].content.parts:
             print("LLM Handler Warning: No content parts in Gemini response.")
             return "I received an empty response. Please try rephrasing your query."
        return response.text
    except Exception as e:
        print(f"LLM Handler Error: During Gemini API call: {e}")
        return "Sorry, I encountered an error while generating a response for your query."

if __name__ == '__main__':
    try:
        configure_gemini()
    except Exception as e:
        print(f"LLM Handler Test: Failed to configure Gemini: {e}"); exit()

    print("\n--- LLM Handler Test: Query Parser (No History) ---")
    test_q1 = "Tell me about Sony WH-1000XM5 battery and noise cancellation."
    parsed1 = parse_user_query_with_gemini(test_q1)
    print("-" * 20)

    print("\n--- LLM Handler Test: Query Parser (With History) ---")
    dummy_history = [
        ("Tell me about Sony WH-1000XM5.", "The Sony WH-1000XM5 is a premium headphone known for excellent noise cancellation and long battery life, up to 40 hours."),
    ]
    formatted_hist_str = format_conversation_history_for_prompt(dummy_history)
    test_q2_follow_up = "What about its price and available colors?"
    parsed2 = parse_user_query_with_gemini(test_q2_follow_up, formatted_hist_str)
    print("-" * 20)

    print("\n--- LLM Handler Test: Answer Generation (No History, Dummy Context) ---")
    # Create dummy images if they don't exist
    dummy_image_dir = "dummy_test_images_llm"
    os.makedirs(dummy_image_dir, exist_ok=True)
    dummy_img_path1 = os.path.join(dummy_image_dir, "llm_dummy1.jpg")
    if not os.path.exists(dummy_img_path1): Image.new('RGB', (60, 30), color = 'cyan').save(dummy_img_path1)

    dummy_context_no_hist = [{
        "type": "text_derived_context", "text_content": "Sony WH-1000XM5 features industry-leading noise cancellation and up to 30 hours battery with NC on.",
        "associated_product_id": "XM5", "associated_product_info": {"title": "Sony WH-1000XM5"},
        "associated_images": [{"image_path": dummy_img_path1, "primary_caption": "Black Sony XM5 headphones"}]
    }]
    answer_no_hist = generate_answer_with_gemini(test_q1, dummy_context_no_hist, parsed1)
    print(f"Answer (No History):\n{answer_no_hist}")
    print("-" * 20)
    
    print("\n--- LLM Handler Test: Answer Generation (With History, Dummy Context for follow-up) ---")
    # Context for the follow-up might be empty if the specific details (price, color) aren't retrieved for this turn
    # The LLM should ideally use the history to know "its" refers to XM5.
    answer_with_hist = generate_answer_with_gemini(test_q2_follow_up, [], parsed2, formatted_hist_str)
    print(f"Answer (With History for follow-up, no new context):\n{answer_with_hist}")