# llm_handler.py
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import json # For parsing LLM's JSON output

# --- Configuration ---
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" 
# For parsing, sometimes a pure text model can also be efficient if JSON output is clean.
# However, gemini-1.5-flash should handle this well.
QUERY_PARSER_MODEL_NAME = "gemini-1.5-flash-latest" # Can be same or different

def configure_gemini():
    """Configures the Gemini API with the API key."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
        raise EnvironmentError("GOOGLE_API_KEY not configured.")
    
    try:
        genai.configure(api_key=api_key)
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        raise

def parse_user_query_with_gemini(user_query):
    """
    Uses Gemini to parse the user query and extract potential product names,
    brands, features, and query type.
    Returns a dictionary with extracted information.
    """
    print(f"\n🤖 Parsing user query with Gemini: '{user_query}'")
    try:
        parser_model = genai.GenerativeModel(QUERY_PARSER_MODEL_NAME)
        
        
        parsing_prompt = f"""Your sole task is to analyze the user's shopping query and extract key information.
User Query: "{user_query}"

Strictly return your answer ONLY as a JSON object with the following keys. If a category has no information, use an empty list [] or "N/A".
- "intent_type": (String) Classify the query's intent. Examples: "specific_product_feature", "product_comparison", "general_product_search", "opinion_request", "availability_check", "other_unrelated".
- "product_entities": (List of Strings) Specific product model names or series mentioned (e.g., ["WH-1000XM4", "Rockerz 430"]). Include brand if part of model name.
- "brand_entities": (List of Strings) Distinct brand names mentioned if not part of a specific model entity (e.g., ["Sony", "JBL", "boAt"]).
- "key_features_attributes": (List of Strings) Features, attributes, or constraints mentioned by the user (e.g., ["battery life", "noise cancellation", "color:blue", "price_max:2000", "price_min:1000", "material:leather", "type:over-ear", "suitable_for:running"]). Try to normalize constraints like color and price.
- "comparison_entities": (List of Strings) If intent_type is "product_comparison", a list of exactly two distinct, core product/brand names being compared. Otherwise, an empty list.

Example 1 (Specific Feature):
User Query: "What is the battery life of Sony WH-1000XM4 headphones?"
Output JSON: {{"intent_type": "specific_product_feature", "product_entities": ["Sony WH-1000XM4"], "brand_entities": [], "key_features_attributes": ["battery life"], "comparison_entities": []}}

Example 2 (Comparison):
User Query: "compare jbl tune 510bt and sony wh-ch520 for travel"
Output JSON: {{"intent_type": "product_comparison", "product_entities": ["jbl tune 510bt", "sony wh-ch520"], "brand_entities": [], "key_features_attributes": ["suitable_for:travel"], "comparison_entities": ["jbl tune 510bt", "sony wh-ch520"]}}

Example 3 (Multi-Constraint Search):
User Query: "blue over-ear headphones under 2000 rupees with good noise cancellation"
Output JSON: {{"intent_type": "general_product_search", "product_entities": [], "brand_entities": [], "key_features_attributes": ["color:blue", "type:over-ear", "price_max:2000", "good noise cancellation"], "comparison_entities": []}}

Example 4 (Opinion):
User Query: "Are Boat headphones comfortable for long hours?"
Output JSON: {{"intent_type": "opinion_request", "product_entities": [], "brand_entities": ["Boat"], "key_features_attributes": ["comfort for long hours"], "comparison_entities": []}}

Now, parse the User Query provided at the top. Output ONLY the JSON object.
"""
        # print(f"DEBUG: Parsing prompt sent to Gemini:\n{parsing_prompt}") 
        response = parser_model.generate_content(parsing_prompt)
        
        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[len("```json"):]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-len("```")]
        cleaned_response_text = cleaned_response_text.strip()
            
        # print(f"DEBUG: Cleaned JSON response from parser: {cleaned_response_text}")
        
        parsed_info = json.loads(cleaned_response_text)
        print(f"Parsed query info: {parsed_info}")
        return parsed_info

    except Exception as e:
        print(f"Error during query parsing with Gemini: {e}")
        response_text_for_error = "N/A"
        if 'response' in locals() and hasattr(response, 'text'):
            response_text_for_error = response.text
        print(f"Problematic response text (if available): {response_text_for_error}")
        return { # Fallback structure
            "intent_type": "general_search", 
            "product_entities": [], "brand_entities": [],
            "key_features_attributes": [user_query], 
            "comparison_entities": []
        }

def generate_answer_with_gemini(user_query, multimodal_contexts_list, parsed_query_info=None):
    if not multimodal_contexts_list and not user_query:
        return "I need more information or a query to help you."
    if not multimodal_contexts_list: 
        return "I couldn't find specific product information related to your query to form an answer."

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        print(f"Error initializing Gemini model '{GEMINI_MODEL_NAME}': {e}")
        return "Sorry, I'm having trouble connecting to my brain right now."

    prompt_parts = []
    image_parts_to_send = {} 
    
    is_comparison_task = parsed_query_info and parsed_query_info.get("intent_type") == "product_comparison" and \
                         len(parsed_query_info.get("comparison_entities", [])) == 2
    
    product_names_for_comparison = parsed_query_info.get("comparison_entities", []) if is_comparison_task else []

    # --- System Role / Initial Instruction ---
    if is_comparison_task:
        prompt_parts.append(f"""You are a helpful and precise **visual shopping assistant**.
Your task is to **compare two products: '{product_names_for_comparison[0].strip().title()}' and '{product_names_for_comparison[1].strip().title()}'** based on the user's question and the specific context provided below for each product.
Base your comparison STRICTLY on the provided context for '{product_names_for_comparison[0].strip().title()}' and the separate context for '{product_names_for_comparison[1].strip().title()}'.
Highlight key differences and similarities relevant to the user's query: "{user_query}".
If specific information for comparison on a certain aspect is missing for one or both products in their respective context sections, state that clearly.
Do NOT use any external knowledge or make assumptions.

---
**User’s Original Comparison Question:** "{user_query}"
---
**Information for Comparison Will Follow (organized by product):**
""")
    else: 
        prompt_parts.append(f"""You are a helpful and precise **visual shopping assistant**.
Your task is to answer the user's question about products.
Base your answer STRICTLY on the provided context below. The context includes textual information (product specifications, descriptions, reviews) and associated product images with their AI-generated captions.
Refer to all provided textual details, image captions, and the images themselves to understand product attributes.

⚠️ **Crucial Instructions:**
1. **Strictly Adhere to Context:** Do NOT use any external knowledge or make assumptions beyond what is explicitly stated.
2. **Cannot Answer:** If the information to answer the question is not in the context, you MUST reply with: "I cannot answer this based on the information I have." Do not guess or infer.
3. **Be Factual and Relevant:** Focus on clarity, factual accuracy, and direct relevance. Do not speculate.

---
**User’s Question:** "{user_query}"
---
**Provided Context:**
""")

    context_text_blob = [] 
    current_product_being_described_for_comparison = None
    max_llm_context_items = 6 

    for i, context_item in enumerate(multimodal_contexts_list[:max_llm_context_items]):
        item_type = context_item.get("type")

        if item_type == "comparison_intro":
            product_name = context_item.get("product_name", "Unnamed Product")
            if current_product_being_described_for_comparison and current_product_being_described_for_comparison.lower() != product_name.lower() :
                 context_text_blob.append("\n--- End of Details for Previous Product ---") 
            current_product_being_described_for_comparison = product_name.strip().title()
            context_text_blob.append(f"\n\n**CONTEXT BLOCK FOR PRODUCT: {current_product_being_described_for_comparison.upper()}**")
            continue 

        context_text_blob.append(f"\nContext Snippet (Source Type: {item_type}):")

        if item_type == 'text_derived_context':
            context_text_blob.append(f"  Relevance Score (Text): {context_item.get('text_score', 0):.2f}")
            if context_item.get('text_metadata_details'):
                aspect = context_item['text_metadata_details'].get('aspect', 'N/A')
                context_text_blob.append(f"  Regarding Aspect: {aspect}")
            context_text_blob.append(f"  Textual Detail: \"{context_item.get('text_content', '')}\"")
        elif item_type == 'image_derived_context':
            context_text_blob.append(f"  Relevance Score (Image): {context_item.get('image_score', 0):.2f}")
            context_text_blob.append(f"  Image (for reference): '{os.path.basename(context_item.get('image_path', 'N/A'))}'")
            context_text_blob.append(f"  Image Caption: \"{context_item.get('image_caption', 'N/A')}\"")
        
        if context_item.get('associated_product_info'):
            p_info = context_item['associated_product_info']
            product_title_in_context = p_info.get('title', 'N/A').strip().title()
            if not current_product_being_described_for_comparison or \
               (current_product_being_described_for_comparison and product_title_in_context.startswith(current_product_being_described_for_comparison)):
                context_text_blob.append(f"  Product Name: {product_title_in_context} (ID: {context_item.get('associated_product_id', 'N/A')})")
            elif current_product_being_described_for_comparison:
                context_text_blob.append(f"  (Details for product: {product_title_in_context})")
            if p_info.get('price'):
                 context_text_blob.append(f"  Price: {p_info.get('price')}")

        images_in_this_item = context_item.get('associated_images', [])
        if images_in_this_item:
            context_text_blob.append("  Associated Visual(s) for this snippet:")
            for img_detail in images_in_this_item[:1]: 
                img_path = img_detail.get('image_path')
                img_caption = img_detail.get('caption', 'N/A') 
                if img_path and os.path.exists(img_path):
                    context_text_blob.append(f"    - Image: '{os.path.basename(img_path)}' with Caption: \"{img_caption}\" (Image data will be provided if unique).")
                    if img_path not in image_parts_to_send: 
                        try: img_pil = Image.open(img_path).convert("RGB"); image_parts_to_send[img_path] = img_pil 
                        except Exception as e: print(f"Warning: Could not load image {img_path} for LLM: {e}")
                elif img_path: context_text_blob.append(f"    - Image ref (not found): {os.path.basename(img_path)}")
    
    prompt_parts.append("\n".join(context_text_blob)) 

    if image_parts_to_send:
        prompt_parts.append("\n\n--- Attached Image Data (filenames referenced above) ---")
        for img_path_key, pil_image in image_parts_to_send.items():
            prompt_parts.append(f"\nImage Content for '{os.path.basename(img_path_key)}':")
            prompt_parts.append(pil_image) 

    if is_comparison_task and len(product_names_for_comparison) == 2:
        prompt_parts.append(f"""
---
Based ONLY on the details provided in the 'CONTEXT BLOCK FOR PRODUCT: {product_names_for_comparison[0].strip().title().upper()}' and 'CONTEXT BLOCK FOR PRODUCT: {product_names_for_comparison[1].strip().title().upper()}' sections above, please now address the user's comparison question: "{user_query}"
Focus on similarities and differences for the aspects mentioned or implied by the user's question.
If information for a comparison point is missing for one or both products, explicitly state that. Do not invent information.
""")
    elif is_comparison_task: 
         prompt_parts.append(f"""
---
Now, based ONLY on the details provided above, please address the user's comparison question: "{user_query}"
Remember to state if information is missing.
""")

    try:
        print(f"\nSending request to Gemini ({GEMINI_MODEL_NAME}) with {len(image_parts_to_send)} image(s)...")
        generation_config = genai.types.GenerationConfig(max_output_tokens=800, temperature=0.5)
        response = model.generate_content(prompt_parts, generation_config=generation_config)
        return response.text
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return "Sorry, I encountered an error while generating a response for your query."

if __name__ == '__main__':
    try:
        configure_gemini()
    except Exception as e:
        print(f"Failed to configure Gemini for testing: {e}")
        exit()

    print("\n--- Testing Query Parser ---")
    test_queries_for_parser = [
        "does boat rockers 430 have voice assistant",
        "compare jbl tune 510bt and sony wh-ch520 price",
        "blue headphones under 1500",
        "What do users say about comfort of Bose QuietComfort Ultra?",
        "Tell me about Sony headphones"
    ]
    for t_query in test_queries_for_parser:
        parsed = parse_user_query_with_gemini(t_query)
        print("-" * 20)
    
    # ... (You can add back your more detailed dummy context tests for generate_answer_with_gemini if needed) ...