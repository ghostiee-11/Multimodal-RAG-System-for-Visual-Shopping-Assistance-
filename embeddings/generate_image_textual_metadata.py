# generate_image_textual_metadata.py
import os
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps 
from transformers import BlipProcessor, BlipForConditionalGeneration
import time
import json
from sentence_transformers import SentenceTransformer, util
import re
import logging
import pytesseract 
from collections import Counter
import numpy as np

try:
    import cv2
    opencv_available = True
except ImportError:
    opencv_available = False

# --- Configuration ---
ALL_SCRAPED_IMAGES_INFO_CSV = 'all_product_images_info_scraped.csv' 
PRODUCTS_CSV_FOR_TRUSTED_TEXT = '/Users/amankumar/Desktop/Aims/final data/products_final_with_all_image_paths.csv' 
ALL_DOCS_CSV_FOR_TRUSTED_TEXT = '/Users/amankumar/Desktop/Aims/final data/all_documents.csv'

CAPTION_MODEL_ID_BLIP = "Salesforce/blip-image-captioning-large"
CAPFILT_SCORING_MODEL_ST = 'all-MiniLM-L6-v2' 
CAPFILT_SIMILARITY_THRESHOLD = 0.25 

# OCR Parameters
MIN_OCR_LINE_LENGTH_STRICT = 4       
MAX_OCR_LINE_WORDS_STRICT = 10      
MIN_OCR_ALPHA_NUM_RATIO_STRICT = 0.4 
MIN_OCR_CAPITAL_WORD_RATIO_STRICT = 0.2 # Defined for use in OCR filtering
MAX_OCR_SEGMENTS_TO_KEEP = 7          # Max distinct OCR phrases to keep per image

# BLIP Caption Generation Parameters
MAX_RAW_CAPTIONS_PER_IMAGE_BLIP = 5
# MAX_FILTERED_BLIP_CAPTIONS_TO_STORE = 3 # This will be implicitly handled by MAX_COMBINED_TEXTS_TO_STORE

# Combined Output
MAX_COMBINED_TEXTS_TO_STORE = 6 # How many total (OCR + BLIP) to store per image

OUTPUT_COMBINED_TEXTS_CSV = 'image_combined_blip_ocr_filtered_final.csv'

# --- Logging Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH_GEN = os.path.join(current_script_dir, "image_text_generation.log")
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH_GEN, mode='w'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
if opencv_available: logger.info("OpenCV available for advanced image processing.")
else: logger.warning("OpenCV not available, OCR preprocessing will be PIL-based.")

# --- Global Model Variables & DataFrames ---
caption_processor_blip = None
caption_model_blip = None
caption_device_blip = None
capfilt_model_st_instance = None
df_products_trusted_text = None
df_specs_trusted_text = None

def load_trusted_text_data_for_capfilt():
    global df_products_trusted_text, df_specs_trusted_text
    loaded_successfully = True
    try:
        if df_products_trusted_text is None:
            df_products_trusted_text = pd.read_csv(PRODUCTS_CSV_FOR_TRUSTED_TEXT)
            df_products_trusted_text['product_id'] = df_products_trusted_text['product_id'].astype(str).str.strip()
            logger.info(f"CapFilt: Loaded products data ({df_products_trusted_text.shape}) from '{PRODUCTS_CSV_FOR_TRUSTED_TEXT}'")
    except Exception as e:
        logger.error(f"CapFilt Error loading products data: {e}", exc_info=True); loaded_successfully = False
    try:
        if df_specs_trusted_text is None:
            df_alldocs_temp = pd.read_csv(ALL_DOCS_CSV_FOR_TRUSTED_TEXT)
            relevant_doc_types = ['specification', 'description_paragraph', 'description_full']
            df_specs_trusted_text = df_alldocs_temp[df_alldocs_temp['doc_type'].isin(relevant_doc_types)].copy()
            df_specs_trusted_text['product_id'] = df_specs_trusted_text['product_id'].astype(str).str.strip()
            logger.info(f"CapFilt: Loaded relevant documents ({df_specs_trusted_text.shape}) from '{ALL_DOCS_CSV_FOR_TRUSTED_TEXT}'")
    except Exception as e:
        logger.error(f"CapFilt Error loading all_documents data: {e}", exc_info=True); loaded_successfully = False
    return loaded_successfully

def get_trusted_text_for_product_capfilt(product_id):
    if df_products_trusted_text is None or df_specs_trusted_text is None: return ""
    trusted_texts_list = []
    product_row = df_products_trusted_text[df_products_trusted_text['product_id'] == product_id]
    if not product_row.empty and pd.notna(product_row.iloc[0]['title']):
        trusted_texts_list.append(str(product_row.iloc[0]['title']))
    spec_rows = df_specs_trusted_text[df_specs_trusted_text['product_id'] == product_id]
    if not spec_rows.empty:
        specs_to_add = spec_rows['text_content'].dropna().astype(str).tolist()
        short_specs = [s for s in specs_to_add if len(s.split()) < 30][:3]
        long_specs = [s for s in specs_to_add if len(s.split()) >= 30][:2]
        trusted_texts_list.extend(short_specs); trusted_texts_list.extend(long_specs)
    combined_text = ". ".join(list(set(trusted_texts_list)))
    combined_text = re.sub(r'\s+', ' ', combined_text).strip()
    if not combined_text and not product_row.empty and pd.notna(product_row.iloc[0]['title']):
        return str(product_row.iloc[0]['title'])
    return combined_text

def initialize_all_models():
    global caption_processor_blip, caption_model_blip, caption_device_blip, capfilt_model_st_instance
    models_ready = True
    if caption_model_blip is None or caption_processor_blip is None:
        logger.info(f"Loading BLIP model: {CAPTION_MODEL_ID_BLIP}...")
        caption_device_blip = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {caption_device_blip} for BLIP.")
        try:
            caption_processor_blip = BlipProcessor.from_pretrained(CAPTION_MODEL_ID_BLIP)
            caption_model_blip = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID_BLIP).to(caption_device_blip)
            caption_model_blip.eval(); logger.info("BLIP model loaded.")
        except Exception as e: logger.error(f"BLIP load error: {e}", exc_info=True); models_ready = False
    if capfilt_model_st_instance is None:
        logger.info(f"Loading SentenceTransformer for CapFilt: {CAPFILT_SCORING_MODEL_ST}...")
        st_device = "cuda" if torch.cuda.is_available() else "cpu" 
        try:
            capfilt_model_st_instance = SentenceTransformer(CAPFILT_SCORING_MODEL_ST, device=st_device)
            logger.info(f"SentenceTransformer for CapFilt loaded on {st_device}.")
        except Exception as e: logger.error(f"ST CapFilt load error: {e}", exc_info=True); models_ready = False
    return models_ready

def load_all_scraped_images_info():
    try:
        df = pd.read_csv(ALL_SCRAPED_IMAGES_INFO_CSV)
        if 'full_image_path' not in df.columns or 'product_id' not in df.columns:
            raise ValueError("CSV must contain 'full_image_path' and 'product_id'.")
        df.dropna(subset=['full_image_path', 'product_id'], inplace=True)
        df['product_id'] = df['product_id'].astype(str).str.strip()
        df['full_image_path'] = df['full_image_path'].astype(str).str.strip()
        logger.info(f"Loaded {len(df)} image records from {ALL_SCRAPED_IMAGES_INFO_CSV}.")
        return df
    except FileNotFoundError: logger.error(f"Error: CSV not found at {ALL_SCRAPED_IMAGES_INFO_CSV}"); return None
    except ValueError as ve: logger.error(f"Error in CSV structure: {ve}"); return None
    except Exception as e: logger.error(f"Error loading {ALL_SCRAPED_IMAGES_INFO_CSV}: {e}", exc_info=True); return None

def basic_heuristic_caption_filter(caption_text, min_length=8, max_word_repetitions_allowed=3, min_words=3):
    if not caption_text or not isinstance(caption_text, str) or len(caption_text) < min_length: return False
    words = [word for word in re.findall(r'\b\w+\b', caption_text.lower()) if len(word) > 1]
    if not words or len(words) < min_words: return False
    word_counts = Counter(words)
    if word_counts and word_counts.most_common(1)[0][1] > max_word_repetitions_allowed and len(words) > max_word_repetitions_allowed + 2:
        if len(set(words)) > 2 : return False
    if len(set(words)) <= 2 and len(words) > max_word_repetitions_allowed +1 : return False
    generic_starts = ("an image of", "a picture of", "a close up of", "this is a", "there is a", "photo of a", "image shows")
    if any(caption_text.lower().startswith(s) for s in generic_starts) and len(words) < 6: return False
    if caption_text.lower() in ["unknown", "n/a", "none", "image", "picture", "photo", "error", "failed"]: return False
    return True

def preprocess_image_for_ocr_focused(pil_image, scale_factor=2.0):
    logger.debug(f"    Starting advanced OCR preprocessing...")
    try:
        img = pil_image.convert('L') 
        width, height = img.size
        target_min_dimension = 700 
        if width < target_min_dimension or height < target_min_dimension:
            current_scale_factor = scale_factor
            if width > 0 and width * current_scale_factor < target_min_dimension: current_scale_factor = target_min_dimension / width
            if height > 0 and height * current_scale_factor < target_min_dimension: current_scale_factor = max(current_scale_factor, target_min_dimension / height)
            new_width = int(width * current_scale_factor)
            new_height = int(height * current_scale_factor)
            if new_width > 0 and new_height > 0:
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug(f"      Resized image from ({width}x{height}) to {img.size}")
        
        if opencv_available:
            img_cv = np.array(img) 
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img_cv_sharpened = cv2.filter2D(img_cv, -1, sharpen_kernel)
            img_thresh_cv = cv2.adaptiveThreshold(img_cv_sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 3) 
            logger.debug("      Applied OpenCV Adaptive Thresholding (Gaussian, Binary).")
            img_final_ocr = Image.fromarray(img_thresh_cv)
        else: 
            img_final_ocr = ImageOps.autocontrast(img, cutoff=1)
            img_final_ocr = img_final_ocr.filter(ImageFilter.UnsharpMask(radius=1.2, percent=180, threshold=2))
            logger.debug("      Applied PIL-based preprocessing.")
        return img_final_ocr
    except Exception as e:
        logger.error(f"Advanced OCR Preprocessing Error: {e}", exc_info=True)
        return pil_image.convert('L') 

def extract_text_with_ocr_focused(pil_image_original):
    if pil_image_original is None: return []
    preprocessed_pil_image = preprocess_image_for_ocr_focused(pil_image_original.copy())
    if preprocessed_pil_image is None: logger.warning("    OCR preprocessed image is None."); return []
    all_ocr_lines = set()
    ocr_configs = [
        (r'--oem 3 --psm 11 -l eng', "PSM 11 (Sparse)"), 
        (r'--oem 3 --psm 7 -l eng',  "PSM 7 (Single Line)"),
        (r'--oem 3 --psm 12 -l eng', "PSM 12 (Sparse OSD)"),
        (r'--oem 3 --psm 6 -l eng',  "PSM 6 (Single Block)"),
        (r'--oem 3 --psm 4 -l eng', "PSM 4 (Single Column)"),
    ]
    # logger.info("    Attempting OCR with different Tesseract PSM modes...") # Now done in main loop
    for config_str, config_name in ocr_configs:
        logger.debug(f"      OCR attempt with config: {config_name} ({config_str})")
        try:
            extracted_text = pytesseract.image_to_string(preprocessed_pil_image, config=config_str, timeout=12)
            lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
            logger.debug(f"        Raw lines from {config_name}: {lines}")
            for line in lines:
                cleaned_line = re.sub(r'[^\x20-\x7E]+', '', line) 
                cleaned_line = re.sub(r'\s\s+', ' ', cleaned_line).strip()
                cleaned_line = re.sub(r'^[^a-zA-Z0-9]+', '', cleaned_line)
                cleaned_line = re.sub(r'[^a-zA-Z0-9]+$', '', cleaned_line)
                cleaned_line = cleaned_line.strip()
                if not cleaned_line: continue
                words = re.findall(r'\b\w[\w\'.&\/-]*\w\b|\b\w\b', cleaned_line)
                num_words = len(words)
                # Use constants defined in the Configuration section
                if not (1 <= num_words <= MAX_OCR_LINE_WORDS_STRICT and MIN_OCR_LINE_LENGTH_STRICT <= len(cleaned_line) < 75): continue
                num_alpha = sum(c.isalpha() for c in cleaned_line); num_digit = sum(c.isdigit() for c in cleaned_line); num_alnum = num_alpha + num_digit
                if len(cleaned_line) > 0 and (num_alnum / len(cleaned_line)) < MIN_OCR_ALPHA_NUM_RATIO_STRICT: continue
                if num_alnum < 2 : continue 
                is_title_case_like = sum(1 for word in words if word and word[0].isupper()) / num_words > MIN_OCR_CAPITAL_WORD_RATIO_STRICT if num_words > 0 else False # Corrected constant
                has_digits = num_digit > 0
                if not (is_title_case_like or has_digits or num_words <=2):
                    if num_words > 2 : continue
                if cleaned_line.isupper() and num_words > 2 and len(cleaned_line) > 10 : continue
                if len(set(char for char in cleaned_line if char.isalnum())) < 3 and len(cleaned_line) > 4: continue
                all_ocr_lines.add(cleaned_line)
        except RuntimeError as e_rt: logger.debug(f"        OCR Runtime Error with {config_name}: {e_rt}")
        except Exception as ocr_e: logger.debug(f"        OCR Extraction Error with {config_name}: {ocr_e}")
    meaningful_lines = list(all_ocr_lines)
    meaningful_lines.sort(key=lambda x: (len(x.split()), len(x)), reverse=True) 
    final_ocr_texts = meaningful_lines[:MAX_OCR_SEGMENTS_TO_KEEP] # Corrected Constant
    # logger.info(f"    OCR extracted {len(final_ocr_texts)} unique, cleaned candidate lines:") # Moved to main loop for clarity per image
    # for i, line in enumerate(final_ocr_texts): print(f"      OCR Final Text {i+1}: {line}") # Moved
    return final_ocr_texts

def generate_raw_multiple_captions_blip(pil_image):
    global caption_processor_blip, caption_model_blip, caption_device_blip
    if caption_model_blip is None: raise RuntimeError("BLIP model not initialized.")
    raw_captions_set = set()
    common_generate_args = {"max_length": 55, "min_length": 6, "repetition_penalty": 1.25} 
    try:
        inputs = caption_processor_blip(pil_image, return_tensors="pt").to(caption_device_blip)
        with torch.no_grad():
            outputs_beam = caption_model_blip.generate(**inputs, num_beams=4, early_stopping=True, num_return_sequences=1, **common_generate_args)
            raw_captions_set.add(caption_processor_blip.decode(outputs_beam[0], skip_special_tokens=True).strip())
        with torch.no_grad():
            outputs_nucleus = caption_model_blip.generate(**inputs, do_sample=True, top_p=0.9, num_return_sequences=2, num_beams=1, **common_generate_args)
        for out_seq in outputs_nucleus: raw_captions_set.add(caption_processor_blip.decode(out_seq, skip_special_tokens=True).strip())
        with torch.no_grad():
            outputs_topk = caption_model_blip.generate(**inputs, do_sample=True, top_k=40, num_return_sequences=2, num_beams=1, **common_generate_args)
        for out_seq in outputs_topk: raw_captions_set.add(caption_processor_blip.decode(out_seq, skip_special_tokens=True).strip())
        prompts_for_blip = [
            "a detailed view of the headphones showing earcup design, padding, and control buttons",
            "the key visual features and materials of this headphone, including headband and microphone if visible",
            "describe the overall style and any specific branding elements on these headphones",
            "this image shows headphones with specific features like"
        ]
        for prompt_text in prompts_for_blip:
            inputs_prompted = caption_processor_blip(pil_image, text=prompt_text, return_tensors="pt").to(caption_device_blip)
            with torch.no_grad():
                outputs_prompted = caption_model_blip.generate(**inputs_prompted, num_beams=3, **common_generate_args) 
            cap_prompted = caption_processor_blip.decode(outputs_prompted[0], skip_special_tokens=True).strip()
            if cap_prompted.lower() != prompt_text.lower() and len(cap_prompted) > len(prompt_text.split(':')[0]) + 10:
                 raw_captions_set.add(cap_prompted)
    except Exception as e_blip: logger.warning(f"  Error during BLIP caption generation: {e_blip}")
    return [cap for cap in list(raw_captions_set) if basic_heuristic_caption_filter(cap)][:MAX_RAW_CAPTIONS_PER_IMAGE_BLIP]

def filter_texts_capfilt_style(all_candidate_texts, trusted_reference_text, product_id, blip_captions_from_generation, ocr_texts_from_generation):
    global capfilt_model_st_instance
    if not capfilt_model_st_instance:
        logger.warning("CapFilt: Scoring model not initialized..."); print("  CapFilt: Scoring model not ready...") 
        return [text for text in all_candidate_texts if basic_heuristic_caption_filter(text)][:MAX_COMBINED_TEXTS_TO_STORE] # Corrected Constant
    if not trusted_reference_text:
        logger.debug(f"CapFilt: No trusted text for product {product_id}..."); print(f"  CapFilt: No trusted text for {product_id}...") 
        return [text for text in all_candidate_texts if basic_heuristic_caption_filter(text)][:MAX_COMBINED_TEXTS_TO_STORE] # Corrected Constant
    if not all_candidate_texts: return []
    print(f"    CapFilt: Trusted reference for {product_id}: '{trusted_reference_text[:100]}...'")
    try:
        valid_candidates = [text for text in all_candidate_texts if text and isinstance(text, str) and len(text.split()) >= 1 and len(text) >= MIN_OCR_LINE_LENGTH_STRICT]
        if not valid_candidates: logger.debug(f"    CapFilt: No valid candidates for {product_id}."); return []
        
        trusted_text_embedding = capfilt_model_st_instance.encode(trusted_reference_text, convert_to_tensor=True)
        candidate_embeddings = capfilt_model_st_instance.encode(valid_candidates, convert_to_tensor=True)
        similarities = util.cos_sim(trusted_text_embedding, candidate_embeddings)[0]
        
        scored_candidates = []
        print(f"    CapFilt: Scoring {len(valid_candidates)} candidates for {product_id}:")
        for i, text_candidate in enumerate(valid_candidates): 
            score = similarities[i].item()
            is_ocr = text_candidate in ocr_texts_from_generation
            source_type = "OCR" if is_ocr else "BLIP" 
            scored_candidates.append({"text": text_candidate, "score": score, "source": source_type})
            print(f"      - Score: {score:.3f} | Source: {source_type} | Candidate: '{text_candidate[:70]}...'") 
        
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        filtered_texts_primary = [cand["text"] for cand in scored_candidates if cand["score"] >= CAPFILT_SIMILARITY_THRESHOLD]
        final_selection = list(dict.fromkeys(filtered_texts_primary)) 
        
        if len(final_selection) < MAX_COMBINED_TEXTS_TO_STORE and len(scored_candidates) > len(final_selection): # Corrected Constant
            for cand in scored_candidates:
                if len(final_selection) >= MAX_COMBINED_TEXTS_TO_STORE: break # Corrected Constant
                if cand["text"] not in final_selection:
                    if cand["source"] == "OCR" and len(cand["text"].split()) <= 5 and cand["score"] > (CAPFILT_SIMILARITY_THRESHOLD * 0.5):
                        print(f"    CapFilt: Adding potentially good OCR (lenient): '{cand['text'][:70]}...' (Score: {cand['score']:.3f})")
                        final_selection.append(cand["text"])
                    elif cand["source"] == "BLIP" and cand["score"] > (CAPFILT_SIMILARITY_THRESHOLD * 0.7):
                        print(f"    CapFilt: Adding decent BLIP (lenient): '{cand['text'][:70]}...' (Score: {cand['score']:.3f})")
                        final_selection.append(cand["text"])
        
        final_selection = list(dict.fromkeys(final_selection))
        print(f"    CapFilt: Kept {len(final_selection)} texts after considering scores and heuristics.")
        
        if not final_selection and scored_candidates: 
            best_overall_text = scored_candidates[0]["text"]
            logger.info(f"  CapFilt: All texts below threshold for {product_id}. Keeping best overall: '{best_overall_text[:50]}...' (Score: {scored_candidates[0]['score']:.3f})")
            print(f"    CapFilt: All below threshold. Keeping single best overall: '{best_overall_text[:70]}...' (Score: {scored_candidates[0]['score']:.3f})")
            return [best_overall_text]

        final_output_texts_with_scores = []
        for text in final_selection:
            original_entry = next((sc_entry for sc_entry in scored_candidates if sc_entry["text"] == text), None)
            if original_entry:
                final_output_texts_with_scores.append(original_entry)
        
        final_output_texts_with_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return [item["text"] for item in final_output_texts_with_scores][:MAX_COMBINED_TEXTS_TO_STORE] # Corrected Constant
    except Exception as e:
        logger.error(f"CapFilt Error during semantic filtering for {product_id}: {e}", exc_info=True)
        print(f"    CapFilt Error: {e}. Returning heuristically filtered raw candidates.")
        return [text for text in all_candidate_texts if basic_heuristic_caption_filter(text)][:MAX_COMBINED_TEXTS_TO_STORE] # Corrected Constant


if __name__ == '__main__':
    if not load_trusted_text_data_for_capfilt(): logger.error("Exiting: Failed trusted text load."); exit()
    df_all_images = load_all_scraped_images_info()
    if df_all_images is None or df_all_images.empty: logger.error("Exiting: No image data."); exit()
    if not initialize_all_models(): logger.error("Exiting: Model init failed."); exit()

    logger.info(f"\nStarting OCR, BLIP caption generation & filtering for {len(df_all_images)} images...")
    start_total_time = time.time()
    captioning_results = []

    for index, row in df_all_images.iterrows():
        image_full_path = row['full_image_path']
        product_id = row['product_id']
        print(f"\n---\nProcessing image {index + 1}/{len(df_all_images)}: {os.path.basename(image_full_path)} for Product ID: {product_id}")
        if not os.path.exists(image_full_path):
            logger.warning(f"    Image file not found: {image_full_path}. Skipping.")
            print(f"    SKIPPED: Image file not found: {image_full_path}")
            captioning_results.append({'full_image_path': image_full_path, 'product_id': product_id, 'generated_texts_json': json.dumps([])}); continue
        try:
            pil_image_original = Image.open(image_full_path)
        except Exception as e_img:
            logger.warning(f"    Could not open image {image_full_path}. Error: {e_img}. Skipping.")
            print(f"    SKIPPED: Could not open image {image_full_path}")
            captioning_results.append({'full_image_path': image_full_path, 'product_id': product_id, 'generated_texts_json': json.dumps([])}); continue

        start_img_processing_time = time.time()
        
        print("    1. Performing Focused OCR...")
        ocr_texts_raw = extract_text_with_ocr_focused(pil_image_original.copy()) 

        print("    2. Generating BLIP captions...")
        blip_captions_raw = generate_raw_multiple_captions_blip(pil_image_original.copy())
        print(f"       BLIP generated {len(blip_captions_raw)} heuristically valid raw captions: {blip_captions_raw}")
        
        all_candidate_texts = list(set(blip_captions_raw + ocr_texts_raw)) 
        print(f"    3. Total {len(all_candidate_texts)} unique candidate texts (BLIP + OCR) before semantic filtering.")

        if not all_candidate_texts:
             logger.warning(f"    No valid raw captions or OCR text for {os.path.basename(image_full_path)}.")
             print("       No valid candidates for filtering.")
             captioning_results.append({'full_image_path': image_full_path, 'product_id': product_id, 'generated_texts_json': json.dumps([])}); continue

        trusted_ref_text = get_trusted_text_for_product_capfilt(product_id)
        final_filtered_texts = filter_texts_capfilt_style(all_candidate_texts, trusted_ref_text, product_id, blip_captions_raw, ocr_texts_raw) 
        end_img_processing_time = time.time()
        
        if final_filtered_texts:
            print(f"    ---> Kept {len(final_filtered_texts)} texts for {os.path.basename(image_full_path)} (Processing time: {end_img_processing_time - start_img_processing_time:.2f}s).")
            for i, text_val in enumerate(final_filtered_texts): print(f"      Final Text {i+1}: {text_val}")
            captioning_results.append({'full_image_path': image_full_path, 'product_id': product_id,
                                       'generated_texts_json': json.dumps(final_filtered_texts)})
        else:
            logger.warning(f"    No texts kept after filtering for {os.path.basename(image_full_path)}")
            print(f"    ---> No texts kept for {os.path.basename(image_full_path)} after semantic filtering.")
            captioning_results.append({'full_image_path': image_full_path, 'product_id': product_id, 'generated_texts_json': json.dumps([]) })
        print("---")
        if index < len(df_all_images) -1 : time.sleep(0.2)
        
    end_total_time = time.time()
    logger.info(f"\nFinished OCR, BLIP caption generation & filtering for all images in {end_total_time - start_total_time:.2f} seconds.")
    df_final_captions = pd.DataFrame(captioning_results)
    output_cols = ['product_id', 'full_image_path', 'generated_texts_json'] 
    df_final_captions = df_final_captions.reindex(columns=[col for col in output_cols if col in df_final_captions.columns])
    
    df_final_captions.to_csv(OUTPUT_COMBINED_TEXTS_CSV, index=False, encoding='utf-8-sig')
    logger.info(f"\nSaved final filtered (BLIP + OCR) texts to {OUTPUT_COMBINED_TEXTS_CSV}")
    if not df_final_captions.empty: logger.info(f"Sample of generated texts:\n{df_final_captions.head()}")
    logger.info("\nImage Textual Metadata Generation Complete.")