import os
import pandas as pd
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

# --- Configuration ---
VALID_IMAGES_CSV_PATH = 'valid_product_images.csv'
# Model for image captioning
CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-large" 

OUTPUT_CAPTIONS_CSV = 'image_captions.csv'

# Global model and processor
caption_processor = None
caption_model = None
caption_device = None

def load_valid_image_data_for_captioning():
    """Loads the validated image data CSV."""
    try:
        df = pd.read_csv(VALID_IMAGES_CSV_PATH)
        # We only need unique full_image_path and their product_id
        df_unique_images = df[['full_image_path', 'product_id']].drop_duplicates(subset=['full_image_path']).reset_index(drop=True)
        print(f"Loaded {VALID_IMAGES_CSV_PATH}, found {len(df_unique_images)} unique images for captioning.")
        return df_unique_images
    except FileNotFoundError:
        print(f"Error: Valid image data CSV not found at {VALID_IMAGES_CSV_PATH}")
        return None

def initialize_captioning_model():
    """Initializes the BLIP model and processor."""
    global caption_processor, caption_model, caption_device

    if caption_model is None or caption_processor is None:
        print(f"Loading image captioning model: {CAPTION_MODEL_ID}...")
        caption_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {caption_device}")

        try:
            caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_ID)
            caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID).to(caption_device)
            caption_model.eval() # Set to evaluation mode
            print("Image captioning model and processor loaded.")
        except Exception as e:
            print(f"Error loading captioning model: {e}")
            raise

def generate_caption_for_image(image_path):
    """Generates a caption for a single image."""
    global caption_processor, caption_model, caption_device

    if caption_model is None or caption_processor is None:
        raise RuntimeError("Captioning model not initialized. Call initialize_captioning_model() first.")

    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: Could not open image {image_path}. Error: {e}")
        return None

    # You can provide a conditional prompt if desired, or let BLIP generate unconditionally
    # text_prompt = "a photography of" # Example conditional prompt
    # inputs = caption_processor(pil_image, text_prompt, return_tensors="pt").to(caption_device)
    
    # Unconditional captioning
    inputs = caption_processor(pil_image, return_tensors="pt").to(caption_device)

    with torch.no_grad(): # Important for inference
        outputs = caption_model.generate(**inputs, max_length=74, num_beams=3) # max_length and num_beams can be tuned

    caption = caption_processor.decode(outputs[0], skip_special_tokens=True)
    return caption.strip()


if __name__ == '__main__':
    df_images_to_caption = load_valid_image_data_for_captioning()

    if df_images_to_caption is not None and not df_images_to_caption.empty:
        try:
            initialize_captioning_model()
        except Exception as e:
            print(f"Failed to initialize captioning model: {e}")
            exit()

        print(f"\nStarting caption generation for {len(df_images_to_caption)} images...")
        start_total_time = time.time()
        
        results = []
        for index, row in df_images_to_caption.iterrows():
            img_path = row['full_image_path']
            product_id = row['product_id']
            
            print(f"  Processing image {index + 1}/{len(df_images_to_caption)}: {os.path.basename(img_path)}")
            start_img_time = time.time()
            caption = generate_caption_for_image(img_path)
            end_img_time = time.time()
            
            if caption:
                print(f"    Caption: {caption} (took {end_img_time - start_img_time:.2f}s)")
                results.append({
                    'full_image_path': img_path,
                    'product_id': product_id,
                    'generated_caption': caption
                })
            else:
                print(f"    Failed to generate caption for {os.path.basename(img_path)}")
                results.append({ # Still add a row so we know it was attempted
                    'full_image_path': img_path,
                    'product_id': product_id,
                    'generated_caption': None 
                })
        
        end_total_time = time.time()
        print(f"\nFinished caption generation for all images in {end_total_time - start_total_time:.2f} seconds.")

        df_captions = pd.DataFrame(results)
        df_captions.to_csv(OUTPUT_CAPTIONS_CSV, index=False)
        print(f"\nSaved image captions to {OUTPUT_CAPTIONS_CSV}")
        print(df_captions.head())
        print(f"\nNumber of successful captions: {df_captions['generated_caption'].notna().sum()}/{len(df_captions)}")
        
        print("\nStep 2.3: Image Caption Generation Complete.")
    else:
        print("No valid image data loaded for captioning.")