import os
import pandas as pd
import pytesseract
import cv2
from PIL import Image

# Paths
csv_path = "/Users/amankumar/Desktop/Aims/final data/products_final_with_all_image_paths.csv"
image_base_path = "/Users/amankumar/Desktop/Aims/final data/images_all_scraped"

# Load CSV
df = pd.read_csv(csv_path)

# Prepare results
results = []

# Iterate through the DataFrame
for idx, row in df.iterrows():
    product_id = row['product_id']
    image_paths = str(row['image_paths']).split(',')  # Assumes multiple image paths are comma-separated

    for rel_path in image_paths:
        rel_path = rel_path.strip()
        full_image_path = os.path.join(image_base_path, rel_path)

        if os.path.exists(full_image_path):
            try:
                # Load and preprocess image
                image = cv2.imread(full_image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
                _, thresh = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY_INV)
                inverted = cv2.bitwise_not(thresh)
                pil_img = Image.fromarray(inverted)

                # OCR
                ocr_text = pytesseract.image_to_string(pil_img).strip()

                # Print in terminal
                print("="*80)
                print(f"Product ID: {product_id}")
                print(f"Image Path: {full_image_path}")
                print("OCR Text:\n", ocr_text)
                print("="*80)

                # Store result
                results.append({
                    "product_id": product_id,
                    "image_path": full_image_path,
                    "ocr_text": ocr_text
                })

            except Exception as e:
                print(f"Error processing {full_image_path}: {e}")
        else:
            print(f"Image not found: {full_image_path}")

# Save to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("image_ocr_texts_.csv", index=False)
print("✅ OCR extraction complete. Data saved to image_ocr_texts_.csv")
