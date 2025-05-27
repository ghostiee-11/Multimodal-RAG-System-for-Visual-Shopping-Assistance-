import pandas as pd
import re

# Load the OCR CSV
df = pd.read_csv("image_ocr_texts_.csv")

# Define OCR cleaning function
def clean_ocr_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
    text = re.sub(r'[^\w\s,.%-]', '', text)  # Remove unwanted special characters except some punctuation
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Optionally remove single/double letter noise
    text = text.strip()
    return text

# Apply cleaner
df["ocr_text_cleaned"] = df["ocr_text"].apply(clean_ocr_text)

# Save cleaned CSV
df.to_csv("image_ocr_texts_cleaned.csv", index=False)
print("Cleaned OCR text saved to image_ocr_texts_cleaned.csv")
