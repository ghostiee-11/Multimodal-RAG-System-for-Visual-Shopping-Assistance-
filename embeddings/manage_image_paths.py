import os
import pandas as pd

# --- Configuration ---
PRODUCTS_CSV_PATH = 'final data/products_final.csv'
IMAGE_BASE_FOLDER = '/Users/amankumar/Desktop/Aims/final data/images' 
OUTPUT_VALID_IMAGE_DATA_CSV = 'valid_product_images.csv' 

def load_product_data():
    """Loads product data."""
    try:
        df = pd.read_csv(PRODUCTS_CSV_PATH)
        df['product_id'] = df['product_id'].str.strip()
        print(f"Loaded {PRODUCTS_CSV_PATH}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Product CSV not found at {PRODUCTS_CSV_PATH}")
        return None

def consolidate_and_validate_image_paths(df_prods):
    """
    Extracts all image paths from the DataFrame, validates their existence,
    and returns a DataFrame of valid image paths with their product_ids.
    """
    if df_prods is None:
        return pd.DataFrame(columns=['product_id', 'image_filename', 'full_image_path', 'exists'])
        
    all_image_data_list = []
    processed_paths = set() 

    print("\nConsolidating and validating image paths...")
    for index, row in df_prods.iterrows():
        product_id = row['product_id']
        image_paths_str = row.get('image_paths') 

        if pd.notna(image_paths_str) and str(image_paths_str).strip():
            relative_paths = str(image_paths_str).split(',')
            for rel_path in relative_paths:
                image_filename = rel_path.strip()
                if not image_filename: 
                    continue
                
                full_image_path = os.path.join(IMAGE_BASE_FOLDER, image_filename)
                
                # Process each unique full path only once
                if full_image_path not in processed_paths:
                    image_exists = os.path.exists(full_image_path)
                    all_image_data_list.append({
                        'product_id': product_id,
                        'image_filename': image_filename, 
                        'full_image_path': full_image_path,
                        'exists': image_exists
                    })
                    processed_paths.add(full_image_path)
                    if not image_exists:
                        print(f"  Warning: Image file NOT FOUND: {full_image_path} (for product_id: {product_id})")
        else:
            print(f"  Warning: No image_paths entry for product_id: {product_id}")

    df_image_data = pd.DataFrame(all_image_data_list)
    
    # Separate valid and missing images
    df_valid_images = df_image_data[df_image_data['exists'] == True].copy()
    df_missing_images = df_image_data[df_image_data['exists'] == False].copy()

    print(f"\nTotal unique image references found in CSV: {len(df_image_data)}")
    print(f"Number of unique image files confirmed to exist: {len(df_valid_images)}")
    print(f"Number of unique image files NOT found: {len(df_missing_images)}")

    if not df_missing_images.empty:
        print("\nDetails of missing image files:")
        for _, row in df_missing_images.iterrows():
            print(f"  - Product ID: {row['product_id']}, Missing Path: {row['full_image_path']}")
    
    return df_valid_images # Return only the DataFrame of valid images

if __name__ == '__main__':
    df_products = load_product_data()
    
    if df_products is not None:
        df_valid_product_images = consolidate_and_validate_image_paths(df_products)
        
        if not df_valid_product_images.empty:
            print(f"\nSuccessfully created a list of {len(df_valid_product_images)} valid product images.")
            print("Sample of valid image data:")
            print(df_valid_product_images.head())
            
            # Save this DataFrame for future use (e.g., as input to image embedding)
            df_valid_product_images.to_csv(OUTPUT_VALID_IMAGE_DATA_CSV, index=False)
            print(f"\nSaved the list of valid product images to: {OUTPUT_VALID_IMAGE_DATA_CSV}")
            
            
        else:
            print("\nNo valid image paths found or df_products was empty.")