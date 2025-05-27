# prepare_text_chunks.py
import pandas as pd
import uuid
from data_loader import load_and_clean_data 

def create_retrievable_text_df(df_products, df_reviews, df_alldocs):
    """
    Creates a DataFrame of individual text chunks for embedding,
    based on the loaded product, review, and all_documents data.
    """
    if df_products is None or df_reviews is None or df_alldocs is None:
        print("One or more input DataFrames are None. Cannot proceed.")
        return None

    retrievable_texts_list = []

    # 1. Process Customer Reviews (df_reviews)
    print("\nProcessing customer reviews for text chunking...")
    for index, row in df_reviews.iterrows():
        product_id = row['product_id']
        original_doc_id = row['doc_id']
        aspect_name = row['aspect_name']
        overall_sentiment = row['overall_sentiment']

        if pd.notna(row['summary_text']) and str(row['summary_text']).strip():
            retrievable_texts_list.append({
                'text_chunk_id': f"chunk_{uuid.uuid4()}",
                'product_id': product_id,
                'original_doc_id': original_doc_id,
                'source_file': 'customer_reviews.csv',
                'text_type': 'aspect_summary',
                'aspect': aspect_name,
                'sentiment': overall_sentiment,
                'text_content': str(row['summary_text']).strip()
            })

        for i in range(1, 4):
            snippet_col = f'snippet_{i}_text'
            highlight_col = f'snippet_{i}_highlight'
            if pd.notna(row[snippet_col]) and str(row[snippet_col]).strip():
                retrievable_texts_list.append({
                    'text_chunk_id': f"chunk_{uuid.uuid4()}",
                    'product_id': product_id,
                    'original_doc_id': original_doc_id,
                    'source_file': 'customer_reviews.csv',
                    'text_type': 'review_snippet',
                    'aspect': aspect_name,
                    'sentiment': overall_sentiment,
                    'text_content': str(row[snippet_col]).strip(),
                    'highlight': str(row[highlight_col]) if pd.notna(row[highlight_col]) else None
                })
    print(f"Processed {len(df_reviews)} review rows for text chunking.")

    # 2. Process All Documents (df_alldocs)
    print("\nProcessing all_documents for text chunking...")
    for index, row in df_alldocs.iterrows():
        product_id = row['product_id']
        original_doc_id = row['doc_id']
        doc_type = row['doc_type']
        text_content = row['text_content']

        if pd.notna(text_content) and str(text_content).strip():
            current_aspect = 'General'
            if doc_type == 'specification':
                current_aspect = 'Specification'
            elif 'description' in doc_type:
                current_aspect = 'Description'

            retrievable_texts_list.append({
                'text_chunk_id': f"chunk_{uuid.uuid4()}",
                'product_id': product_id,
                'original_doc_id': original_doc_id,
                'source_file': 'all_documents.csv',
                'text_type': doc_type,
                'aspect': current_aspect,
                'sentiment': 'N/A',
                'text_content': str(text_content).strip()
            })
    print(f"Processed {len(df_alldocs)} all_docs rows for text chunking.")

    df_retrievable_texts = pd.DataFrame(retrievable_texts_list)
    return df_retrievable_texts

if __name__ == '__main__':
    # Define paths to your CSV files
    PRODUCTS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/products_final.csv'
    REVIEWS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/customer_reviews.csv'
    ALL_DOCS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/all_documents.csv'

    # --- Load data using the imported function ---
    df_products, df_reviews, df_alldocs = load_and_clean_data(
        PRODUCTS_CSV_PATH,
        REVIEWS_CSV_PATH,
        ALL_DOCS_CSV_PATH
    )

    if df_products is not None: # Check if data loading was successful
        # --- Create the retrievable texts DataFrame ---
        df_retrievable_texts = create_retrievable_text_df(df_products, df_reviews, df_alldocs)

        if df_retrievable_texts is not None:
            # --- Inspect the new df_retrievable_texts ---
            print("\n--- Retrievable Texts DataFrame (df_retrievable_texts) ---")
            print("Shape:", df_retrievable_texts.shape)
            print("\nHead:")
            print(df_retrievable_texts.head(10))
            print("\nInfo:")
            df_retrievable_texts.info()
            print("\nText Types Counts:")
            print(df_retrievable_texts['text_type'].value_counts())
            print("\nAspect Counts:")
            print(df_retrievable_texts['aspect'].value_counts())
            print("\nMissing values per column:\n", df_retrievable_texts.isnull().sum())
            print(f"\nTotal text chunks created: {len(df_retrievable_texts)}")

            # Verify Product ID linkage
            product_ids_in_retrievable = set(df_retrievable_texts['product_id'].unique())
            product_ids_in_master_list = set(df_products['product_id'].unique())
            missing_pids_in_products = product_ids_in_retrievable - product_ids_in_master_list
            if missing_pids_in_products:
                print(f"\nWARNING: {len(missing_pids_in_products)} product_ids found in retrievable texts but NOT in df_products.")
                # print(f"Missing IDs: {missing_pids_in_products}")
            else:
                print("\nAll product_ids in retrievable texts are present in df_products. Good!")

            # Save for inspection
            df_retrievable_texts.to_csv('prepared_text_chunks.csv', index=False)
            print("\nSaved prepared_text_chunks.csv for inspection.")
        else:
            print("Failed to create df_retrievable_texts.")
    else:
        print("Data loading failed. Cannot proceed with text chunk preparation.")