import pandas as pd
import time
import os
import re
import uuid
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

# --- Configuration ---
# Option 1: Provide a list of URLs directly (Commented out to prioritize CSV)
# PRODUCT_URLS = [
#     "https://www.amazon.in/Rockerz-425-Bluetooth-Headphones-Signature/dp/B09QL3NQHX",
#     "https://www.amazon.in/Boat-Rockerz-450-Wireless-Bluetooth-Headphone/dp/B07PR1CL3S",
#     # Add more product URLs here
# ]

# Option 2: Read from your all_products.csv
INPUT_CSV_PATH = "all_products.csv"
PRODUCT_URL_COLUMN = "product_url" # Column name for URLs in your CSV
PRODUCT_ID_COLUMN = "product_id"   # Column name for Product IDs in your CSV

OUTPUT_REVIEWS_CSV = "customer_reviews_scraped_v3.csv"
OUTPUT_ASPECTS_CSV = "customer_say_aspects_scraped_v3.csv"
OUTPUT_DIR_BASE = "customer_data_output_v3"
HEADLESS_MODE = False # Set to True for production, False for debugging
MAX_INDIVIDUAL_REVIEWS_PER_PRODUCT = 15
MAX_ASPECT_SNIPPETS_TO_SAVE = 3
# --- End Configuration ---

DEBUG_HTML_DIR = os.path.join(OUTPUT_DIR_BASE, "debug_html")
os.makedirs(DEBUG_HTML_DIR, exist_ok=True)

HEADERS_FOR_SELENIUM = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Accept-Language': 'en-IN,en;q=0.9',
}

def get_driver(headless=False):
    print(f"Initializing WebDriver (Headless: {headless})...")
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox"); chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu"); chrome_options.add_argument(f"user-agent={HEADERS_FOR_SELENIUM['User-Agent']}")
    chrome_options.add_argument("window-size=1920,1080")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        if not headless: driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        print("WebDriver initialized successfully.")
        return driver
    except Exception as e: print(f"Error initializing WebDriver: {e}"); return None

def save_debug_html(driver, filename_prefix="debug_page"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_prefix = re.sub(r'[<>:"/\\|?*]', '_', filename_prefix)
    filepath = os.path.join(DEBUG_HTML_DIR, f"{safe_prefix}_{timestamp}.html")
    try:
        with open(filepath, "w", encoding="utf-8") as f: f.write(driver.page_source)
        print(f"Saved debug HTML to: {filepath}")
    except Exception as e: print(f"Error saving debug HTML '{filepath}': {e}")

def get_page_source_with_selenium(driver, url, wait_for_selector=None, scroll_pause_time=1.0, scroll_attempts=4):
    try:
        print(f"Navigating to: {url}")
        driver.get(url)
        primary_wait_selector = wait_for_selector if wait_for_selector else "#customerReviews, #reviewsMedley, #cr-insights-widget-aspects, body"
        WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, primary_wait_selector))
        )
        time.sleep(1.0)
        print("Scrolling to load dynamic content...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        for attempt in range(scroll_attempts):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time + (attempt * 0.3))
            new_height = driver.execute_script("return document.body.scrollHeight")
            print(f"  Scroll attempt {attempt+1}, new height: {new_height}, last height: {last_height}")
            if new_height == last_height and attempt > 1:
                print("  Scroll height stabilized.")
                break
            last_height = new_height
            if attempt == scroll_attempts -1: print("  Completed all scroll attempts.")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
        return driver.page_source
    except TimeoutException:
        print(f"Timeout waiting for page/element at {url}.")
        save_debug_html(driver, f"timeout_URL_{url.split('/dp/')[1].split('/')[0] if '/dp/' in url else 'page'}")
        return None
    except Exception as e:
        print(f"Error fetching {url} with Selenium: {e}")
        save_debug_html(driver, f"error_URL_{url.split('/dp/')[1].split('/')[0] if '/dp/' in url else 'page'}")
        return None

def get_text_from_element(element, selector_list, default_val="N/A"):
    if not element: return default_val
    for selector in selector_list:
        found_el = element.select_one(selector)
        if found_el:
            all_texts = [text_node.strip() for text_node in found_el.find_all(string=True, recursive=True) if text_node.strip()]
            text = " ".join(all_texts)
            if text: return text
    return default_val

def extract_review_text_with_expander(review_element_soup):
    data_str = ""
    expander_content_class = "a-expander-content reviewText review-text-content a-expander-partial-collapse-content"
    for item in review_element_soup.find_all("div", class_=expander_content_class.replace("\\\n    ", " ").strip()):
        data_str = data_str + item.get_text(separator=" ", strip=True)
    if not data_str:
        review_body_hook = review_element_soup.select_one('span[data-hook="review-body"]')
        if review_body_hook:
            all_texts = [text_node.strip() for text_node in review_body_hook.find_all(string=True, recursive=True) if text_node.strip()]
            data_str = " ".join(all_texts)
    return data_str.strip() if data_str.strip() else "N/A"

def extract_product_reviews(soup, product_id_from_csv):
    reviews_data = []
    reviews_section = soup.select_one("#reviewsMedley, #customerReviews")
    if not reviews_section:
        print(f"  Reviews: Primary review section container not found for {product_id_from_csv}. Searching whole page.")
        reviews_section = soup

    review_elements = reviews_section.select('div[data-hook="review"]')
    if not review_elements:
        review_elements = soup.select('div.review[data-hook="review"]')
    
    if not review_elements:
        print(f"  Reviews: No review elements (div[data-hook='review']) found for {product_id_from_csv}.")
        return []

    print(f"  Reviews: Found {len(review_elements)} potential review blocks for {product_id_from_csv}.")
    
    for i, review_el in enumerate(review_elements):
        if i >= MAX_INDIVIDUAL_REVIEWS_PER_PRODUCT:
            print(f"  Reviews: Reached MAX_INDIVIDUAL_REVIEWS_PER_PRODUCT limit ({MAX_INDIVIDUAL_REVIEWS_PER_PRODUCT}).")
            break

        review_id_on_page = review_el.get('id', f"generated_review_id_{uuid.uuid4()}")
        reviewer_name = get_text_from_element(review_el, ['span.a-profile-name'])
        review_title_selectors = [
            'a[data-hook="review-title"] span:not([class*="a-color-secondary"])',
            'a[data-hook="review-title"]'
        ]
        review_title = get_text_from_element(review_el, review_title_selectors)
        
        star_rating_element = review_el.select_one('i[data-hook="review-star-rating"] span.a-icon-alt, i[data-hook="cmps-review-star-rating"] span.a-icon-alt')
        star_rating = "N/A"
        if star_rating_element:
            star_rating_text = star_rating_element.text.strip()
            match = re.search(r'(\d+(\.\d)?)\s*out of 5 stars', star_rating_text)
            if match: star_rating = match.group(1)

        review_date = get_text_from_element(review_el, ['span[data-hook="review-date"]'])
        review_body_text = extract_review_text_with_expander(review_el)
        
        verified_purchase_el = review_el.select_one('span[data-hook="avp-badge"], span.avp-badge-linkless')
        is_verified_purchase = True if verified_purchase_el and "Verified Purchase" in verified_purchase_el.get_text(strip=True) else False

        if review_body_text != "N/A" and len(review_body_text) > 10:
            reviews_data.append({
                'doc_id': str(uuid.uuid4()),
                'product_id': product_id_from_csv,
                'review_id_on_page': review_id_on_page,
                'reviewer_name': reviewer_name,
                'review_title': review_title,
                'star_rating': star_rating,
                'review_date': review_date,
                'review_text': review_body_text,
                'is_verified_purchase': is_verified_purchase,
                'doc_type': 'review'
            })
    return reviews_data

def extract_customer_say_aspects(soup, product_id):
    aspects_data = []
    aspects_widget = soup.select_one('div[data-hook="cr-insights-widget-aspects"]')

    if not aspects_widget:
        print(f"  Aspects: 'Customer Say' widget (div[data-hook='cr-insights-widget-aspects']) not found for {product_id}.")
        return []

    aspect_buttons = aspects_widget.select('a[data-hook="cr-insights-aspect-link"]')
    if not aspect_buttons:
        print(f"  Aspects: No aspect links (a[data-hook='cr-insights-aspect-link']) found for {product_id}.")
        return []
        
    print(f"  Aspects: Found {len(aspect_buttons)} aspect buttons for {product_id}.")

    detail_panes = aspects_widget.select('div._Y3Itc_insight-bottom-sheet-content_3cnVT[data-aspect]')
    aspect_details_map = {pane['data-aspect']: pane for pane in detail_panes if pane.has_attr('data-aspect')}

    for aspect_button in aspect_buttons:
        aspect_name_raw = aspect_button.get_text(strip=True)
        aspect_name = re.sub(r'\s*\(\d+\)$', '', aspect_name_raw).strip()

        aria_label = aspect_button.get('aria-label', '')
        overall_sentiment = "N/A"
        if "Positive aspect" in aria_label: overall_sentiment = "Positive"
        elif "Negative aspect" in aria_label: overall_sentiment = "Negative"
        elif "Mixed aspect" in aria_label: overall_sentiment = "Mixed"
        
        if overall_sentiment == "N/A":
            csa_item_id = aspect_button.get('data-csa-c-item-id', '')
            if csa_item_id:
                parts = csa_item_id.split('_')
                if len(parts) > 1:
                    sentiment_from_id = parts[-1].capitalize()
                    if sentiment_from_id in ["Positive", "Negative", "Mixed"]:
                        overall_sentiment = sentiment_from_id

        aspect_data = {
            'doc_id': str(uuid.uuid4()),
            'product_id': product_id,
            'aspect_name': aspect_name,
            'overall_sentiment': overall_sentiment,
            'mentions_total': "N/A",
            'mentions_positive': "N/A",
            'mentions_negative': "N/A",
            'summary_text': "N/A",
            'doc_type': 'aspect_summary'
        }

        detail_pane = aspect_details_map.get(aspect_name)

        if detail_pane:
            count_container = detail_pane.select_one('._Y3Itc_stat-info-box_1akN2')
            if count_container:
                total_mention_el = count_container.select_one('span.a-color-base')
                if total_mention_el:
                    total_text = total_mention_el.get_text(strip=True)
                    match = re.search(r'(\d+)\s*customer', total_text, re.IGNORECASE)
                    if match: aspect_data['mentions_total'] = int(match.group(1))

                positive_mention_el = count_container.select_one('span._Y3Itc_text-positive_QRaJ2')
                if positive_mention_el:
                    positive_text = positive_mention_el.get_text(strip=True)
                    match = re.search(r'(\d+)\s*positive', positive_text, re.IGNORECASE)
                    if match: aspect_data['mentions_positive'] = int(match.group(1))
                
                negative_mention_el = count_container.select_one('span._Y3Itc_text-negative_zjq0Y')
                if negative_mention_el:
                    negative_text = negative_mention_el.get_text(strip=True)
                    match = re.search(r'(\d+)\s*negative', negative_text, re.IGNORECASE)
                    if match: aspect_data['mentions_negative'] = int(match.group(1))

            summary_el = detail_pane.select_one('._Y3Itc_aspect-summary-label_19a1a p.a-spacing-small')
            if summary_el:
                aspect_data['summary_text'] = summary_el.get_text(separator=" ", strip=True)

            snippet_elements = detail_pane.select('div._Y3Itc_snippet_2SpLd')
            for i, snippet_el in enumerate(snippet_elements):
                if i >= MAX_ASPECT_SNIPPETS_TO_SAVE: break
                p_tag = snippet_el.select_one('p')
                if p_tag:
                    full_snippet_text = p_tag.get_text(separator=" ", strip=True)
                    full_snippet_text = re.sub(r'\s*Read more$', '', full_snippet_text, flags=re.IGNORECASE).strip()
                    highlighted_phrase = "N/A"
                    b_tag = p_tag.select_one('b')
                    if b_tag:
                        highlighted_phrase = b_tag.get_text(strip=True)
                    aspect_data[f'snippet_{i+1}_text'] = full_snippet_text
                    aspect_data[f'snippet_{i+1}_highlight'] = highlighted_phrase
        aspects_data.append(aspect_data)
    return aspects_data

# --- Main Logic ---
if __name__ == "__main__":
    product_list = []
    # Check if INPUT_CSV_PATH is defined and not empty
    if 'INPUT_CSV_PATH' in globals() and INPUT_CSV_PATH:
        try:
            input_df = pd.read_csv(INPUT_CSV_PATH)
            if PRODUCT_ID_COLUMN not in input_df.columns or PRODUCT_URL_COLUMN not in input_df.columns:
                print(f"ERROR: CSV file '{INPUT_CSV_PATH}' must contain '{PRODUCT_ID_COLUMN}' and '{PRODUCT_URL_COLUMN}' columns.")
                exit()
            for _, row in input_df.iterrows():
                # Ensure values are not NaN before stripping, convert to string first
                product_id_val = str(row[PRODUCT_ID_COLUMN]) if pd.notna(row[PRODUCT_ID_COLUMN]) else ''
                product_url_val = str(row[PRODUCT_URL_COLUMN]) if pd.notna(row[PRODUCT_URL_COLUMN]) else ''
                
                if not product_id_val.strip() or not product_url_val.strip():
                    print(f"Warning: Skipping row due to missing product_id or product_url: {row.to_dict()}")
                    continue

                product_list.append({
                    'product_id': product_id_val.strip(),
                    'product_url': product_url_val.strip()
                })
            print(f"Loaded {len(product_list)} products from '{INPUT_CSV_PATH}'")
        except FileNotFoundError:
            print(f"ERROR: Input CSV file not found at '{INPUT_CSV_PATH}'")
            exit()
        except Exception as e:
            print(f"Error reading CSV '{INPUT_CSV_PATH}': {e}")
            exit()
    # Fallback to PRODUCT_URLS list if CSV path is not defined or fails (though exit() is called on failure)
    # And if PRODUCT_URLS is defined and not empty
    elif 'PRODUCT_URLS' in globals() and PRODUCT_URLS:
        for url in PRODUCT_URLS:
            asin_match = re.search(r'/dp/([A-Z0-9]{10})', url)
            if not asin_match:
                 asin_match = re.search(r'/product-reviews/([A-Z0-9]{10})', url)
            pid = asin_match.group(1) if asin_match else f"pid_from_url_{str(uuid.uuid4())[:8]}"
            product_list.append({'product_id': pid, 'product_url': url})
        print(f"Using {len(product_list)} URLs from the PRODUCT_URLS list (CSV not used or failed).")
    else:
        print("No product data source specified. Provide PRODUCT_URLS list or INPUT_CSV_PATH.")
        exit()

    if not product_list:
        print("No products to process. Exiting.")
        exit()

    driver = get_driver(headless=HEADLESS_MODE)
    if not driver: print("Failed to initialize Selenium WebDriver. Exiting."); exit()

    all_extracted_reviews = []
    all_extracted_aspects = []
    processed_product_count = 0
    total_products_to_process = len(product_list)
    current_product_id_for_error = "unknown"

    try:
        for item_info in product_list:
            product_id = item_info['product_id'] # This now comes directly from the CSV
            current_product_id_for_error = product_id
            product_url = item_info['product_url']

            if not product_url.startswith("http"):
                print(f"Skipping invalid URL for ID {product_id}: {product_url}"); continue
            
            if '/product-reviews/' in product_url:
                main_product_url = product_url.replace('/product-reviews/', '/dp/')
                print(f"  URL is for reviews, attempting main product page for aspects: {main_product_url}")
            else:
                main_product_url = product_url
            
            print(f"\nProcessing product {processed_product_count + 1}/{total_products_to_process}: ID {product_id} (URL: {main_product_url})")
            
            page_source = get_page_source_with_selenium(driver, main_product_url, 
                                                        wait_for_selector="#reviewsMedley, #customerReviews, #cr-insights-widget-aspects", 
                                                        scroll_attempts=6, scroll_pause_time=1.8)

            if page_source:
                soup = BeautifulSoup(page_source, 'html.parser')
                
                reviews_for_product = extract_product_reviews(soup, product_id)
                if reviews_for_product:
                    all_extracted_reviews.extend(reviews_for_product)
                    print(f"  SUCCESS (Reviews): Extracted {len(reviews_for_product)} reviews for {product_id}.")
                else:
                    print(f"  INFO (Reviews): No reviews found/extracted for {product_id} using defined selectors.")

                aspects_for_product = extract_customer_say_aspects(soup, product_id)
                if aspects_for_product:
                    all_extracted_aspects.extend(aspects_for_product)
                    print(f"  SUCCESS (Aspects): Extracted {len(aspects_for_product)} aspects for {product_id}.")
                else:
                    print(f"  INFO (Aspects): No 'Customer Say' aspects found/extracted for {product_id}.")
                
                if not reviews_for_product and not aspects_for_product:
                    save_debug_html(driver, f"no_data_found_pid_{product_id}")
            else:
                print(f"  WARNING: Failed to get page source for {product_id}.")
            
            processed_product_count += 1
            if processed_product_count < total_products_to_process :
                delay = max(3.0, 3.0 + (uuid.uuid4().int % 4000 / 1000.0)) 
                print(f"--- Delaying for {delay:.2f}s before next product ---")
                time.sleep(delay)

    except KeyboardInterrupt: print("\nScraping interrupted by user. Saving collected data...")
    except Exception as e:
        print(f"\nAn UNEXPECTED ERROR occurred while processing product ID '{current_product_id_for_error}': {e}")
        import traceback
        traceback.print_exc()
        if driver: save_debug_html(driver, f"main_loop_error_pid_{current_product_id_for_error}")
    finally:
        if driver: driver.quit(); print("WebDriver closed.")

    if all_extracted_reviews:
        reviews_output_df = pd.DataFrame(all_extracted_reviews)
        reviews_output_df.to_csv(OUTPUT_REVIEWS_CSV, index=False, encoding='utf-8-sig')
        print(f"\nSaved {len(reviews_output_df)} review entries to '{OUTPUT_REVIEWS_CSV}'")
        if not reviews_output_df.empty:
             print(f"Review data for {reviews_output_df['product_id'].nunique()} unique products.")
    else: print("\nNo customer reviews collected.")

    if all_extracted_aspects:
        aspects_output_df = pd.DataFrame(all_extracted_aspects)
        cols_order = ['doc_id', 'product_id', 'aspect_name', 'overall_sentiment', 
                      'mentions_total', 'mentions_positive', 'mentions_negative', 
                      'summary_text', 'doc_type']
        snippet_cols = sorted([col for col in aspects_output_df.columns if col.startswith('snippet_')])
        final_cols = [col for col in cols_order if col in aspects_output_df.columns] + snippet_cols
        aspects_output_df = aspects_output_df[final_cols]
        aspects_output_df.to_csv(OUTPUT_ASPECTS_CSV, index=False, encoding='utf-8-sig')
        print(f"\nSaved {len(aspects_output_df)} aspect entries to '{OUTPUT_ASPECTS_CSV}'")
        if not aspects_output_df.empty:
            print(f"Aspect data for {aspects_output_df['product_id'].nunique()} unique products.")
    else: print("\nNo 'Customer Say' aspects collected.")

    print(f"\n--- Scraping Complete ---")
    print(f"Total products attempted: {processed_product_count} out of {total_products_to_process}")