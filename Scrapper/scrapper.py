import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import re
import json
import uuid
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# --- Configuration ---
BASE_URL = "https://www.amazon.in"
INITIAL_SEARCH_URL = "https://www.amazon.in/s?k=headphones&page=12&qid=1747955309&xpid=GPlTvG6ozhsd3&ref=sr_pg_12" # Example search for wireless headphones on .in
# You can replace INITIAL_SEARCH_URL with a more specific one like:
# INITIAL_SEARCH_URL = "https://www.amazon.in/s?k=headphones&crid=15BTZ22NNVGER&sprefix=headphone%2Caps%2C219&ref=nb_sb_noss_2"

MAX_PRODUCTS = 60  # START WITH A SMALL NUMBER (e.g., 2-5) FOR TESTING! Then increase to ~50.
OUTPUT_DIR_BASE = "final_data4"
# --- End Configuration ---

# Dynamic output directory based on BASE_URL
domain_name = BASE_URL.split('.')[-2] if '.' in BASE_URL else "unknown_site"
OUTPUT_DIR = f"{OUTPUT_DIR_BASE}_{domain_name}"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
RAW_PRODUCTS_CSV = os.path.join(OUTPUT_DIR, "products_raw.csv")
RAW_DOCUMENTS_CSV = os.path.join(OUTPUT_DIR, "documents_raw.csv")
DEBUG_HTML_DIR = os.path.join(OUTPUT_DIR, "debug_html")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DEBUG_HTML_DIR, exist_ok=True)

HEADERS_FOR_IMAGES = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36', # Update if needed
    'Accept-Language': 'en-IN,en;q=0.9' if "amazon.in" in BASE_URL else 'en-US,en;q=0.9',
}

# --- Selenium WebDriver Setup ---
def get_driver(headless=False): # Set headless=True for production, False for debugging
    """Initializes and returns a Selenium WebDriver."""
    print(f"Initializing WebDriver (Headless: {headless})...")
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(f"user-agent={HEADERS_FOR_IMAGES['User-Agent']}")
    chrome_options.add_argument("window-size=1920,1080")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    # Optional: To potentially bypass some bot detections (use with caution)
    # chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    # chrome_options.add_experimental_option("useAutomationExtension", False)

    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        # Optional:
        # if not headless:
        #     driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        print("WebDriver initialized successfully.")
        return driver
    except Exception as e:
        print(f"Error initializing WebDriver: {e}")
        print("Ensure Chrome is installed and webdriver_manager can download ChromeDriver.")
        return None

def save_debug_html(driver, filename_prefix="debug_page"):
    """Saves the current page source for debugging."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(DEBUG_HTML_DIR, f"{filename_prefix}_{timestamp}.html")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print(f"Saved debug HTML to: {filepath}")
    except Exception as e:
        print(f"Error saving debug HTML: {e}")


# --- Helper Functions ---
def get_page_source_with_selenium(driver, url, wait_for_selector=None, scroll_pause_time=2, scroll_times=3):
    """Fetches page source using Selenium, with scrolling and optional element wait."""
    try:
        print(f"Fetching with Selenium: {url}")
        driver.get(url)
        if wait_for_selector:
            print(f"Waiting for selector: {wait_for_selector}")
            WebDriverWait(driver, 20).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, wait_for_selector))
            )
            print("Selector found.")
        else: # General wait if no specific selector
            WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(2) # Initial settle time

        if scroll_times > 0:
            print(f"Scrolling page {scroll_times} time(s)...")
            for i in range(scroll_times):
                driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight * {(i+1)/scroll_times});")
                time.sleep(scroll_pause_time)
        return driver.page_source
    except TimeoutException:
        print(f"Timed out waiting for page/element at {url}.")
        save_debug_html(driver, f"timeout_{url.split('/')[-1] if '/' in url else 'page'}")
        return None
    except Exception as e:
        print(f"Error fetching {url} with Selenium: {e}")
        save_debug_html(driver, f"error_{url.split('/')[-1] if '/' in url else 'page'}")
        return None

def extract_product_urls_from_search(driver, search_url, limit=10):
    print(f"Extracting product URLs from search: {search_url}")
    # Common selectors for search result items on Amazon
    search_item_selectors = [
        'div[data-component-type="s-search-result"]',
        'div.s-result-item[data-asin]',
    ]
    page_source = get_page_source_with_selenium(driver, search_url, wait_for_selector=search_item_selectors[0], scroll_times=1)
    if not page_source: return []

    soup = BeautifulSoup(page_source, 'html.parser')
    product_links = set() # Use set to avoid duplicates initially

    for item_selector in search_item_selectors:
        search_results_divs = soup.select(item_selector)
        print(f"Using selector '{item_selector}', found {len(search_results_divs)} potential items.")
        for result_div in search_results_divs:
            # Selectors for the link within the item
            link_selectors = [
                'h2 a.a-link-normal.s-underline-text.s-underline-link-text.s-link-style.a-text-normal', # .com style
                'h2 a.a-link-normal.a-text-normal', # .in style often
                'a.a-link-normal[href*="/dp/"]' # More generic link to product page
            ]
            link_element = None
            for link_sel in link_selectors:
                link_element = result_div.select_one(link_sel)
                if link_element: break
            
            if link_element and link_element.get('href'):
                href = link_element.get('href')
                if not href.startswith('http'): href = BASE_URL + href
                
                # Filter out sponsored, redirects, and non-product page links
                if '/dp/' in href and 'slredirect' not in href and '/sspa/' not in href and 'customerReviews' not in href:
                    clean_href = href.split('/ref=')[0].split('?')[0]
                    product_links.add(clean_href)
                    if len(product_links) >= limit: break
        if len(product_links) >= limit: break
        if product_links: break # If one item selector worked, no need to try others for this page

    print(f"Found {len(product_links)} unique product URLs.")
    if not product_links and page_source:
        print("No product URLs extracted. Saving search page HTML for debugging.")
        save_debug_html(driver, "search_no_links")
    return list(product_links)[:limit]


def download_image(image_url, product_id, img_index):
    if not image_url or not image_url.startswith('http'):
        print(f"Skipping invalid or non-absolute image URL: {image_url}")
        return None
    try:
        img_response = requests.get(image_url, headers=HEADERS_FOR_IMAGES, stream=True, timeout=15)
        img_response.raise_for_status()
        content_type = img_response.headers.get('content-type', '').lower()
        extension = 'jpg'
        if 'jpeg' in content_type: extension = 'jpg'
        elif 'png' in content_type: extension = 'png'
        elif 'webp' in content_type: extension = 'webp'
        
        filename = f"{product_id}_img{img_index}.{extension}"
        filepath = os.path.join(IMAGES_DIR, filename)
        with open(filepath, 'wb') as f:
            for chunk in img_response.iter_content(chunk_size=8192): f.write(chunk)
        print(f"Downloaded image: {filename}")
        time.sleep(0.5)
        return filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image {image_url}: {e}")
        return None

def get_text_from_soup(soup_element, selector_list, default_val="N/A"):
    """Helper to try multiple selectors and get stripped text."""
    if not soup_element: return default_val
    for selector in selector_list:
        element = soup_element.select_one(selector)
        if element and element.text.strip():
            return element.text.strip()
    return default_val
    
def extract_product_data(driver, product_url):
    print(f"Scraping product page: {product_url}")
    page_source = get_page_source_with_selenium(driver, product_url, wait_for_selector="#productTitle, #centerCol", scroll_times=3)
    if not page_source:
        save_debug_html(driver, f"product_page_load_failed_{product_url.split('/dp/')[1].split('/')[0] if '/dp/' in product_url else 'unknown_product'}")
        return None, []

    soup = BeautifulSoup(page_source, 'html.parser')
    product_data = {}
    textual_docs = []

    # 1. Product ID (ASIN)
    pid = None
    pid_match = re.search(r'/dp/([A-Z0-9]{10})', product_url)
    if pid_match: pid = pid_match.group(1)
    
    if not pid: # Try finding from page elements if not in URL (should be rare)
        asin_selectors_th = [
            '#detailBullets_productDetails_teaser_feature_div th:contains("ASIN")',
            '#productDetails_detailBullets_sections1 th:contains("ASIN")',
            '#productDetailsTable th:contains("ASIN")'
        ]
        for sel in asin_selectors_th:
            th_element = soup.select_one(sel)
            if th_element and th_element.find_next_sibling('td'):
                pid = th_element.find_next_sibling('td').text.strip()
                break
        if not pid: pid = soup.select_one('input#ASIN')['value'] if soup.select_one('input#ASIN') else None

    if not pid:
        print(f"CRITICAL: Could not extract ASIN for {product_url}. Skipping product.")
        save_debug_html(driver, f"product_no_asin_{product_url.replace('/', '_')}")
        return None, []
    product_data['product_id'] = pid
    product_data['product_url'] = product_url

    # 2. Title
    product_data['title'] = get_text_from_soup(soup, ['span#productTitle'])

    # 3. Price
    price_text = get_text_from_soup(soup, [
        '#corePrice_feature_div span.a-price .a-offscreen', # Common
        '#corePriceDisplay_desktop_feature_div span.a-price .a-offscreen',
        'span.priceToPay .a-offscreen',
        '#priceblock_ourprice', '#priceblock_dealprice',
        'span.a-price[data-a-size="xl"] span.a-offscreen', # another variant
        'div[data-feature-name="priceBlock"] span.a-price.a-text-price span.a-offscreen' # More specific path
    ])
    product_data['price'] = price_text if price_text != "N/A" else get_text_from_soup(soup, ['span.a-price-whole']) # Fallback to whole part if symbol missing

    # 4. Category (Breadcrumbs)
    breadcrumb_lis = soup.select('#wayfinding-breadcrumbs_feature_div ul li span.a-list-item a, #nav-subnav a.nav-a')
    product_data['category_scraped'] = " > ".join([b.text.strip() for b in breadcrumb_lis if b.text.strip()]) if breadcrumb_lis else "Unknown"

    # 5. Specifications (Feature Bullets)
    spec_list_items = soup.select('#feature-bullets ul.a-unordered-list li span.a-list-item, div#detailBullets_feature_div ul li span.a-list-item')
    for spec_item in spec_list_items:
        spec_text = spec_item.text.strip()
        if spec_text and "customer review" not in spec_text.lower() and "ratings" not in spec_text.lower():
            textual_docs.append({'doc_id': str(uuid.uuid4()), 'product_id': pid, 'doc_type': 'specification', 'text_content': spec_text})

    # 6. Product Description
    desc_div = soup.select_one('#productDescription, #aplus_feature_div, #aplus') # Try multiple description containers
    if desc_div:
        desc_paragraphs = desc_div.find_all('p', recursive=False) # Direct children paragraphs
        if desc_paragraphs:
            for p in desc_paragraphs:
                p_text = p.text.strip()
                if p_text and len(p_text) > 15: textual_docs.append({'doc_id': str(uuid.uuid4()), 'product_id': pid, 'doc_type': 'description_paragraph', 'text_content': p_text})
        else: # Fallback to all text in the div if no direct P tags
            full_desc = desc_div.get_text(separator='\n', strip=True)
            if full_desc and len(full_desc) > 20: textual_docs.append({'doc_id': str(uuid.uuid4()), 'product_id': pid, 'doc_type': 'description_full', 'text_content': full_desc})

    # 7. User Reviews (Get top few)
    review_elements = soup.select('div[data-hook="review"]')
    for i, review_el in enumerate(review_elements):
        if i >= 5: break # Max 5 reviews
        review_body = get_text_from_soup(review_el, ['span[data-hook="review-body"] span', 'span[data-hook="review-body"]'])
        if review_body != "N/A":
            textual_docs.append({'doc_id': str(uuid.uuid4()), 'product_id': pid, 'doc_type': 'review', 'text_content': review_body})

    # 8. Q&A Threads (Get top few)
    # Q&A is notoriously hard to scrape consistently. These are common patterns.
    qa_items = soup.select('#askDPSearchResults div.a-fixed-left-grid.a-spacing-base, #ask_lazy_load_div div.a-spacing-base, div.askTeaserQuestions > div.askTeaserQuestion')
    for i, item in enumerate(qa_items):
        if i >= 3: break # Max 3 Q&A
        q_text = get_text_from_soup(item, ['a.a-link-normal span.askQuestionText', 'div.a-col-right a span', 'span.askQuestionText > a'])
        a_text = get_text_from_soup(item, ['div.a-col-right span.a-truncate-full', 'span.askAnswersText span', 'div.a-col-right > div > span:not([class*="expander"])']) # Trying various answer text selectors
        if q_text != "N/A" and a_text != "N/A":
            textual_docs.append({'doc_id': str(uuid.uuid4()), 'product_id': pid, 'doc_type': 'qa_answer', 'text_content': f"Q: {q_text}\nA: {a_text}"})
    
    # 9. Main Image(s)
    image_filenames = []
    main_image_tag = soup.select_one('#landingImage, #imgTagWrapperId img, #ivLargeImage img')
    if main_image_tag:
        img_url = None
        # Try attributes in order of preference for higher quality
        if main_image_tag.get('data-old-hires'): img_url = main_image_tag.get('data-old-hires')
        elif main_image_tag.get('data-a-dynamic-image'):
            try:
                dynamic_images = json.loads(main_image_tag.get('data-a-dynamic-image'))
                if dynamic_images: img_url = list(dynamic_images.keys())[0] # Simplistic: take first (often largest)
            except: pass
        if not img_url and main_image_tag.get('src') and 'data:image' not in main_image_tag.get('src'):
             img_url = main_image_tag.get('src')

        if img_url:
            filename = download_image(img_url, pid, 1)
            if filename: image_filenames.append(filename)

    # Try for a second image from thumbnails if first one was successful and we need more
    if image_filenames and len(image_filenames) < 2:
        thumb_elements = soup.select('#altImages li.imageThumbnail img, #imageBlockThumbsทุนล่างทุนบน li.a-spacing-small img') # Common thumbnail lists
        for thumb_img_tag in thumb_elements:
            thumb_url = None
            if thumb_img_tag.get('src') and 'data:image' not in thumb_img_tag.get('src'):
                # Thumbnails often have low-res src; high-res is sometimes in parent or complex JS.
                # This is a simplification: assuming src might be good enough or a placeholder for JS.
                # For truly high-res from thumbs, more complex interaction (clicks) or JS parsing is needed.
                thumb_url = thumb_img_tag.get('src').replace('_AC_US40_', '_AC_SL1500_').replace('_SX38_SY50_CR,0,0,38,50_','').replace('_SX35_SY45_QL70_','_SL1500_') # Heuristic to get larger version
            if thumb_url:
                # Avoid downloading the same image again if its base name matches the first
                if not any(thumb_url.split('/')[-1].split('.')[0] == f.split('/')[-1].split('.')[0] for f in image_filenames):
                    filename_thumb = download_image(thumb_url, pid, len(image_filenames) + 1)
                    if filename_thumb:
                        image_filenames.append(filename_thumb)
                        if len(image_filenames) >= 2: break
    product_data['image_paths'] = ",".join(image_filenames) if image_filenames else None
    if not product_data.get('image_paths'): print(f"No images found/downloaded for {pid}")

    print(f"Extracted {len(textual_docs)} text docs for {pid}. Title: {product_data.get('title', 'N/A')}")
    return product_data, textual_docs

# --- Main Scraping Logic ---
if __name__ == "__main__":
    # Start with headless=False for debugging selectors
    # Once confident, switch to headless=True for faster, non-UI runs
    driver = get_driver(headless=False) # <<<< SET TO True FOR "PRODUCTION" RUNS
    if not driver:
        print("Failed to initialize Selenium WebDriver. Exiting.")
        exit()

    all_products_metadata = []
    all_textual_documents = []
    scraped_product_ids = set()

    try:
        print(f"Starting scrape. Target: {INITIAL_SEARCH_URL}")
        print(f"Will attempt to get up to {MAX_PRODUCTS} products.")

        product_urls_to_scrape = extract_product_urls_from_search(driver, INITIAL_SEARCH_URL, limit=MAX_PRODUCTS + 5) # Get a few extra

        if not product_urls_to_scrape:
            print("No product URLs found from search page. Exiting.")
        else:
            print(f"\n--- Found {len(product_urls_to_scrape)} URLs, will process up to {MAX_PRODUCTS} unique products ---")
            for i, url in enumerate(product_urls_to_scrape):
                if len(scraped_product_ids) >= MAX_PRODUCTS:
                    print(f"Reached MAX_PRODUCTS limit ({MAX_PRODUCTS}). Stopping URL processing.")
                    break

                print(f"\nProcessing URL {i+1}/{len(product_urls_to_scrape)}: {url}")
                # Simple ASIN check from URL to avoid re-processing known ASINs too early
                temp_asin_match = re.search(r'/dp/([A-Z0-9]{10})', url)
                if temp_asin_match and temp_asin_match.group(1) in scraped_product_ids:
                    print(f"Skipping URL for already processed ASIN: {temp_asin_match.group(1)}")
                    continue

                product_meta, product_texts = extract_product_data(driver, url)

                if product_meta and product_meta.get('product_id'):
                    pid_scraped = product_meta['product_id']
                    if pid_scraped not in scraped_product_ids:
                        # Basic check: Ensure we got a title and at least one image or some text docs
                        if (product_meta.get('title') and product_meta.get('title') != "N/A") and \
                           (product_meta.get('image_paths') or len(product_texts) > 0):
                            all_products_metadata.append(product_meta)
                            all_textual_documents.extend(product_texts)
                            scraped_product_ids.add(pid_scraped)
                            print(f"SUCCESS: Scraped product ID {pid_scraped}. Total unique products: {len(scraped_product_ids)}")
                        else:
                            print(f"WARNING: Scraped product ID {pid_scraped} but essential data (title/image/text) missing. Skipping.")
                            save_debug_html(driver, f"product_missing_data_{pid_scraped}")
                    else:
                        print(f"Skipping duplicate ASIN after full scrape: {pid_scraped}")
                else:
                    print(f"Failed to scrape essential data (or ASIN) for URL: {url}")
                    save_debug_html(driver, f"product_scrape_failed_{url.replace('/', '_')}")

                # Crucial delay to avoid getting blocked
                delay_seconds = max(5, 12 - (len(scraped_product_ids) % 4)) # e.g. 12, 11, 10, 9, then 12...
                print(f"--- Delaying for {delay_seconds} seconds before next product ---")
                time.sleep(delay_seconds)

    except Exception as e:
        print(f"An UNEXPECTED ERROR occurred during the main scraping process: {e}")
        import traceback
        traceback.print_exc()
        if driver: save_debug_html(driver, "main_process_error")
    finally:
        if driver:
            driver.quit()
            print("WebDriver closed.")

    # --- Save to CSV ---
    if all_products_metadata:
        products_df = pd.DataFrame(all_products_metadata)
        products_df.to_csv(RAW_PRODUCTS_CSV, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
        print(f"\nSaved {len(products_df)} product metadata entries to {RAW_PRODUCTS_CSV}")
    else:
        print("\nNo product metadata was scraped successfully.")

    if all_textual_documents:
        documents_df = pd.DataFrame(all_textual_documents)
        documents_df.to_csv(RAW_DOCUMENTS_CSV, index=False, encoding='utf-8-sig')
        print(f"Saved {len(documents_df)} textual documents to {RAW_DOCUMENTS_CSV}")
    else:
        print("\nNo textual documents were scraped successfully.")

    print("\n--- Selenium Scraping Phase 1 Complete ---")
    print(f"Total unique products successfully scraped: {len(scraped_product_ids)}")
    print(f"Total textual documents collected: {len(all_textual_documents)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Debug HTML files (if any) saved in: {DEBUG_HTML_DIR}")