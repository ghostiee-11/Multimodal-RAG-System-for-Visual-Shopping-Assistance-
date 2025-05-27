# enhanced_image_scraper_from_urls.py
import os
import time
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
import json
import logging

# --- Configuration ---
PRODUCTS_INPUT_CSV = '/Users/amankumar/Desktop/Aims/final data/products_final.csv'
IMAGE_OUTPUT_FOLDER = '/Users/amankumar/Desktop/Aims/final data/images_all_scraped/' # Ensure this folder exists or script creates it
IMAGE_INFO_OUTPUT_CSV = 'all_product_images_info_scraped.csv'
UPDATED_PRODUCTS_CSV_OUTPUT = '/Users/amankumar/Desktop/Aims/final data/products_final_with_all_image_paths.csv'

os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)

# --- Logging Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(current_script_dir, "image_scraper.log")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

HEADERS_FOR_IMAGES = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36', # ### REPLACE WITH YOUR ACTUAL USER AGENT ###
    'Accept-Language': 'en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Referer': 'https://www.amazon.in/'
}

def setup_selenium_driver(headless=True):
    logger.info(f"Initializing WebDriver (Headless: {headless})...")
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(f"user-agent={HEADERS_FOR_IMAGES['User-Agent']}")
    chrome_options.add_argument("window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        if not headless:
             driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        logger.info("WebDriver initialized successfully.")
        return driver
    except Exception as e:
        logger.error(f"Error initializing WebDriver: {e}", exc_info=True)
        return None

def download_image_robust(image_url, save_folder, product_id, img_index):
    if not image_url or not isinstance(image_url, str) or not image_url.startswith('http'):
        logger.warning(f"    Skipping invalid image URL: {image_url}")
        return None
    try:
        base_url = re.sub(r'\._[A-Z0-9,_SXUYACVTZI]+_\.(jpg|png|jpeg|webp)', r'.\1', image_url, flags=re.IGNORECASE)
        if base_url == image_url and not re.search(r'SL\d{3,}', image_url, flags=re.IGNORECASE):
            if any(thumb_pattern in image_url for thumb_pattern in ['_US40_', '_SS40_', '_SX', '_SY', '_AC_','QL70']):
                 base_name_part = image_url.split('._')[0] 
                 extension_part_match = re.search(r'\.(jpg|png|jpeg|webp)', image_url, flags=re.IGNORECASE)
                 if extension_part_match and base_name_part:
                     extension = extension_part_match.group(0)
                     base_url = f"{base_name_part}._SL1500_{extension}"
                 else: base_url = image_url
        
        logger.debug(f"    Attempting to download from: {base_url}")
        img_response = requests.get(base_url, headers=HEADERS_FOR_IMAGES, stream=True, timeout=30)
        img_response.raise_for_status()
        content_type = img_response.headers.get('content-type', '').lower()
        extension = '.jpg' 
        if 'jpeg' in content_type: extension = '.jpg'
        elif 'png' in content_type: extension = '.png'
        elif 'webp' in content_type: extension = '.webp'
        filename = f"{product_id}_allsrp_img{img_index}{extension}"
        filepath = os.path.join(save_folder, filename)
        with open(filepath, 'wb') as f:
            for chunk in img_response.iter_content(chunk_size=8192): f.write(chunk)
        logger.info(f"      Downloaded: {filename} from {base_url}")
        time.sleep(0.3) 
        return filename
    except requests.exceptions.Timeout: logger.warning(f"    Timeout downloading {base_url if 'base_url' in locals() else image_url}")
    except requests.exceptions.RequestException as e: logger.warning(f"    Error downloading {base_url if 'base_url' in locals() else image_url}: {e}")
    except Exception as e: logger.warning(f"    Generic error saving image {base_url if 'base_url' in locals() else image_url}: {e}", exc_info=True)
    return None

def extract_all_image_urls_from_product_page(driver, product_url):
    image_urls = set()
    logger.info(f"  Scraping images from: {product_url}")
    try:
        driver.get(product_url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "dp")))
        time.sleep(4) 
    except Exception as e:
        logger.error(f"    Error loading page {product_url}: {e}", exc_info=True); return []

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    
    # Strategy 1: Script Tag Parsing (Prioritized as it was successful for one case)
    logger.debug("    Trying Strategy 1: Parsing script tags for image data...")
    scripts = soup.find_all('script', type='text/javascript') # Or just soup.find_all('script')
    found_in_script = False
    for script_tag in scripts:
        script_content = script_tag.string
        if script_content:
            if any(key_term in script_content for key_term in ['imageBlockVariations', 'colorImages', 'dpImages', 'galleryImages', 'mainImageInfo', 'imageListData']):
                logger.debug(f"    Found script potentially containing images. Keywords matched. Length: {len(script_content)}")
                
                # Regex for 'imageGalleryData' pattern
                gallery_match = re.search(r'var\s+imageGalleryData\s*=\s*(\[.*?\]);', script_content, re.DOTALL)
                if gallery_match:
                    try:
                        gallery_json_str = gallery_match.group(1)
                        gallery_data = json.loads(gallery_json_str)
                        for item in gallery_data:
                            if isinstance(item, dict) and item.get('mainUrl') and 'images-amazon.com' in item['mainUrl']:
                                image_urls.add(item['mainUrl']); found_in_script = True
                        if found_in_script: logger.info(f"    SUCCESS: Found {len(image_urls)} URLs via 'imageGalleryData'.")
                    except Exception as e_json_gallery: logger.debug(f"    Could not parse 'imageGalleryData' JSON: {e_json_gallery}")

                # Regex for 'colorImages': {'initial': [...] } pattern
                if not found_in_script:
                    color_images_match = re.search(r"\'colorImages\'\s*:\s*\{\s*\'initial\'\s*:\s*(\[[\s\S]*?\])\s*\}", script_content)
                    if color_images_match:
                        try:
                            initial_array_str = color_images_match.group(1)
                            initial_array_str = initial_array_str.replace("'", '"') # Basic quote normalization
                            initial_array_str = re.sub(r',\s*([\}\]])', r'\1', initial_array_str) # Remove trailing commas
                            
                            image_data_list = json.loads(initial_array_str)
                            for item in image_data_list:
                                if isinstance(item, dict):
                                    for key in ['hiRes', 'large', 'mainUrl']: # Check common keys
                                        url = item.get(key)
                                        if url and isinstance(url, str) and 'images-amazon.com' in url and not url.endswith('.gif'):
                                            image_urls.add(url); found_in_script = True; break 
                            if found_in_script: logger.info(f"    SUCCESS: Found {len(image_urls)} URLs via 'colorImages:initial' pattern.")
                        except Exception as e_json_color: logger.debug(f"    Could not parse 'colorImages:initial' JSON: {e_json_color}")
                
                # Generic regex for URLs in script if specific patterns fail
                if not found_in_script:
                    generic_script_urls = re.findall(r'[\'"](https?://(?:m\.media-amazon\.com|images-amazon\.com)/images/I/[^\'"\s]+\.(?:jpg|png|jpeg|webp))[\'"]', script_content, re.IGNORECASE)
                    if generic_script_urls:
                        for url in generic_script_urls: image_urls.add(url)
                        found_in_script = True
                        logger.info(f"    Found {len(generic_script_urls)} potential URLs via generic script regex.")
                
                if found_in_script and len(image_urls) > 5: break # Stop if we got a good number

    if found_in_script: logger.info(f"    After Strategy 1 (Script Parsing), current total unique URLs: {len(image_urls)}")

    # Strategy 2: Main image attributes as fallback
    if not image_urls or len(image_urls) < 2:
        logger.debug("    Trying Strategy 2 (Fallback): #landingImage attributes")
        try:
            main_img_element = driver.find_element(By.ID, "landingImage")
            dynamic_image_data_str = main_img_element.get_attribute('data-a-dynamic-image')
            if dynamic_image_data_str:
                try:
                    dynamic_images_json = json.loads(dynamic_image_data_str)
                    for url_key in dynamic_images_json.keys():
                        if isinstance(url_key, str) and 'images-amazon.com' in url_key and not url_key.endswith(('.gif','.mp4')):
                            image_urls.add(url_key)
                except: pass
            src_attr = main_img_element.get_attribute("src")
            if src_attr and isinstance(src_attr, str) and 'images-amazon.com' in src_attr and not src_attr.endswith('.gif'):
                image_urls.add(src_attr)
            logger.info(f"    After Strategy 2 (landingImage), current total unique URLs: {len(image_urls)}")
        except Exception as e_strat2: logger.debug(f"    Strategy 2 (#landingImage) error: {e_strat2}")

    # Strategy 3: Thumbnail clicking (use as a final fallback if others yield very few)
    if not image_urls or len(image_urls) < 3: # If still very few
        logger.debug("    Trying Strategy 3 (Fallback): Clicking #altImages thumbnails")
        try:
            alt_images_container = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "altImages")))
            thumbnail_buttons_xpath = ".//li[contains(@class, 'item') or contains(@class, 'imageThumbnail')]//span[contains(@class, 'a-button-thumbnail') or contains(@class, 'thumb') or contains(@class, 'item')]/descendant-or-self::input[@type='submit'] | .//li[contains(@class, 'item') or contains(@class, 'imageThumbnail')]//span[contains(@class, 'a-button-thumbnail')]"
            all_thumb_elements = alt_images_container.find_elements(By.XPATH, thumbnail_buttons_xpath)
            logger.info(f"    Found {len(all_thumb_elements)} clickable thumbnail elements in #altImages.")

            for i in range(len(all_thumb_elements)):
                if len(image_urls) >= 10: break # Limit total images from clicks
                try:
                    current_thumb_elements = driver.find_element(By.ID, "altImages").find_elements(By.XPATH, thumbnail_buttons_xpath)
                    if i >= len(current_thumb_elements): break
                    thumb_to_click = current_thumb_elements[i]
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", thumb_to_click)
                    time.sleep(0.5)
                    driver.execute_script("arguments[0].click();", thumb_to_click)
                    time.sleep(2.5) 
                    main_image_after_click = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "landingImage")))
                    dynamic_data_ac = main_image_after_click.get_attribute('data-a-dynamic-image')
                    if dynamic_data_ac:
                        try:
                            dynamic_json_ac = json.loads(dynamic_data_ac)
                            for url_k in dynamic_json_ac.keys():
                                if 'images-amazon.com' in url_k and not url_k.endswith('.gif'): image_urls.add(url_k)
                        except: pass
                    src_ac = main_image_after_click.get_attribute("src")
                    if src_ac and 'images-amazon.com' in src_ac and not src_ac.endswith('.gif'): image_urls.add(src_ac)
                except Exception as e_click_thumb: logger.debug(f"      Error processing thumbnail click {i+1}: {e_click_thumb}")
            logger.info(f"    After Strategy 3 (altImages clicks), current total unique URLs: {len(image_urls)}")
        except Exception as e_strat3: logger.debug(f"    Strategy 3 (#altImages clicks) error: {e_strat3}")

    final_filtered_urls = set()
    for url in image_urls:
        if url and isinstance(url, str) and \
           ('images-amazon.com' in url or 'm.media-amazon.com' in url) and \
           not url.endswith(('.gif', '.mp4')) and \
           all(kw not in url.lower() for kw in ['sprite', 'icon', 'grey-pixel', 'spinner', 'loading', 'transparent', 'beacon', 'play-icon', 'gradient', 'adpixel', 'advertising', 'deals', 'prime', 'gift', 'badges', 'ellipsis']):
            base_url_for_dedupe = re.sub(r'\._[A-Z0-9,_SXUYACVTZI]+_\.', '.', url, flags=re.IGNORECASE).split('?')[0]
            is_new_visual_base = True
            for existing_final_url in final_filtered_urls:
                existing_base = re.sub(r'\._[A-Z0-9,_SXUYACVTZI]+_\.', '.', existing_final_url, flags=re.IGNORECASE).split('?')[0]
                if base_url_for_dedupe == existing_base:
                    is_new_visual_base = False; break
            if is_new_visual_base: final_filtered_urls.add(url)
            
    logger.info(f"    Returning {len(final_filtered_urls)} unique, filtered image URLs for {product_url}.")
    return list(final_filtered_urls)

if __name__ == "__main__":
    # --- For full run, use this: ---
    df_products = pd.read_csv(PRODUCTS_INPUT_CSV)
    # --- For debugging, use a smaller list: ---
    # product_list_for_debug = [
    #     {'product_id': 'B09QL3NQHX', 'product_url': 'https://www.amazon.in/Rockerz-425-Bluetooth-Headphones-Signature/dp/B09QL3NQHX'},
    #     {'product_id': 'B07PR1CL3S', 'product_url': 'https://www.amazon.in/Rockerz-450-Wireless-Bluetooth-Headphone/dp/B07PR1CL3S'},
    #     # Add a few more diverse product URLs for testing selector robustness
    #      {'product_id': 'B0863FR3S9', 'product_url': 'https://www.amazon.in/Sony-WH-1000XM4-Cancelling-Headphones-Bluetooth/dp/B0863FR3S9'}, # Sony
    #      {'product_id': 'B0DHDDF5J2', 'product_url': 'https://www.amazon.in/boAt-Bassheads-900-Pro-Unidirectional/dp/B0DHDDF5J2'} # Another Boat
    # ]
    # df_products = pd.DataFrame(product_list_for_debug)
    
    all_scraped_image_records = []
    product_id_to_image_filenames_map = {str(row['product_id']).strip(): "" for index, row in df_products.iterrows()}

    # Set headless=False for debugging to see the browser, True for actual run
    driver = setup_selenium_driver(headless=True) 
    if not driver:
        logger.error("Exiting due to WebDriver setup failure.")
        exit()

    try:
        for index, row in df_products.iterrows():
            product_id = str(row['product_id']).strip()
            product_url = str(row['product_url']).strip()
            
            logger.info(f"\nProcessing Product ID: {product_id} ({index+1}/{len(df_products)})")
            if not product_url or not product_url.startswith("http"):
                logger.warning(f"  Skipping invalid URL for {product_id}: {product_url}")
                continue
            
            retrieved_image_urls = extract_all_image_urls_from_product_page(driver, product_url)
            
            current_product_downloaded_filenames = []
            if retrieved_image_urls:
                logger.info(f"  Attempting to download {len(retrieved_image_urls)} images for {product_id}.")
                for i, img_url in enumerate(retrieved_image_urls):
                    if i >= 7: # Limit to 7 images per product for this run
                        logger.info(f"    Reached image limit (7) for {product_id}")
                        break
                    
                    downloaded_filename = download_image_robust(img_url, IMAGE_OUTPUT_FOLDER, product_id, i + 1)
                    if downloaded_filename:
                        full_path = os.path.join(IMAGE_OUTPUT_FOLDER, downloaded_filename)
                        all_scraped_image_records.append({
                            'product_id': product_id,
                            'image_filename': downloaded_filename,
                            'full_image_path': full_path,
                            'original_url': img_url
                        })
                        current_product_downloaded_filenames.append(downloaded_filename)
            else:
                logger.warning(f"  No image URLs extracted for {product_id} from {product_url}")
            
            product_id_to_image_filenames_map[product_id] = ",".join(current_product_downloaded_filenames)
            
            if index < len(df_products) - 1 :
                delay_seconds = 5 # Be respectful with delays
                logger.info(f"--- Delaying for {delay_seconds} seconds before next product ({product_id} done) ---")
                time.sleep(delay_seconds) 
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user.")
    except Exception as e_main:
        logger.error(f"An unexpected error occurred in the main loop: {e_main}", exc_info=True)
    finally:
        if driver:
            driver.quit()
            logger.info("WebDriver closed.")

    if all_scraped_image_records:
        df_all_images_info = pd.DataFrame(all_scraped_image_records)
        df_all_images_info.to_csv(IMAGE_INFO_OUTPUT_CSV, index=False, encoding='utf-8-sig')
        logger.info(f"\nSaved info for {len(df_all_images_info)} scraped images to {IMAGE_INFO_OUTPUT_CSV}")
    else:
        logger.info("\nNo images were successfully scraped and recorded.")

    df_products_original_for_update = pd.read_csv(PRODUCTS_INPUT_CSV) 
    df_products_to_update = df_products_original_for_update.copy()
    df_products_to_update['product_id'] = df_products_to_update['product_id'].astype(str).str.strip()
    
    def map_image_paths(pid_map_val):
        pid_str = str(pid_map_val).strip()
        new_paths = product_id_to_image_filenames_map.get(pid_str)
        
        if new_paths: 
            return new_paths
        else: 
            original_paths_series = df_products_original_for_update.loc[
                df_products_original_for_update['product_id'].astype(str).str.strip() == pid_str, 'image_paths'
            ]
            if not original_paths_series.empty and pd.notna(original_paths_series.iloc[0]):
                return original_paths_series.iloc[0]
            return "" 

    df_products_to_update['image_paths'] = df_products_to_update['product_id'].apply(map_image_paths)
    
    df_products_to_update.to_csv(UPDATED_PRODUCTS_CSV_OUTPUT, index=False, encoding='utf-8-sig')
    logger.info(f"Updated products CSV with potentially new image paths saved to {UPDATED_PRODUCTS_CSV_OUTPUT}")

    logger.info("\nImage scraping from URLs process complete.")