# Multimodal RAG System for Visual Shopping Assistance

This project implements a progressively developed **Multimodal Retrieval-Augmented Generation (RAG)** system to answer e-commerce product queries using both **textual** and **visual data**. Built for the **headphones** category on Amazon.in, it leverages product specifications, user reviews, images, BLIP captions, and OCR-derived text to provide grounded, context-aware responses via Google Gemini.

---

## Project Overview

The system evolves across three major iterations:

| Iteration | Description                                                    | Entry Point                   |
|-----------|----------------------------------------------------------------|------------------------------|
| 1         | Basic RAG using text-only chunks and a single BLIP caption per image | `main_assistant.py`           |
| 2         | Adds multiple BLIP captions and ViLT-based reranking           | `main_assistant_new.py`       |
| 3         | Integrates filtered BLIP captions and OCR text into a unified retrieval pipeline | `main_assistant_new_one.py` (Recommended) |

---

## Recommended File Structure

For smooth execution across embedding, retrieval, and assistant pipelines, flatten all scripts and folders into a single directory structure:


```project-root/
├── scrapped_dataset/              # Final CSVs: metadata, reviews, specs, image info
├── scraper/                       # Web scraping scripts
├── embedding/                     # Embedding generation, BLIP, OCR scripts

├── retriever.py                   # Iteration 1 retriever
├── retriever_new.py               # Iteration 2 retriever
├── retriever_new_one.py           # Iteration 3 retriever

├── llm_handler.py                 # Iteration 1 LLM handler
├── llm_handler_new.py             # Iteration 2 LLM handler
├── llm_handler_new_one.py         # Iteration 3 LLM handler

├── main_assistant.py              # Iteration 1 chatbot
├── main_assistant_new.py          # Iteration 2 chatbot
├── main_assistant_new_one.py      # Iteration 3 chatbot (recommended)

├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```
## Installation

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd Multimodal-RAG-System-for-Visual-Shopping-Assistance
2.Install dependencies:
pip install -r requirements.txt
3.Set environment variables:
export PINECONE_API_KEY=your_pinecone_api_key
export PINECONE_ENVIRONMENT=your_pinecone_environment
export GEMINI_API_KEY=your_gemini_api_key
Required API Keys
Google Gemini Flash API Key

Pinecone API Key and Environment

These are used for language generation and vector database storage/retrieval, respectively.

Models Used
The following pre-trained models are used throughout the pipeline:

Purpose	Model Name
Text Embedding	all-mpnet-base-v2
Cross Encoder	cross-encoder/stsb-roberta-base
Image Embedding	openai/clip-vit-base-patch32
Image Captioning	Salesforce/blip-image-captioning-large
Image Reranking	dandelin/vilt-b32-finetuned-vqa


How to Run the Chatbot
Run one of the following assistant scripts:

# Iteration 1 (Basic RAG)
python main_assistant.py

# Iteration 2 (BLIP + ViLT reranking)
python main_assistant_new.py

# Iteration 3 (BLIP + OCR + Semantic Filtering) - Recommended
python main_assistant_new_one.py

Dataset Details
All cleaned and processed dataset files are located in the scrapped_dataset/ directory:

products_final.csv: Product metadata (title, price, category, image paths)

customer_reviews_scraped_v3.csv: User reviews and aspect summaries

all_documents.csv: Descriptions and specifications

all_product_images_info_scraped.csv: Image metadata

valid_product_images.csv: Filtered list of usable images

image_captions_multiple.csv: Multiple BLIP-generated captions per image

image_ocr_texts_cleaned.csv: Cleaned OCR outputs

image_combined_blip_ocr_filtered_final.csv: Final filtered captions + OCR metadata

Notes
Flatten all subfolders into a single directory before running scripts.

Update all relative paths in the scripts if you restructure the project.

Use the latest iteration (main_assistant_new_one.py) for the most accurate and visually grounded responses.

Embeddings can be regenerated using scripts in the embedding/ directory.

