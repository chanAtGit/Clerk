import os
import platform
import gc
import torch
import re, unicodedata
import numpy as np
from sentence_transformers import SentenceTransformer

from diskcache import Cache # for caching mean embeddings
from pathlib import Path
from PIL import Image
from docx2pdf import convert
from pdf2image import convert_from_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader

poppler_path = None
embedding_model = None 
cache = None 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=0,
)

def embeddings_init():
    global poppler_path
    poppler_path = os.getenv("POPPLER_PATH")

    if not poppler_path and platform.system() != 'Linux': # only linux has poppler automatically installed
        raise ValueError("POPPLER_PATH environment variable is not set.")

def clear_vram():
    """Forces Python garbage collection and clears PyTorch CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()

def get_file_id(file_path: str):
    # Fetch file system attributes
    stat_info = os.stat(file_path)
    # Pair device ID and inode/file index
    return (stat_info.st_dev, stat_info.st_ino)

def store_cache_embedding(file_path: str, embedding):
    global cache
    try:
        if cache is None:
            print("Cache Error: Cache is not initialised.")
            return
        if not os.path.exists(file_path):
            print(f"Cache Error: File does not exist: {file_path}")
            return
        # get unique file id
        file_id = get_file_id(file_path)
        value = {
            "embedding": embedding,
            "modified_time": os.path.getmtime(file_path) # get modification timestamp
        }
        # set key-value pair
        cache.set(file_id, value)
    except Exception as e:
        print(f"Cache Error: Failed to store embedding for {file_path}: {e}")

def get_cache_embedding(file_path: str):
    global cache
    try:
        if cache is None:
            print("Cache Error: Cache is not initialised.")
            return None
        if not os.path.exists(file_path):
            print(f"Cache Error: File does not exist: {file_path}")
            return None
    except Exception as e:
        print(f"Cache Error: Failed to retrieve embedding for {file_path}: {e}")
        return None
    # get unique file id
    file_id = get_file_id(file_path)
    # get value (embedding) from id
    value = cache.get(file_id)
    if value is None or os.path.getmtime(file_path) != value["modified_time"]: # cache miss or file has been modified
        return None
    return value["embedding"]

def convert_pdf_to_img(pdf_path: str):
    global poppler_path
    if not os.path.exists(pdf_path):
        return None
    if platform.system() != 'Linux':
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
    else:
        images = convert_from_path(pdf_path)
    return images

def cache_init():
    global cache
    # Create or load cache. Cap the cache at 1000 items, using the Least Recently Used (LRU) policy
    if cache is None:
        cache = Cache('embedding-cache', limit=1000, evict='lru')

def load_embedding_model():
    global embedding_model
    local_path = "models/qwen3-vl-embedding-2b"
    if embedding_model is None:
        print("Loading embedding model into VRAM...")
        if os.path.exists(local_path):
            print("Loading local model...")
            embedding_model = SentenceTransformer(local_path, device="cuda", local_files_only=True)
        else:
            embedding_model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device="cuda")
            embedding_model.save("models/qwen3-vl-embedding-2b")

def unload_embedding_model():
    global embedding_model, cache
    if embedding_model is not None:
        print("Unloading embedding model from VRAM...")
        del embedding_model
        embedding_model = None
        clear_vram()
    if cache is not None:
        cache.close()
        del cache
        cache = None

def clean_text(text):
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    contractions = {
        "don't": "do not", "can't": "cannot", "won't": "will not",
        "n't": " not", "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    text = text.replace("'", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_pdf_mean_embedding(pdf_path: str, status_callback=None):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        if text_splitter.split_documents(docs):
            text_chunks = text_splitter.split_documents(docs)
            cleaned_sentences = [clean_text(str(chunk)) for chunk in text_chunks]
            embeddings = embedding_model.encode(cleaned_sentences, convert_to_numpy=True)
            return np.mean(embeddings, axis=0)
        else:
            images = convert_pdf_to_img(pdf_path)
            embeddings = []
            for img in images:
                img_embedding = embedding_model.encode(img, convert_to_numpy=True)
                embeddings.append(img_embedding)
            return np.mean(embeddings, axis=0)
    except Exception as e:
        msg = f"  → Error processing {pdf_path}: {e}"
        print(msg)
        if status_callback: status_callback(msg)
        return None

def generate_docx_mean_embedding(docx_path: str, status_callback=None):
    try:
        loader = UnstructuredWordDocumentLoader(docx_path, mode="single")
        docs = loader.load()
        text_chunks = text_splitter.split_documents(docs)
        if text_chunks:
            cleaned_sentences = [clean_text(str(chunk)) for chunk in text_chunks]
            embeddings = embedding_model.encode(cleaned_sentences, convert_to_numpy=True)
            return np.mean(embeddings, axis=0)
        else:
            temp_pdf_path = "temp_doc.pdf"
            convert(docx_path, temp_pdf_path)
            images = convert_pdf_to_img(temp_pdf_path)
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            embeddings = []
            for img in images:
                img_embedding = embedding_model.encode(img, convert_to_numpy=True)
                embeddings.append(img_embedding)
            return np.mean(embeddings, axis=0)
    except Exception as e:
        msg = f"  → Error processing {docx_path}: {e}"
        print(msg)
        if status_callback: status_callback(msg)
        return None

def get_file_mean_embeddings(dir_path: str, files_list: list, status_callback=None, progress_callback=None, check_cancel=None) -> dict:
    if not files_list:
        return None

    mean_embeddings = {}
    total_files = len(files_list)
    cache_init()
    
    for idx, file_name in enumerate(files_list):
        if check_cancel and check_cancel():
            raise InterruptedError()

        file_path = os.path.join(dir_path, file_name)
        
        if not os.path.isfile(file_path):
            continue

        msg = f"Processing embedding: {file_name}"
        print(msg)
        if status_callback: status_callback(msg)

        # Check cache for embedding
        mean_embedding = get_cache_embedding(file_path)
        
        # If the cache misses
        if mean_embedding is None:
            load_embedding_model() # only load model if cache miss
            # generate mean embedding
            match Path(file_path).suffix.lower():
                case '.pdf':
                    mean_embedding = generate_pdf_mean_embedding(file_path, status_callback)
                case '.docx':
                    mean_embedding = generate_docx_mean_embedding(file_path, status_callback)
                case '.png' | '.jpg' | '.jpeg' | '.avif' | '.bmp' | '.tiff'| '.webp':
                    mean_embedding = embedding_model.encode(Image.open(file_path), convert_to_numpy=True)
            if mean_embedding is not None: 
                store_cache_embedding(file_path, mean_embedding)
            else:
                continue
        
        mean_embeddings[file_name] = mean_embedding
        
        if progress_callback:
            progress_callback(int(((idx + 1) / total_files) * 50))

    if len(mean_embeddings) == 0:
        return None
    
    return mean_embeddings

def get_directory_mean_embeddings(subdirectories: list) -> dict:
    # Find all subdirectories in the target directory
    if len(subdirectories) == 0:
        print("No subdirectories found. Exiting.")
        return None

    load_embedding_model()
    mean_embeddings = {} # a dictionary with elements in the key-value format of {file_name: mean_embedding}

    for dir_name in subdirectories:
        mean_embeddings[dir_name] = embedding_model.encode(str(dir_name), convert_to_numpy=True)

    # Ensure we have processed at least some data
    if len(mean_embeddings) == 0:
        print("No embeddings were generated. Exiting.")
        return None
    
    return mean_embeddings

# --- Helper 1: Extract File Content Snippets ---
def get_file_sample(file_path: str) -> str | list:
    """Extracts a text chunk sample or returns PIL Images if the file has no extractable text."""
    global poppler_path
    try:
        suffix = Path(file_path).suffix.lower()
        match suffix:
            case '.pdf':
                loader = PyPDFLoader(file_path)
            case '.docx':
                loader = UnstructuredWordDocumentLoader(file_path, mode="single")
            case _:
                return "(Unsupported format)"

        docs = loader.load()
        snippet_docs = docs[:3] 
        chunks = text_splitter.split_documents(snippet_docs)

        if chunks:
            text_sample = " ".join([str(c.page_content) for c in chunks[:2]])
            return text_sample[:1200]
        else:
            # Fallback to images (OCR/Vision context)
            images = convert_pdf_to_img(file_path)
            return images
    except Exception as e:
        return f"(Could not read file content: {e})"