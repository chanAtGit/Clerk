import os
import shutil
from pathlib import Path
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForMultimodalLM
from sentence_transformers import SentenceTransformer
import ollama
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import re, unicodedata, collections
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from PIL import Image
from docx2pdf import convert
from pdf2image import convert_from_path
import time

# --- Configuration ---
# Specify the directory containing your PDF files here

poppler_path = None  # Update this path to your Poppler bin directory if needed
embedding_model = None  # Global variable to hold the embedding model instance  
# Update this path to your Poppler bin directory

def init_huggingface_and_models():
    """Initialize Hugging Face login and load the embedding model."""
    global embedding_model, poppler_path
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    poppler_path = os.getenv("POPPLER_PATH")  # Get Poppler path from environment variable

    if not huggingface_token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")
    if not poppler_path:
        raise ValueError("POPPLER_PATH environment variable is not set.")

    login(token=huggingface_token)
    embedding_model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device="cuda")

# MODEL_ID = "google/gemma-4-E2B-it"
# processor = AutoProcessor.from_pretrained(MODEL_ID)
# llm = AutoModelForMultimodalLM.from_pretrained(
#     MODEL_ID, 
#     dtype="auto", 
#     device_map="auto"
# )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=0,
)
# ---------------------

# Pre-processing #
def clean_text(text):
    """Advanced text cleaning pipeline for better clustering performance"""
    # 1. Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove URLs and emails
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # 4. Remove isolated numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    # 5. Remove punctuation but keep apostrophes in contractions
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # 6. Handle contractions
    contractions = {
        "don't": "do not", "can't": "cannot", "won't": "will not",
        "n't": " not", "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # 7. Remove remaining apostrophes
    text = text.replace("'", "")
    
    # 8. Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def optimal_kmeans_clustering(data:list):
    data = np.array(data)
    centroid_values= np.arange(1, int(data.shape[0] / 2))
    # Clinton's note: why is the maximum possible number of centroids half of the number of data points? 
    # because it assumes that a directory should contain at least 2 files
    best_centroid = None
    best_score = -1
    best_labels = None

    for centroid in centroid_values:
        kmeans = KMeans(n_clusters=int(centroid), random_state=42)
        labels = kmeans.fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        #print(f"Trying centroid={centroid:.3f}: Found {n_clusters} clusters.")
        
        if n_clusters > 1:
            core_samples_mask = labels != -1
            search_labels = labels[core_samples_mask]
            
            # Ensure we still have multiple clusters after removing noise
            if len(set(search_labels)) > 1:
                score = silhouette_score(data[core_samples_mask], search_labels, metric="cosine")
                #print(f"Silhouette Score (excluding noise): {score:.3f}") 
                
                if score > best_score: 
                    best_score = score 
                    best_centroid = centroid
                    best_labels = labels

    print(f"Best clusters: {best_centroid}, score: {best_score}")
    return best_centroid, best_labels

def generate_pdf_mean_embedding(pdf_path: str):
    print(f'Processing PDF: {pdf_path}')
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"  → Loaded {len(docs)} page(s).")
                
        text_chunks = text_splitter.split_documents(docs)
        # Split into chunks
        if text_chunks:
            cleaned_sentences = [clean_text(str(chunk)) for chunk in text_chunks]
            # Part 1: Encode with qwen3-vl-embedding
            embeddings = embedding_model.encode(cleaned_sentences, convert_to_numpy=True)
            # Calculate mean embedding for the entire document
            return np.mean(embeddings, axis=0)
        else:
            # Convert PDF to images
            images = convert_from_path(pdf_path, poppler_path = poppler_path)
            # Initialize list to store embeddings
            embeddings = []
            # Process each image
            for img in images:
                # Generate embedding for the image (assuming you have an image embedding model)
                img_embedding = embedding_model.encode(img, convert_to_numpy=True)
                embeddings.append(img_embedding)
            # Calculate mean embedding for the entire document
            return np.mean(embeddings, axis=0)
    except Exception as e:
        print(f"  → Error processing {pdf_path}: {e}")
        return None

def generate_docx_mean_embedding(docx_path: str):
    print(f'Processing docx: {docx_path}')
    try:
        # Load docx
        loader = UnstructuredWordDocumentLoader(docx_path, mode="single")
        docs = loader.load()
        print(f"  → Loaded {len(docs)} page(s).")
                
        text_chunks = text_splitter.split_documents(docs)
        # Split into chunks
        if text_chunks:
            cleaned_sentences = [clean_text(str(chunk)) for chunk in text_chunks]
            # Part 1: Encode with qwen3-vl-embedding
            embeddings = embedding_model.encode(cleaned_sentences, convert_to_numpy=True)
            # Calculate mean embedding for the entire document
            return np.mean(embeddings, axis=0)
        else:
            temp_pdf_path = "temp_doc.pdf"
            # Convert docx to pdf
            convert(docx_path, temp_pdf_path)
            # Convert pdf to images
            images = convert_from_path(temp_pdf_path)
            # Remove temp docx
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            # Initialize list to store embeddings
            embeddings = []
            # Process each image
            for img in images:
                # Generate embedding for the image (assuming you have an image embedding model)
                img_embedding = embedding_model.encode(img, convert_to_numpy=True)
                embeddings.append(img_embedding)
            # Calculate mean embedding for the entire document
            return np.mean(embeddings, axis=0)
    except Exception as e:
        print(f"  → Error processing {docx_path}: {e}")
        return None

def get_file_mean_embeddings(dir_path: str, files_list: list) -> dict:
    if not files_list:
        print("No files found. Exiting.")
        return None

    mean_embeddings = {} # a dictionary with elements in the key-value format of {file_name: mean_embedding}

    # Dynamic loading and embedding loop
    for file_name in files_list:
        file_path = os.path.join(dir_path, file_name)
        if not os.path.isfile(file_path):
            continue

        mean_embedding = None
        print(f"\nProcessing: {file_name}")

        match Path(file_path).suffix:
            case '.pdf': # processing pdf file
                mean_embedding = generate_pdf_mean_embedding(file_path)
            case '.docx': # processing word file
                mean_embedding = generate_docx_mean_embedding(file_path)
            case '.png' | '.jpg' | '.jpeg' | '.bmp' | '.tiff': # processing image file
                mean_embedding = embedding_model.encode(Image.open(file_path), convert_to_numpy=True)
        
        if mean_embedding is not None: # mean_embedding is generated
            # Append to the dictionary
            mean_embeddings[file_name] = mean_embedding
    # Ensure we have processed at least some data
    if len(mean_embeddings) == 0:
        print("No embeddings were generated. Exiting.")
        return None
    
    return mean_embeddings

def SemanticClustering(dir_path: str) -> dict:
    files_list = [f for f in os.listdir(dir_path)]
    if not files_list:
        print("No files found. Exiting.")
        return None
    
    file_embeddings_dict = get_file_mean_embeddings(dir_path, files_list)
    if not file_embeddings_dict:
        return None
    
    file_names = list(file_embeddings_dict.keys())
    mean_embeddings = list(file_embeddings_dict.values())

    # Part 2: Semantic clustering with K Means Clustering #
    best_centroids, labels_final = optimal_kmeans_clustering(mean_embeddings)

    groups_final = collections.defaultdict(list)
    for i, lab in enumerate(labels_final):
        groups_final[str(lab)].append(file_names[i])

    groups_final = sorted(groups_final.items(), key=lambda x: (int(x[0]) == -1, int(x[0])))

    return groups_final 

def AutoLabelClusters(dir_path: str, groups_final: list) -> dict:
    # Auto labelling of the clusters based on content using a language model.
    if not groups_final:
        return {}
    print("\n--- Generating Cluster Labels via LLM Content Analysis ---")

    # Step 1: Reload a snippet of text from each file to understand its content
    file_contents = {}

    for lab, items in groups_final:
        if str(lab) == "-1":  # Fixed condition to match string "-1"
            continue
        for file_name in items:
            file_path = os.path.join(dir_path, file_name)
            loader = None
            try:
                match Path(file_path).suffix:
                    case '.pdf': # processing pdf file
                        loader = PyPDFLoader(file_path)
                    case '.docx': # processing word file
                        loader = UnstructuredWordDocumentLoader(file_path, mode="single")

                docs = loader.load()
                # Extract up to the first 3 pages to avoid overflowing the LLM prompt window
                snippet_docs = docs[:3] 
                chunks = text_splitter.split_documents(snippet_docs)

                if chunks:
                    # Combine the first 2 chunks as a representative summary of the document's intro
                    text_sample = " ".join([str(c.page_content) for c in chunks[:2]])
                    file_contents[file_name] = text_sample[:1200]  # Cap characters per file
                else:
                    match Path(file_path).suffix:
                        case '.pdf': # processing pdf file
                            images = convert_from_path(file_path, poppler_path = poppler_path) # convert the pdf to list of images
                            file_contents[file_name] = images
                        case '.docx': # processing word file
                            temp_pdf_path = "temp_doc.pdf"
                            # Convert docx to pdf
                            convert(file_path, temp_pdf_path)
                            images = convert_from_path(temp_pdf_path, poppler_path = poppler_path) # convert the pdf to list of images
                            # Remove temp docx
                            if os.path.exists(temp_pdf_path):
                                os.remove(temp_pdf_path)
                            file_contents[file_name] = images
    
            except Exception:
                file_contents[file_name] = "(Could not read file content)"

    # Step 2: Prompt the LLM with the actual text snippets and build a new labeled dict
    labeled_groups = {}

    for lab, items in groups_final:
        if str(lab) == "-1":
            labeled_groups["Misc"] = items # label the cluster as Misc
            continue
            
        if len(items) == 1:
            continue # skip over clusters that only contain one file

        # Build a context string containing snippets from the files in this cluster
        cluster_context = ""
        img_index = 0
        img_context = []
        temp_img_dir = os.path.join(dir_path, f"cluster_{lab}_img")

        for file_name in items:
            if isinstance(file_contents.get(file_name, ''), str): # if the file_content of the file contains words
                cluster_context += f"--- File: {file_name} ---\n"
                cluster_context += f"Content Sample: {file_contents.get(file_name, '')}\n\n"
            else: # file_contents contain images
                for page in file_contents[file_name]:
                    if not os.path.exists(temp_img_dir):
                        os.mkdir(temp_img_dir) # create temporary image directory
                    image_name = f"{img_index}.png"
                    full_output_path = os.path.join(temp_img_dir, image_name) # save images inside folder
                     # Save the file
                    page.save(full_output_path, "PNG")
                    img_context.append(full_output_path)
                    img_index += 1
        
        prompt = f'''You are an expert data cataloger. Review the following text snippets from a group of files
            that belong to the same semantic cluster. IF the CLUSTER CONTENT IS EMPTY, rely on the given images instead.
            Based on their actual content, generate a highly concise,precise 2-to-4 word topic label for this cluster. 
            Do not include introductory text, do not use quotes, 
            and MOST IMPORTANTLY just return the label.\n\n
            Cluster Content:\n{cluster_context}\n
            Topic Label: '''
        
        try:
            # Utilizing ollama for local processing
            response = ollama.chat(
                model='gemma4:cloud',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': img_context
                    }
                ]
            )
            print(response['message']['content'])
            label = response['message']['content']
            
            # Sanitize the label to ensure it can be a valid directory name
            label = re.sub(r'[\\/*?:"<>|]', "", label)
            if label:
                labeled_groups[label] = items
            else:
                labeled_groups[f"Cluster_{lab}"] = items
        except Exception as e:
            print(f"Warning: Failed to generate label for Cluster {lab} via LLM ({e}).")
            labeled_groups[f"Cluster_{lab}"] = items
        
        # Delete temp img directory and everything inside it
        if os.path.exists(temp_img_dir):
            shutil.rmtree(temp_img_dir)

    # Update the print loop to show the content-driven dynamic labels
    print(f"\nFinal clusters found:")
    for lab, items in labeled_groups.items():
        print(f"\n{lab} ({len(items)} items):")
        for item in items:
            print(f"  • {item}")

    return labeled_groups

def MoveFiles(dir_path: str, groups_final: dict):
    if not groups_final:
        return
    # Create directories for each cluster and move files into them
    for lab, items in groups_final.items():
        if not lab or not items or len(items) == 1:
            continue
        cluster_dir = os.path.join(dir_path, lab)
        
        # Ensure the directory exists cleanly
        os.makedirs(cluster_dir, exist_ok=True)
        
        for item in items:
            src_path = os.path.join(dir_path, item)
            dest_path = os.path.join(cluster_dir, item)
            try:
                if os.path.exists(src_path):
                    os.rename(src_path, dest_path)
            except Exception as e:
                print(f"Error moving {item} to {lab}: {e}")