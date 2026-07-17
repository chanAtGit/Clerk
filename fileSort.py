import os
import platform
import shutil
import torch
import gc
import numpy as np
import re, unicodedata
import ollama
from diskcache import Cache # for caching mean embeddings
from pathlib import Path
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForMultimodalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from PIL import Image
from docx2pdf import convert
from pdf2image import convert_from_path

# --- Configuration ---
poppler_path = None
processor = None
llm = None  
embedding_model = None 
cache = None 

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
    if cache is None:
        print("Cache Error: Cache is not initialised.")
        return
    if not os.path.exists(file_path):
        print(f"Cache Error: File does not exist: {file_path}")
        return
    # get unique file id
    file_id = get_file_id(file_path)
    # set key-value pair
    cache.set(file_id, embedding)

def get_cache_embedding(file_path: str):
    global cache
    if cache is None:
        print("Cache Error: Cache is not initialised.")
        return None
    if not os.path.exists(file_path):
        print(f"Cache Error: File does not exist: {file_path}")
        return None
    # get unique file id
    file_id = get_file_id(file_path)
    # get value (embedding) from id
    embedding = cache.get(file_id)
    return embedding

def convert_pdf_to_img(pdf_path: str):
    global poppler_path
    if not os.path.exists(pdf_path):
        return None
    if platform.system() != 'Linux':
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
    else:
        images = convert_from_path(pdf_path)
    return images

def load_embedding_model():
    global embedding_model, cache
    if embedding_model is None:
        print("Loading embedding model into VRAM...")
        embedding_model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device="cuda")
    if cache is None:
        cache = Cache('my-cache', limit=10000, evict='lru')
        # Create or load cache. Cap the cache at 10,000 items, using the Least Recently Used (LRU) policy

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

def load_llm():
    global processor, llm
    if llm is None:
        print("Loading LLM into VRAM...")
        model_id = "google/gemma-4-E2B-it"
        processor = AutoProcessor.from_pretrained(model_id)
        llm = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")

def unload_llm():
    global processor, llm
    if llm is not None:
        print("Unloading LLM from VRAM...")
        del llm
        del processor
        llm = None
        processor = None
        clear_vram()

def file_sort_init():
    """Initialize Hugging Face login and paths. Models are loaded dynamically later."""
    global poppler_path
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    poppler_path = os.getenv("POPPLER_PATH")  

    if not huggingface_token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")
    
    if not poppler_path and platform.system() != 'Linux': # only linux has poppler automatically installed
        raise ValueError("POPPLER_PATH environment variable is not set.")

    login(token=huggingface_token)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=0,
)

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

def optimal_clustering(embeddings:list, at_root:bool=True, recursive:bool=True) -> list:
    if at_root: # if the clustering takes place at root directory, PCA is needed to remove noise
        print("Start clustering...")
        pca = PCA(n_components=3, random_state=42)
        reduced = pca.fit_transform(np.array(embeddings))
    else:
        reduced = np.array(embeddings)
    cluster_values = np.arange(1, int(reduced.shape[0]))
    
    best_clusters = None
    best_score = -1
    best_labels = None

    for cluster in cluster_values:
        model = AgglomerativeClustering(n_clusters=cluster, metric="cosine", linkage='average')
        # cosine metric is the best for high dimensional embeddings
        labels = model.fit_predict(reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:
            core_samples_mask = labels != -1
            search_labels = labels[core_samples_mask]
            
            if len(set(search_labels)) > 1:
                score = silhouette_score(reduced[core_samples_mask], search_labels, metric="cosine")

                if score > best_score: 
                    best_score = score 
                    best_clusters = cluster
                    best_labels = labels

    if best_score > 0.45:
        print(f"Best clusters: {best_clusters}, score: {best_score:.3f}")
        best_labels = [str(lab + 1) for lab in best_labels] # convert the labels from int to str. First label is "1"
        
        if recursive:
            for cluster_lab in range(1, best_clusters + 1):
                cluster_embeddings = []
                for i in range(len(embeddings)):
                    if best_labels[i] == str(cluster_lab):
                        # get all embeddings that belong to a cluster
                        cluster_embeddings.append(embeddings[i])
                
                if len(cluster_embeddings) < 3: # skipping over clusters that only have 2 elements
                    continue
                
                # generate sub labels
                sub_labels = optimal_clustering(cluster_embeddings, at_root=False)
                if sub_labels:
                    # append sub labels to labels (eg. A label would be 1.1, meaning it belongs to cluster 1 and its subcluster 1)
                    current = 0
                    for i in range(len(best_labels)):
                        if best_labels[i] == str(cluster_lab):
                            best_labels[i] = best_labels[i] + "." + sub_labels[current]
                            current += 1

        return best_labels
    else:
        print(f"Best score {best_score:.3f} is below 0.45 threshold. No cluster generated.")
        return None

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
            # generate mean embedding
            match Path(file_path).suffix.lower():
                case '.pdf':
                    mean_embedding = generate_pdf_mean_embedding(file_path, status_callback)
                case '.docx':
                    mean_embedding = generate_docx_mean_embedding(file_path, status_callback)
                case '.png' | '.jpg' | '.jpeg' | '.avif' | '.bmp' | '.tiff'| '.webp':
                    mean_embedding = embedding_model.encode(Image.open(file_path), convert_to_numpy=True)
            if mean_embedding is not None: store_cache_embedding(file_path, mean_embedding)
        
        if mean_embedding is not None:
            mean_embeddings[file_name] = mean_embedding
        
        if progress_callback:
            progress_callback(int(((idx + 1) / total_files) * 50))

    if len(mean_embeddings) == 0:
        return None
    
    return mean_embeddings

def insert_groups_dict(groups: dict, lab: str, file_name: str):
    """Recursively inserts a file into a nested hierarchical cluster dictionary."""
    parts = lab.split('.', 1)
    current_key = parts[0]
    
    if len(parts) > 1:
        # We have sub-labels remaining (e.g., "2" from "2.1")
        remaining_lab = parts[1]
        if current_key not in groups:
            groups[current_key] = {}
        elif isinstance(groups[current_key], list):
            # Fallback wrapper if a node suddenly becomes parent of a sub-cluster
            groups[current_key] = {"_unclustered": groups[current_key]}
            
        insert_groups_dict(groups[current_key], remaining_lab, file_name)
    else:
        # We are at the final leaf cluster
        if current_key not in groups:
            groups[current_key] = []
        elif isinstance(groups[current_key], dict):
            # Fallback if leaf label matches an existing parent folder
            if "_unclustered" not in groups[current_key]:
                groups[current_key]["_unclustered"] = []
            groups[current_key]["_unclustered"].append(file_name)
            return
            
        groups[current_key].append(file_name)

def SemanticClustering(dir_path: str, status_callback=None, progress_callback=None, check_cancel=None) -> dict:
    if check_cancel and check_cancel(): raise InterruptedError()
    files_list = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    if not files_list:
        if status_callback: status_callback("No files found to cluster.")
        return None
    
    load_embedding_model()
    
    try:
        file_embeddings_dict = get_file_mean_embeddings(dir_path, files_list, status_callback, progress_callback, check_cancel)

        if not file_embeddings_dict:
            return None
        
        if check_cancel and check_cancel(): raise InterruptedError()
        file_names = list(file_embeddings_dict.keys())
        mean_embeddings = list(file_embeddings_dict.values())

        labels_final = optimal_clustering(embeddings=mean_embeddings)
        groups_final = {}
        if labels_final:
            for i, lab in enumerate(labels_final):
                insert_groups_dict(groups_final, lab, file_names[i])

             # Recursive sorting function
            def sort_nested_dict(d):
                if isinstance(d, dict):
                    def sort_key(k):
                        if k == '-1':
                            return (True, 0)  # Keep Noise (-1) at the absolute bottom
                        try:
                            return (False, float(k))  # Sort numerically (e.g., '2' before '10')
                        except ValueError:
                            return (False, k)  # Alphabetical fallback
                    
                    # Sort dictionary keys and recursively process child nodes
                    sorted_items = sorted(d.items(), key=lambda x: sort_key(x[0]))
                    return [(k, sort_nested_dict(v)) for k, v in sorted_items]
                elif isinstance(d, list):
                    # Alphabetically sort final list of file names
                    return sorted(d)
                return d

            return sort_nested_dict(groups_final)
        else:
            if status_callback: status_callback("No cluster generated due to low silhouette score or not enough files.")
            return None
    finally:
        # Guarantee the embedding model unloads even if user cancels
        unload_embedding_model()

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


# --- Helper 2: LLM Inference Sequence ---
def generate_llm_label(prompt: str, img_paths: list = None, online: bool = False, img_only:bool = False, check_cancel=None) -> str:
    """Executes prompt formatting and tensor processing to query the LLM model."""
    global processor, llm
    try:
        if not online:
            print("Local Inference")
            content_list = []
            loaded_images = []
            
            if img_paths:
                for img_path in img_paths:
                    if check_cancel and check_cancel():
                        raise InterruptedError()
                    content_list.append({"type": "image"})
                    loaded_images.append(Image.open(img_path).convert("RGB"))
                    
            content_list.append({"type": "text", "text": prompt})
            
            messages = [{"role": "user", "content": content_list}]
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            if loaded_images:
                inputs = processor(text=text_prompt, images=loaded_images, return_tensors="pt")
            else:
                inputs = processor(text=text_prompt, return_tensors="pt")
                
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = llm.generate(**inputs, max_new_tokens=40)
                
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)]
            label = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            
            # Strip filesystem-incompatible characters
            label = re.sub(r'[\\/*?:"<>|]', "", label)
            return label if label else "Unlabeled_Cluster"
        else:
            # Online mode with Ollama API
            print("Online Inference")
            if img_paths:
                img_paths = [path for path in img_paths if path.lower().endswith(('.png', '.jpeg', '.jpg'))]
                if img_paths is None or len(img_paths) < 1 and img_only:
                    return "Unlabeled_Cluster"
            
            response = ollama.chat(
                model="gemma4:cloud", 
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': img_paths # Pass the path string or bytes directly
                }]
            )
            label = response.message.content
            label = re.sub(r'[\\/*?:"<>|]', "", label)
            return label if label else "Unlabeled_Cluster"
    except Exception as e:
        print(f"Error executing LLM generation: {e}")
        return "Unlabeled_Cluster"


# --- Core Orchestration & Recursive Processing ---
def AutoLabelClusters(dir_path: str, groups_final: dict | list, status_callback=None, progress_callback=None, check_cancel=None) -> dict:
    global processor, llm
    if not groups_final:
        return {}
        
    msg = "Analyzing clusters via LLM content descriptions..."
    print(msg)
    if status_callback: status_callback(msg)
    
    load_llm()

    # Helpers to support both raw dictionaries and sorted list-of-tuples representations
    def is_leaf_node(node):
        if isinstance(node, list):
            return len(node) == 0 or isinstance(node[0], str)
        return False

    def get_node_children(node):
        if isinstance(node, dict):
            return list(node.items())
        elif isinstance(node, list) and len(node) > 0 and isinstance(node[0], tuple):
            return node
        return None

    # Recursively merges duplicate nodes together
    def merge_nodes(n1, n2):
        if is_leaf_node(n1) and is_leaf_node(n2):
            # Combine file lists and preserve insertion order
            return list(dict.fromkeys(n1 + n2))
            
        elif isinstance(n1, dict) and isinstance(n2, dict):
            # Recursively merge dictionary child nodes key-by-key
            merged = dict(n1)
            for k, v in n2.items():
                if k in merged:
                    merged[k] = merge_nodes(merged[k], v)
                else:
                    merged[k] = v
            return merged
            
        else:
            # Fallback for structural type mismatches (rare edge-cases)
            if isinstance(n1, dict) and is_leaf_node(n2):
                merged = dict(n1)
                if "_unclustered" in merged:
                    merged["_unclustered"] = list(dict.fromkeys(merged["_unclustered"] + n2))
                else:
                    merged["_unclustered"] = n2
                return merged
            elif is_leaf_node(n1) and isinstance(n2, dict):
                merged = dict(n2)
                if "_unclustered" in merged:
                    merged["_unclustered"] = list(dict.fromkeys(merged["_unclustered"] + n1))
                else:
                    merged["_unclustered"] = n1
                return merged
        return n1

    # Determine total labels we need to generate to manage progress bar linear updates
    def count_total_jobs(node):
        if is_leaf_node(node):
            return 1
        children = get_node_children(node)
        if children is not None:
            return 1 + sum(count_total_jobs(child) for _, child in children)
        return 0

    # Format the root-level iterable children
    root_children = list(groups_final.items()) if isinstance(groups_final, dict) else groups_final
    total_jobs = sum(count_total_jobs(node) for _, node in root_children)
    completed_jobs = 0

    def update_progress():
        nonlocal completed_jobs
        completed_jobs += 1
        if progress_callback:
            # Scale smoothly between 50% and 90% progress markers
            val = int(50 + (completed_jobs / total_jobs) * 40)
            progress_callback(val)

    # Recursive Tree Processing
    def label_node_recursive(lab: str, node):
        nonlocal completed_jobs
        if check_cancel and check_cancel():
            raise InterruptedError()

        # Rule 1: Noise remains noise
        if lab == "-1":
            update_progress()
            return "Misc", node

        # Rule 2: Leaf Nodes (Files list) -> Label by file contents
        if is_leaf_node(node):
            text_context = ""
            page_img_index = 0
            img_index = 0
            img_context = []
            temp_img_dir = os.path.join(dir_path, f"cluster_{lab}_img")

            try:
                for file_name in node:
                    if check_cancel and check_cancel():
                        raise InterruptedError()
                    
                    file_path = os.path.join(dir_path, file_name)

                    match Path(file_path).suffix.lower():
                        case '.png' | '.jpg' | '.jpeg' | '.avif' | '.bmp' | '.tiff'| '.webp':
                            # file is an image
                            if img_index >= 5: # Only allow 5 normal images
                                continue
                            img_context.append(file_path)
                            img_index += 1
                        case _:
                            sample = get_file_sample(file_path)
                            if isinstance(sample, str) and sample != "(Unsupported format)":
                                text_context += f"--- File: {file_name} ---\n"
                                text_context += f"Content Sample: {sample}\n\n"
                            elif isinstance(sample, list):  # Returned page-images list
                                os.makedirs(temp_img_dir, exist_ok=True)
                                for page in sample:
                                    if check_cancel and check_cancel():
                                        raise InterruptedError()
                                    image_name = f"{page_img_index}.png"
                                    full_output_path = os.path.join(temp_img_dir, image_name)
                                    page.save(full_output_path, "PNG")
                                    img_context.append(full_output_path)
                                    page_img_index += 1
                img_only = (len(text_context) == 0)
                if not img_only:
                    prompt = (
                        f"You are an expert data cataloger. Review the following text snippets and images from a group of files "
                        f"that belong to the same semantic cluster. Based on their actual content, generate a highly "
                        f"concise, precise 2-to-4 word topic label. Do not include introductory text, do not use quotes, "
                        f"and return ONLY the label.\n\nCluster Content:\n{text_context}\nTopic Label:"
                    )
                else:
                    prompt = (
                        f"You are an expert data cataloger. Review the following images from a group of files "
                        f"that belong to the same semantic cluster. Based on their actual content, generate a highly "
                        f"concise, precise 2-to-4 word topic label. Do not include introductory text, do not use quotes, "
                        f"and return ONLY the label.\nTopic Label:"
                    )

                label = generate_llm_label(prompt, img_context, img_only=img_only, check_cancel=check_cancel)
                if not label or label == "Unlabeled_Cluster":
                    label = f"Cluster_{lab}"

            finally:
                if os.path.exists(temp_img_dir):
                    shutil.rmtree(temp_img_dir)

            if status_callback:
                status_callback(f"Labeled Leaf Cluster {lab} -> '{label}'")
            update_progress()
            return label, node

        # Rule 3: Parent Nodes -> Label recursively by child directory names
        children = get_node_children(node)
        if children is not None:
            labeled_children = {}

            for sub_lab, sub_node in children:
                child_label, child_labeled_node = label_node_recursive(sub_lab, sub_node)
                
                # Merge nodes dynamically if we find a duplicate child label
                if child_label in labeled_children:
                    if status_callback:
                        status_callback(f"Duplicate label found! Merging nodes under child directory: '{child_label}'")
                    labeled_children[child_label] = merge_nodes(labeled_children[child_label], child_labeled_node)
                else:
                    labeled_children[child_label] = child_labeled_node

            # Extract the unique, finalized list of child labels for prompt context
            child_labels_list = list(labeled_children.keys())

            # Construct parent directory context
            children_str = "\n".join([f"- {name}" for name in child_labels_list])
            parent_prompt = (
                f"You are an expert data cataloger. Review the following folder names which are sub-directories "
                f"within a parent directory. Based on these folder names, generate a highly concise, precise "
                f"2-to-4 word topic label for the parent directory. Do not include introductory text, do not use quotes, "
                f"and return ONLY the label.\n\nSub-directories:\n{children_str}\nParent Directory Label:"
            )

            parent_label = generate_llm_label(parent_prompt, check_cancel=check_cancel)
            if not parent_label or parent_label == "Unlabeled_Cluster":
                parent_label = f"Folder_{lab}"

            if status_callback:
                status_callback(f"Labeled Parent Directory {lab} -> '{parent_label}' based on children: {child_labels_list}")
            update_progress()
            return parent_label, labeled_children

        return f"Cluster_{lab}", node

    try:
        # Run recursive traversal at the top-level items
        labeled_tree = {}
        for root_lab, root_node in root_children:
            top_label, top_labeled_node = label_node_recursive(root_lab, root_node)
            
            # Merge nodes dynamically if duplicate top-level folder names are generated
            if top_label in labeled_tree:
                if status_callback:
                    status_callback(f"Duplicate label found! Merging nodes under root directory: '{top_label}'")
                labeled_tree[top_label] = merge_nodes(labeled_tree[top_label], top_labeled_node)
            else:
                labeled_tree[top_label] = top_labeled_node

        return labeled_tree
    finally:
        unload_llm()

def MoveFiles(rootdir_path: str, groups_final: dict, subdir_path:str = None, status_callback=None, progress_callback=None, check_cancel=None):
    if not groups_final:
        return
    
    if not subdir_path: # initial call
        subdir_path = rootdir_path

    total_labels = len(groups_final)
    for idx, lab in enumerate(groups_final.keys()):
        if check_cancel and check_cancel(): raise InterruptedError()

        cluster_dir = subdir_path
        if os.path.basename(os.path.normpath(subdir_path)) != lab: 
            #if the parent directory does not have the same name as the child directory
            cluster_dir = os.path.join(subdir_path, lab)

        if isinstance(groups_final[lab], dict): # Current label is a parent directory
            os.makedirs(cluster_dir, exist_ok=True)
            num_of_subdir = len(groups_final[lab].keys())
            # the following code avoids unnecessary sub directories (pnly 1 subdirectory inside a directory)
            if num_of_subdir > 1:
                MoveFiles(rootdir_path, groups_final[lab], subdir_path=cluster_dir)
            else:
                MoveFiles(rootdir_path, groups_final[lab], subdir_path=subdir_path)

        elif isinstance(groups_final[lab], list): # Current label is a leaf directory
            files = groups_final[lab]
            
            os.makedirs(cluster_dir, exist_ok=True)

            for item in files:
                if check_cancel and check_cancel(): raise InterruptedError()
                src_path = os.path.join(rootdir_path, item)
                dest_path = os.path.join(cluster_dir, item)
                try:
                    if os.path.exists(src_path):
                        os.rename(src_path, dest_path)
                except Exception as e:
                    msg = f"Error moving {item} to {lab}: {e}"
                    print(msg)
                    if status_callback: status_callback(msg)
                
        if progress_callback:
            progress_callback(int(90 + ((idx + 1) / total_labels) * 10))