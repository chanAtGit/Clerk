import os
import shutil
import torch
import gc
from pathlib import Path
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForMultimodalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import re, unicodedata, collections
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

def clear_vram():
    """Forces Python garbage collection and clears PyTorch CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()

def load_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model into VRAM...")
        embedding_model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device="cuda")

def unload_embedding_model():
    global embedding_model
    if embedding_model is not None:
        print("Unloading embedding model from VRAM...")
        del embedding_model
        embedding_model = None
        clear_vram()

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

def semantics_init():
    """Initialize Hugging Face login and paths. Models are loaded dynamically later."""
    global poppler_path
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    poppler_path = os.getenv("POPPLER_PATH")  

    if not huggingface_token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")
    if not poppler_path:
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
        model = AgglomerativeClustering(n_clusters=cluster, linkage='average')
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
                sub_labels = optimal_clustering(cluster_embeddings, at_root=False, recursive=recursive)
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
            images = convert_from_path(pdf_path, poppler_path=poppler_path)
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
            images = convert_from_path(temp_pdf_path)
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

        mean_embedding = None
        match Path(file_path).suffix.lower():
            case '.pdf':
                mean_embedding = generate_pdf_mean_embedding(file_path, status_callback)
            case '.docx':
                mean_embedding = generate_docx_mean_embedding(file_path, status_callback)
            case '.png' | '.jpg' | '.jpeg' | '.bmp' | '.tiff':
                mean_embedding = embedding_model.encode(Image.open(file_path), convert_to_numpy=True)
        
        if mean_embedding is not None:
            mean_embeddings[file_name] = mean_embedding
        
        if progress_callback:
            progress_callback(int(((idx + 1) / total_files) * 50))

    if len(mean_embeddings) == 0:
        return None
    
    return mean_embeddings

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
        groups_final = collections.defaultdict(list)
        if labels_final:
            for i, lab in enumerate(labels_final):
                groups_final[lab].append(file_names[i])

            groups_final = sorted(
                groups_final.items(), 
                key=lambda x: (x[0] == '-1', [int(part) for part in x[0].split('.')])
            )

            return groups_final 
        else:
            return None
    finally:
        # Guarantee the embedding model unloads even if user cancels
        unload_embedding_model()

def AutoLabelClusters(dir_path: str, groups_final: list, status_callback=None, progress_callback=None, check_cancel=None) -> dict:
    global processor, llm
    if not groups_final:
        return {}
    
    msg = "Analyzing clusters via LLM content descriptions..."
    print(msg)
    if status_callback: status_callback(msg)
    
    load_llm()

    try:
        file_contents = {}
        for lab, items in groups_final:
            if lab == "-1":
                continue
            for file_name in items:
                if check_cancel and check_cancel():
                    raise InterruptedError()
                file_path = os.path.join(dir_path, file_name)
                try:
                    match Path(file_path).suffix.lower():
                        case '.pdf':
                            loader = PyPDFLoader(file_path)
                        case '.docx':
                            loader = UnstructuredWordDocumentLoader(file_path, mode="single")
                    docs = loader.load()
                    snippet_docs = docs[:3] 
                    chunks = text_splitter.split_documents(snippet_docs)

                    if chunks:
                        text_sample = " ".join([str(c.page_content) for c in chunks[:2]])
                        file_contents[file_name] = text_sample[:1200]
                    else:
                        images = convert_from_path(file_path, poppler_path=poppler_path)
                        file_contents[file_name] = images
                except Exception:
                    file_contents[file_name] = "(Could not read file content)"

        labeled_groups = {}
        total_groups = len(groups_final)

        for idx, (lab, items) in enumerate(groups_final):
            if check_cancel and check_cancel(): 
                raise InterruptedError()
                
            if lab == "-1":
                labeled_groups["Misc"] = items
                continue
                
            if len(items) == 1:
                labeled_groups[f"Single_Files_Cluster_{lab}"] = items
                continue

            cluster_context = ""
            img_index = 0
            img_context = []
            temp_img_dir = os.path.join(dir_path, f"cluster_{lab}_img")

            for file_name in items:
                if isinstance(file_contents.get(file_name, ''), str):
                    cluster_context += f"--- File: {file_name} ---\n"
                    cluster_context += f"Content Sample: {file_contents.get(file_name, '')}\n\n"
                else:
                    for page in file_contents[file_name]:
                        if check_cancel and check_cancel():
                            if os.path.exists(temp_img_dir): shutil.rmtree(temp_img_dir)
                            raise InterruptedError()
                        if not os.path.exists(temp_img_dir):
                            os.mkdir(temp_img_dir)
                        image_name = f"{img_index}.png"
                        full_output_path = os.path.join(temp_img_dir, image_name)
                        page.save(full_output_path, "PNG")
                        img_context.append(full_output_path)
                        img_index += 1
            
            prompt = f'''You are an expert data cataloger. Review the following text snippets from a group of files that belong to the same semantic cluster. Based on their actual content, generate a highly concise, precise 2-to-4 word topic label. Do not include introductory text, do not use quotes, and return ONLY the label.\n\nCluster Content:\n{cluster_context}\nTopic Label:'''
            
            try:
                # 1. Structure the conversation message format for Transformers
                content_list = []
                loaded_images = []
                
                if img_context:
                    for img_path in img_context:
                        content_list.append({"type": "image"})
                        loaded_images.append(Image.open(img_path).convert("RGB"))
                        
                content_list.append({"type": "text", "text": prompt})
                
                messages = [
                    {
                        "role": "user",
                        "content": content_list
                    }
                ]
                
                # 2. Render input tensor representations using the model's processor wrapper
                text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                if loaded_images:
                    inputs = processor(text=text_prompt, images=loaded_images, return_tensors="pt")
                else:
                    inputs = processor(text=text_prompt, return_tensors="pt")
                    
                # Safe move to strictly "cuda" device 
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                # 3. Model Inference Sequence Execution
                with torch.no_grad():
                    generated_ids = llm.generate(**inputs, max_new_tokens=40)
                    
                # Extract generated tokens (excluding prompt token array overhead)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)]
                label = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
                label = re.sub(r'[\\/*?:"<>|]', "", label)
                
                if not label:
                    label = f"Cluster_{lab}"
            except Exception as e:
                print(f"Error generating label for cluster {lab}: {e}")
                label = f"Cluster_{lab}"

            if label in list(labeled_groups.keys()): # if the label already exists, append the items to that label
                labeled_groups[label].extend(items)
            else:
                labeled_groups[label] = items

            msg = f"Generated label: '{label}' for group {lab}"
            print(msg)
            if status_callback: status_callback(msg)
            
            if os.path.exists(temp_img_dir):
                shutil.rmtree(temp_img_dir)
                
            if progress_callback:
                progress_callback(int(50 + ((idx + 1) / total_groups) * 40))

        return labeled_groups
    finally:
        # Guarantee the LLM unloads even if user cancels
        unload_llm()

def MoveFiles(dir_path: str, groups_final: dict, status_callback=None, progress_callback=None, check_cancel=None):
    if not groups_final:
        return
    
    total_labels = len(groups_final)
    for idx, (lab, items) in enumerate(groups_final.items()):
        if check_cancel and check_cancel(): raise InterruptedError()
        if not lab or not items or len(items) == 1:
            continue
        cluster_dir = os.path.join(dir_path, lab)
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir, exist_ok=True)
        
        for item in items:
            if check_cancel and check_cancel(): raise InterruptedError()
            src_path = os.path.join(dir_path, item)
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