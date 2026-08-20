import os
import shutil
# import torch
import numpy as np
import re
import ollama
from pathlib import Path
from huggingface_hub import login
# from transformers import AutoProcessor, AutoModelForMultimodalLM
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from PIL import Image
from collections import defaultdict

from model_commons import unload_embedding_model, load_llm, unload_llm, llm_chat
from file_embeddings import get_file_mean_embeddings, get_directory_mean_embeddings, get_file_sample

# Global variable declarations
processor = None
llm = None
LLM_NAME = "google/gemma-4-E2B-it"

def optimal_clustering(embeddings:list, recursive:bool=True, status_callback = None) -> list:
    THRESHOLD = 0.45
    pca = PCA(n_components = min(6, len(embeddings)), random_state=42)
    data = pca.fit_transform(np.array(embeddings))
    cluster_values = np.arange(1, int(data.shape[0]))
    
    best_clusters = None
    best_score = -1
    best_labels = None

    for cluster in cluster_values:
        model = AgglomerativeClustering(n_clusters=cluster, metric="cosine", linkage='average')
        # cosine metric is the best for high dimensional embeddings
        labels = model.fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:
            core_samples_mask = labels != -1
            search_labels = labels[core_samples_mask]
            
            if len(set(search_labels)) > 1:
                score = silhouette_score(data[core_samples_mask], search_labels, metric="cosine")

                if score > best_score: 
                    best_score = score 
                    best_clusters = cluster
                    best_labels = labels

    if best_score > THRESHOLD:
        print(f"Best clusters: {best_clusters}, score: {best_score:.3f}")
        best_labels = [str(lab + 1) for lab in best_labels] # convert the labels from int to str. First label is "1"
        
        if recursive:
            for cluster_lab in range(1, best_clusters + 1):
                cluster_embeddings = []
                for i in range(len(embeddings)):
                    if best_labels[i] == str(cluster_lab):
                        # get all embeddings that belong to a cluster
                        cluster_embeddings.append(embeddings[i])
                
                if len(cluster_embeddings) < 3: # skipping over clusters that less than 3 elements
                    continue
                
                # generate sub labels
                sub_labels = optimal_clustering(cluster_embeddings)
                if sub_labels:
                    # append sub labels to labels (eg. A label would be 1.1, meaning it belongs to cluster 1 and its subcluster 1)
                    current = 0
                    for i in range(len(best_labels)):
                        if best_labels[i] == str(cluster_lab):
                            best_labels[i] = best_labels[i] + "." + sub_labels[current]
                            current += 1

        return best_labels
    else:
        msg = f"Best silhouette score {best_score:.3f} is below {THRESHOLD}. No clusters generated."
        print(msg)
        if status_callback: status_callback(msg)
        return None

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

def SemanticClustering(dir_path: str, recursive:bool=True, status_callback=None, progress_callback=None, check_cancel=None) -> dict:
    '''Group files into semantic clusters using VL embedding model and hierarchical clustering'''
    if check_cancel and check_cancel(): raise InterruptedError()
    
    try:
        file_embeddings_dict, embedding_model_used = get_file_mean_embeddings(dir_path, status_callback, progress_callback, check_cancel)

        if not file_embeddings_dict:
            return None
        
        if check_cancel and check_cancel(): raise InterruptedError()
        file_names = list(file_embeddings_dict.keys())
        mean_embeddings = list(file_embeddings_dict.values())

        labels_final = optimal_clustering(mean_embeddings, recursive, status_callback)
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
            return None
    finally:
        # Guarantee the embedding model unloads even if user cancels
        if embedding_model_used: unload_embedding_model()


def SortIntoFolders(dir_path: str, status_callback=None, progress_callback=None, check_cancel=None) -> dict:
    '''Sort files into existing folders in the given directory'''
    if check_cancel and check_cancel(): raise InterruptedError()

    dir_list = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    if not dir_list:
        if status_callback: status_callback("No directories found.")
        return None

    try:
        file_embeddings_dict, _ = get_file_mean_embeddings(dir_path, status_callback, progress_callback, check_cancel)
        file_embeddings = np.array(list(file_embeddings_dict.values()))

        if check_cancel and check_cancel(): raise InterruptedError()

        dir_embeddings_dict = get_directory_mean_embeddings(dir_list)
        dir_embeddings = np.array(list(dir_embeddings_dict.values()))

        # normalise each embedding 
        file_embeddings = file_embeddings / np.linalg.norm(file_embeddings, axis=1, keepdims=True)
        dir_embeddings = dir_embeddings / np.linalg.norm(dir_embeddings, axis=1, keepdims=True)

        def softmax(x):
            x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
            x_exp = x_exp / np.sum(x_exp, axis=-1, keepdims=True)
            x_exp = np.round(x_exp, decimals=2) # round the softmax probability to 2 decimal places
            return x_exp
        
        similarity_matrix = softmax(np.dot(file_embeddings, dir_embeddings.T)) # get softmax matrix with cosine similarity as logit (number of files * number of directories)
        max_sim_ind = np.argmax(similarity_matrix, axis = 1) # index of most probable directory per file
        max_sim = np.max(similarity_matrix, axis = 1) # get maximum probability per file (number of files * 1)
        max_counts = np.sum(similarity_matrix == max_sim.reshape(-1,1), axis=1) # Count how many times the maximum value appears in each row
        max_sim_ind = np.where(max_counts > 1, -1, max_sim_ind) # If count is greater than 1, replace index with -1

        file_names = list(file_embeddings_dict.keys())
        dir_names = list(dir_embeddings_dict.keys())
        dir_file_dict = defaultdict(list)
        for i in range(len(file_names)):
            if max_sim_ind[i] == -1: # if the maximum probability has appeared twice (correspond to more than 1 directory)
                continue # skip
            dir_file_dict[dir_names[max_sim_ind[i]]].append(file_names[i])

        print(f"Max sim: {np.max(max_sim)}, Min sim: {np.min(max_sim)}")

        return dir_file_dict
    finally:
        # Embedding model is always loaded in this function, so it must be unloaded
        unload_embedding_model()

# --- Helper 2: LLM Inference Sequence ---
def generate_llm_label(prompt: str, img_paths: list = None, online: bool = False, img_only:bool = False, check_cancel=None) -> str:
    """Executes prompt formatting and tensor processing to query the LLM model."""
    global processor, llm
    try:
        if not online:
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
            label = llm_chat(messages, loaded_images, max_tokens=40)
            
            # Strip filesystem-incompatible characters
            label = re.sub(r'[\\/*?:"<>|]', "", label)
            return label if label else "Unlabeled_Cluster"
        else:
            # Online mode with Ollama API
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
            label = response["message"]["content"]
            label = re.sub(r'[\\/*?:"<>|]', "", label)
            return label if label else "Unlabeled_Cluster"
    except Exception as e:
        print(f"Error executing LLM generation: {e}")
        return "Unlabeled_Cluster"


# --- Core Orchestration & Recursive Processing ---
def AutoLabelClusters(dir_path: str, groups_final: dict | list, online:bool = False, status_callback=None, progress_callback=None, check_cancel=None) -> dict:
    if not groups_final:
        return {}
        
    msg = "Analyzing clusters via LLM content descriptions..."
    print(msg)
    if status_callback: status_callback(msg)
    
    if not online:
        load_llm(LLM_NAME)

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
                        f"and return ONLY the label. \nIf you are unsure, just return 'Unlabeled_Cluster'."
                        f"\nCluster Content:\n{text_context}\nTopic Label:"
                    )
                else:
                    prompt = (
                        f"You are an expert data cataloger. Review the following images from a group of files "
                        f"that belong to the same semantic cluster. Based on their actual content, generate a highly "
                        f"concise, precise 2-to-4 word topic label. Do not include introductory text, do not use quotes, "
                        f"and return ONLY the label.\nIf you are unsure, just return 'Unlabeled_Cluster'. \nTopic Label:"
                    )

                label = generate_llm_label(prompt, img_context, online=online, img_only=img_only, check_cancel=check_cancel)
                if not label or label == "Unlabeled_Cluster":
                    label = f"Folder_{lab}"

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

            parent_label = generate_llm_label(parent_prompt, online=online, check_cancel=check_cancel)
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

def print_groups(groups_final: dict, num_of_indents:int = 0, sort_into_existing:bool = False, print_to_widget = None):
    misc_files = []
    for folder in groups_final.keys():
        if isinstance(groups_final[folder], list):
            if not sort_into_existing and len(groups_final[folder]) == 1: 
                # if the folder only contains 1 file, the file is moved to the misc files
                misc_files.append(groups_final[folder][0])
                continue

            # print(" " * num_of_indents + f"Folder: {folder}")
            if print_to_widget: print_to_widget(" " * num_of_indents + f"Folder: {folder}")

            for files in groups_final[folder]:
                # print(" " * num_of_indents + f"+ {files}")
                if print_to_widget: print_to_widget(" " * num_of_indents + f"➔ {files}")

            # print()
            if print_to_widget: print_to_widget('')
        elif isinstance(groups_final[folder], dict):
            # print(" " * num_of_indents + f"Folder: {folder}")
            if print_to_widget: print_to_widget(" " * num_of_indents + f"Folder: {folder}")

            print_groups(groups_final[folder], num_of_indents + 4, sort_into_existing, print_to_widget)

    # Print files in Misc Folder
    if not sort_into_existing and misc_files:
        # print(" " * num_of_indents + f"Folder: Misc")
        if print_to_widget: print_to_widget(" " * num_of_indents + f"Folder: Misc")

        for files in misc_files:
            # print(" " * num_of_indents + f"+ {files}")
            if print_to_widget: print_to_widget(" " * num_of_indents + f"➔ {files}")

        # print()
        if print_to_widget: print_to_widget('')

def MoveFiles(rootdir_path: str, groups_final: dict, subdir_path:str = None, sort_into_existing:bool = False, status_callback=None, progress_callback=None, check_cancel=None):
    '''Move files into folders given a dictionary object'''
    if not groups_final:
        return
    
    if not subdir_path: # initial call
        subdir_path = rootdir_path

    total_labels = len(groups_final)
    misc_files = []
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

            # The generation of a Misc folder is only exclusive to the sorting mode with generating new folder
            if not sort_into_existing and len(files) == 1:
                misc_files.append(files[0]) # add to misc_files
                continue # skip this lead directory
            
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

    if not sort_into_existing and len(misc_files) > 0:
        misc_dir = os.path.join(subdir_path, "Misc")
        os.makedirs(misc_dir, exist_ok=True)
        for item in misc_files:
            if check_cancel and check_cancel(): raise InterruptedError()
            src_path = os.path.join(rootdir_path, item)
            dest_path = os.path.join(misc_dir, item)
            try:
                if os.path.exists(src_path):
                    os.rename(src_path, dest_path)
            except Exception as e:
                msg = f"Error moving {item} to Misc: {e}"
                print(msg)
                if status_callback: status_callback(msg)