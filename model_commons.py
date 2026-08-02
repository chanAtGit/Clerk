import os
import gc
import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForMultimodalLM

embedding_model = None 
processor = None
llm = None

def models_init():
    """Initialize Hugging Face login and paths. Models are loaded dynamically later."""
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    if not huggingface_token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")

    try:
        login(token=huggingface_token)
    except Exception as e:
        print(f"Error encountered when logging into huggingface: {e}. Using offline mode.")

## USEFUL HELPER FUNCTION
def clear_vram():
    """Forces Python garbage collection and clears PyTorch CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()

## EMBEDDING MODEL FUNCTIONS
def load_embedding_model(model_id: str):
        global embedding_model
        model_last_name = model_id.split('/')[-1].lower()
        local_path = f"models/{model_last_name}"
        if embedding_model is None:
            print("Loading embedding model into VRAM...")
            if os.path.exists(local_path):
                print("Loading local model...")
                embedding_model = SentenceTransformer(local_path, device="cuda", local_files_only=True)
            else:
                embedding_model = SentenceTransformer(model_id, device="cuda")
                embedding_model.save(local_path)

def unload_embedding_model():
        global embedding_model
        if embedding_model is not None:
            print("Unloading embedding model from VRAM...")
            del embedding_model
            embedding_model = None
            clear_vram()

def embedding_encode(*args, **kwargs):
        """Delegates encoding directly to the underlying SentenceTransformer model."""
        global embedding_model
        if embedding_model is None:
            raise RuntimeError(
                "Model is not loaded. Call load_embedding_model() before encoding."
            )

        return embedding_model.encode(*args, **kwargs)

## LLM FUNCTIONS
def load_llm(model_id: str):
    global processor, llm
    if llm is None:
        model_last_name = model_id.split('/')[-1].lower()
        local_path = f"models/{model_last_name}"

        # Check if the model has already been saved locally
        if os.path.exists(local_path):
            print(f"Loading LLM offline from '{local_path}'...")
            processor = AutoProcessor.from_pretrained(local_path)
            llm = AutoModelForMultimodalLM.from_pretrained(
                local_path, 
                dtype=torch.bfloat16,
                local_files_only=True
            ).to("cuda")
        else:
            print(f"Downloading LLM from Hugging Face ({model_id})...")
            processor = AutoProcessor.from_pretrained(model_id)
            llm = AutoModelForMultimodalLM.from_pretrained(
                model_id, 
                dtype=torch.bfloat16
            )
            
            print(f"Saving model and processor locally to '{local_path}'...")
            processor.save_pretrained(local_path)
            llm.save_pretrained(local_path)
            
            llm = llm.to("cuda")

def unload_llm():
    global processor, llm
    if llm is not None:
        print("Unloading LLM from VRAM...")
        del llm
        del processor
        llm = None
        processor = None
        clear_vram()

def llm_chat(messages: list, images: list|None, max_tokens:int = 512):
    global llm
    if llm is None:
        raise RuntimeError(
            "Model is not loaded. Call load_llm() before chatting."
        )

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                 
    if images:
        inputs = processor(text=text_prompt, images=images, return_tensors="pt")
    else:
        inputs = processor(text=text_prompt, return_tensors="pt")
                     
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                 
    with torch.no_grad():
        generated_ids = llm.generate(**inputs, max_new_tokens=max_tokens)
                     
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    return response