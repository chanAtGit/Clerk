import os
import shutil
import uuid
from PIL import Image

from file_sort import LLM_NAME
from file_embeddings import convert_pdf_to_img
from model_commons import load_llm, unload_llm, llm_chat

def get_chat_response(prompt: str, retrieved_context: list, dir_path: str) -> str:
    '''Get chatbot message when using ClerkBot module'''
    try:
        # TODO: Add more advanced features. This is just a placeholder function with no chat memory.

        content_list = []
        text_context = []
        img_context = []
        non_text_dir = str(uuid.uuid4().hex) # generate id for directory storing images from documents that are not text split

        if retrieved_context is not None and len(retrieved_context) != 0:
            # 2 types of context: text and img
            for document, metadata, distance in zip(retrieved_context["documents"][0], retrieved_context["metadatas"][0], retrieved_context["distances"][0]):
                if distance > 0.6:
                    continue # skip retrieved documents with greater vector distance
                file_path = os.path.join(dir_path, metadata["file_name"])
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # retrieved an image
                    if os.path.exists(file_path):
                        content_list.append({"type": "image", "image": file_path})
                        img_context.append(Image.open(file_path).convert("RGB"))
                elif document == None:
                    # retrieved a file that cannot be text splitted (need to convert to image first)
                    os.makedirs(non_text_dir, exist_ok=True)
                    file_images = convert_pdf_to_img(file_path)
                    temp_image_name = f"{uuid.uuid4().hex}.png" # generate random file name
                    full_temp_path = os.path.join(non_text_dir, temp_image_name)
                    file_images[int(metadata["page"]) - 1].save(full_temp_path, "PNG") # save the image page

                    content_list.append({"type": "image", "image": full_temp_path})
                    img_context.append(Image.open(full_temp_path).convert("RGB"))
                else:
                    # retrieved a file that can be text splitted (has text chunk)
                    text_context.append({
                        "file_name": metadata["file_name"],
                        "text_chunk": document,
                        "page_number": metadata["page"]
                    })

            augmented_prompt = f'''
                Consider the images above as context, as well as these retrieved document context below. \n
                Document context: \n
                {text_context} \n
                Answer the user's prompt. You mustreference the file name and page number when you use information from a text chunk.\n
                User prompt: {prompt}
                '''
            content_list.append({"type": "text", "text": augmented_prompt})
        else:
            content_list.append({"type": "text", "text": prompt})  
         
        messages = [{"role": "user", "content": content_list}]

        load_llm(LLM_NAME)
        response = llm_chat(messages, img_context, max_tokens = 1024)

        return response        
    except Exception as e:
        print(f"An error occured while chatting with LLM: {e}")
    finally:
        unload_llm()
        if os.path.exists(non_text_dir):
            shutil.rmtree(non_text_dir)