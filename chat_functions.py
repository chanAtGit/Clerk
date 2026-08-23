import os
import shutil
import uuid
import ollama
from PIL import Image

from file_sort import LLM_NAME
from file_embeddings import convert_pdf_to_img
from model_commons import load_llm, unload_llm, llm_chat

def get_new_chat_title(first_question: str, new_title: list, creating_new_chat: bool = True, online: bool = False):
    '''Create a title for the newly created chat. 
    This function is to be executed in a thread without using ThreadPoolExecutor.
    The new title is to be stored in the initially empty list new_title as a Thread Safe operation.
    '''
    if not creating_new_chat:
        return

    try:
        system_prompt: str = f'''
            You are to create a title for a newly created chat session based on this first prompt asked by the user. \n
            IMPORTANT: Only respond with the title. Keep the title within 5 words.
        '''
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_question}
        ]

        if not online:
            load_llm(LLM_NAME)
            title = llm_chat(messages)
        else:
            output = ollama.chat(
                model="gemma4:cloud",  # Or use "gemma4:31b-cloud" for the dense model
                messages=messages
            )
            title = output['message']['content']
        print(f"Created title for new chat session: {title}")
        new_title.append(title)
    except Exception as e:
        print(f"An error occured while chatting with LLM: {e}")
    finally:
        if not online: unload_llm()

def get_chat_response(prompt: str, retrieved_context: list, dir_path: str, online: bool = False) -> str:
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
                # if distance > 0.6:
                #     continue # skip retrieved documents with greater vector distance
                file_path = os.path.join(dir_path, metadata["file_name"])
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # retrieved an image
                    if os.path.exists(file_path):
                        content_list.append({"type": "image", "image": file_path})
                        if not online:
                            img_context.append(Image.open(file_path).convert("RGB"))
                        else:
                            img_context.append(file_path) # for online LLM, just pass the file path
                elif document == None:
                    # retrieved a file that cannot be text splitted (need to convert to image first)
                    os.makedirs(non_text_dir, exist_ok=True)
                    file_images = convert_pdf_to_img(file_path)
                    temp_image_name = f"{uuid.uuid4().hex}.png" # generate random file name
                    full_temp_path = os.path.join(non_text_dir, temp_image_name)
                    file_images[int(metadata["page"]) - 1].save(full_temp_path, "PNG") # save the image page

                    content_list.append({"type": "image", "image": full_temp_path})
                    if not online:
                        img_context.append(Image.open(full_temp_path).convert("RGB"))
                    else:
                        img_context.append(full_temp_path)

                    text_context.append({
                        "file_name": metadata["file_name"],
                        "text_chunk": document,
                        "page_number": metadata["page"]
                    })
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
                User prompt: {prompt}
                '''
            content_list.append({"type": "text", "text": augmented_prompt})
        else:
            content_list.append({"type": "text", "text": prompt})  

        system_prompt = '''
                Answer the user's prompt. You must reference the file name and page number when you use information from a text chunk.\n
                If there are no document or image context, mention that the files in the current directory do not provide any relevant information.
                Instead, use your general knowledge to answer the question.\n
                ''' 
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if not online:
            messages.append({
                "role": "user", 
                "content": content_list
            })
            load_llm(LLM_NAME)
            response = llm_chat(messages, img_context, max_tokens = 1024)
        else:
            messages.append({
                "role": "user", 
                "content": augmented_prompt,
                "images": img_context # Pass the path string or bytes directly
            })
            output = ollama.chat(
                model="gemma4:cloud",  # Or use "gemma4:31b-cloud" for the dense model
                messages=messages
            )
            response = output['message']['content']

        return response        
    except Exception as e:
        print(f"An error occured while chatting with LLM: {e}")
        raise ValueError(e)
    finally:
        if not online: unload_llm()
        if os.path.exists(non_text_dir):
            shutil.rmtree(non_text_dir)