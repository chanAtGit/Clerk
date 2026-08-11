from file_sort import LLM_NAME
from model_commons import load_llm, unload_llm, llm_chat

def get_chat_response(prompt: str) -> str:
    '''Get chatbot message when using ClerkBot module'''
    try:
        # TODO: Add more advanced features. This is just a placeholder function with no chat memory or RAG.
        load_llm(LLM_NAME)

        content_list = []
        content_list.append({"type": "text", "text": prompt})         
        messages = [{"role": "user", "content": content_list}]

        response = llm_chat(messages, max_tokens = 1024)

        return response        
    except Exception as e:
        print(f"An error occured while chatting with LLM: {e}")
    finally:
        unload_llm()