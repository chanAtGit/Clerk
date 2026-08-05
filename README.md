# Clerk
## Description
Clerk is a project that aims to build a self-hosted, AI-powered file management system. It is programmed with Python and uses popular open source LLMs such as Gemma 4, and embedding models like Qwen3 VL Embedding. 

## Objectives
Below are the objectives that Clerk aims to achieve.
* Understand the content of the files inside a directory and intelligently sort them into folders in a semantically relevant manner.
    - The user can control the extent of AI autonomy in this procedure, eg, the directory where it sorts the files, its permission to view the actual contents of files, etc.
    - Two file sorting options: Sort into existing folders, or sort into generated folders
* Feature a chatbot which utilises Retrieval Augmented Generation, allowing users to Q&A about information regarding the files in the directory.

## Required Software (may expand later)
- User interface: Python (Tkinter) *may switch to PyQt or PySide in later stages
- Database: sqlite3 for relational database, chroma for vector database, diskcache for caching 
- LLM Providers: Transformer, SentenceTransfromer, Ollama

## Related repos
This project is inspired by QiuYannnn's Local-File-Organizer - https://github.com/QiuYannnn/Local-File-Organizer.

## How to run
```
pip install -r requirements.txt
python main.py
```