# Overseer
## Description
Overseer is a project that aims to build a self hosted, AI-powered file management system. It is programmed with Python, and uses LLM platforms such as Ollama or OpenRouter. This project is still in preliminary stages of development.

## Objectives
Below are the objectives that Overseer aims to achieve.
* Understand the content of the files inside a directory and intelligently sort them into folders in a semantically relevant manner.
    - The user can control the extent of AI autonomy in this procedure, eg. the directory where it sorts the files, its permission to view the actual contents of files etc.
* Take account of user interactions with their files in its file organisation, for example making commonly accessed files have a lower 'depth' in the directory stucture, similar to a cache system.
* Feature a chatbot which utilises Retrieval Augmented Generation, allowing users to Q&A about information regarding the files in the directory.

## Required Software (may expand later)
- User interface: Python (Tkinter) *may switch to PyQt or PyQt in later stages
- Database Management: sqlite3 for relational database, ChromaDB for vector database (RAG)
- LLM Providers: Ollama, OpenRouter

## Related repos
This project is inspired by QiuYannnn's Local-File-Organizer - https://github.com/QiuYannnn/Local-File-Organizer.
