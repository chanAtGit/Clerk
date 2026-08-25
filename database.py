import sqlite3
import uuid
import chromadb
import numpy as np
import os

from datetime import datetime
from dataclasses import dataclass

from model_commons import load_embedding_model, unload_embedding_model, embedding_encode, EMBEDDING_MODEL_NAME

@dataclass
class Inquiry:
    prompt: str
    response: str | None
    chatSession_id: str
    dir_path: str

class ChatDB():
    def __init__(self):
        self.con = sqlite3.connect("chatData.db")
        self.cur = self.con.cursor()

        self.cur.execute("PRAGMA foreign_keys = ON;")

        # Create chatSessions table
        self.cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS chatSessions (
                chatSession_id TEXT PRIMARY KEY,
                name TEXT,
                created_date DATETIME
            )
            '''
        )
        # Create inquiries table with chatSession_id column included
        self.cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS inquiries (
                inquiry_id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                response TEXT,
                chatSession_id TEXT,
                dir_id TEXT,
                dir_name TEXT,
                created_date DATETIME,
                FOREIGN KEY (chatSession_id) REFERENCES chatSessions(chatSession_id) ON DELETE CASCADE
            )
            '''
        )
        self.con.commit()

        self.chroma_client = chromadb.PersistentClient(path="clerk_vectordb")
        print("DB: Init complete.")

    def create_chatsession(self, name: str) -> str:
        chatSession_id = uuid.uuid4().hex
        created_date = datetime.now()
        self.cur.execute(
            '''
            INSERT INTO chatSessions (chatSession_id, name, created_date)
            VALUES (?, ?, ?)
            ''',
            (chatSession_id, name, created_date)
        )
        self.con.commit()
        print(f"DB: Created chat session: {name}")
        return chatSession_id

    def create_inquiry(self, inquiry: Inquiry) -> str:
        inquiry_id = uuid.uuid4().hex
        dir_id = os.stat(inquiry.dir_path).st_ino  # Get the inode number of the directory as a unique identifier
        dir_name = os.path.basename(inquiry.dir_path)
        created_date = datetime.now()
        self.cur.execute(
            '''
            INSERT INTO inquiries (inquiry_id, prompt, response, chatSession_id, dir_id, dir_name, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (inquiry_id, inquiry.prompt, inquiry.response, inquiry.chatSession_id, dir_id, dir_name, created_date)
        )
        self.con.commit()
        print(f"DB: Created inquiry: {inquiry_id}")
        return inquiry_id

    def get_all_chatsessions(self) -> list:
        self.cur.execute(
            '''
            SELECT chatSession_id, name FROM chatSessions
            ORDER BY created_date DESC
            '''
        )
        print(f"DB: Got all chat sessions.")
        return self.cur.fetchall()

    def get_inquiries_from_session(self, chatSession_id: str) -> list:
        self.cur.execute(
            '''
            SELECT prompt, response, dir_name FROM inquiries
            WHERE chatSession_id = ?
            ORDER BY created_date ASC
            ''',
            (chatSession_id,)
        )
        print(f"DB: Got inquiries from session: {chatSession_id}")
        return self.cur.fetchall()

    def get_latest_inq_dir_from_session(self, chatSession_id: str) -> str:
        self.cur.execute(
            '''
            SELECT dir_name FROM inquiries
            WHERE chatSession_id = ?
            ORDER BY created_date DESC
            ''',
            (chatSession_id,)
        )
        print(f"DB: Got latest inquiry from session: {chatSession_id}")
        result = self.cur.fetchone()
        return result[0] if result else None

    def get_inquiries_for_llm_history(self, chatSession_id:str, dir_path: str) -> list:
        '''Get the two most recent inquiries from a specific chat session and directory for LLM history'''
        dir_id = os.stat(dir_path).st_ino  # Get the inode number of the directory as a unique identifier
        self.cur.execute(
                    '''
                    SELECT prompt, response, dir_name FROM inquiries
                    WHERE chatSession_id = ? AND dir_id = ?
                    ORDER BY created_date ASC
                    ''',
                    (chatSession_id, dir_id)
                )
        print(f"DB: Got inquiries for llm history from session: {chatSession_id}")
        return self.cur.fetchmany(2)        

    def update_chatsession_name_by_id(self, chatSession_id: str, new_name: str):
        self.cur.execute(
            '''
            UPDATE chatSessions
            SET name = ?
            WHERE chatSession_id = ?
            ''',
            (new_name, chatSession_id)
        )
        print(f"DB: Updated session {chatSession_id} with new name: {new_name}")
        self.con.commit()

    def update_inquiry_response_by_id(self, inquiry_id: str, response: str):
        self.cur.execute(
            '''
            UPDATE inquiries
            SET response = ?
            WHERE inquiry_id = ?
            ''',
            (response, inquiry_id)
        )
        print(f"DB: Updated inquiry {inquiry_id} with response.")
        self.con.commit()

    def delete_chatsession_by_id(self, chatSession_id: str):
        self.cur.execute(
            '''
            DELETE FROM chatSessions
            WHERE chatSession_id = ?
            ''',
            (chatSession_id,)
        )
        print(f"DB: Deleted chat session with id {chatSession_id}")
        self.con.commit()

    def add_file_chunk_embedding(self, file_id: int, file_name: str, page_num: int, embedding: np.ndarray, chunk: str = None):
        chunk_id = uuid.uuid4().hex
        collection = self.chroma_client.get_or_create_collection(name="file_embeddings")
        collection.add(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "file_id": file_id,
                "file_name": file_name,
                "page": page_num
            }]
        )

    def retrieve_file_chunk(self, prompt: str, file_id_list: list):
        try:
            collection = self.chroma_client.get_or_create_collection(name="file_embeddings")
            load_embedding_model(EMBEDDING_MODEL_NAME)

            query_vector = embedding_encode(prompt)
            results = collection.query(
                query_embeddings=query_vector,
                n_results=5,
                include=["documents", "metadatas", "distances"],
                where={"file_id": {"$in": file_id_list}}
            )
            return results
        except Exception as e:
            print(f"Something went wrong while retrieving file chunks: {e}")
            return None
        finally:
            unload_embedding_model()

    def rename_file_chunks_by_id(self, file_id: int, new_name: str):
        collection = self.chroma_client.get_or_create_collection(name="file_embeddings")
        matching = collection.get(
            where={"file_id": file_id},
            include=["metadatas"]
        )
        
        if not matching["ids"]:
            return

        updated_metadatas = []
        for meta in matching["metadatas"]:
            current_meta = meta.copy() if meta else {}
            current_meta["file_name"] = new_name
            updated_metadatas.append(current_meta)

        collection.update(
            ids=matching["ids"],
            metadatas=updated_metadatas
        )

    def delete_file_chunks_by_id(self, file_id: int):
        collection = self.chroma_client.get_or_create_collection(name="file_embeddings")
        collection.delete(where={"file_id": file_id})

    def close(self):
        self.con.close()

chat_db = ChatDB()