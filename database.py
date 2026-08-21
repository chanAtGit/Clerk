import sqlite3
import uuid
import chromadb
import numpy as np

from datetime import datetime
from dataclasses import dataclass

from model_commons import load_embedding_model, unload_embedding_model, embedding_encode, EMBEDDING_MODEL_NAME

@dataclass
class Inquiry:
    prompt: str
    response: str|None
    chatSession_id: str

class ChatDB():
    def __init__(self):
        # Connect to SQLite database and create tables if they don't exist
        self.con = sqlite3.connect("chatData.db")
        self.cur = self.con.cursor()
        # Create chatsessions table
        self.cur.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS chatSessions (
                        chatSession_id TEXT PRIMARY KEY,
                        name TEXT,
                        created_date DATETIME UNIQUE
                    )
                    '''
                )
        # Create inquiries table
        self.cur.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS inquiries (
                        inquiry_id TEXT PRIMARY KEY,
                        prompt TEXT NOT NULL,
                        response TEXT,
                        chatSession_id TEXT,
                        created_date DATETIME UNIQUE
                    )
                    '''
                )
        self.con.commit()

        # Create Chroma client and collection for embeddings
        self.chroma_client = chromadb.PersistentClient(path="clerk_vectordb")

    def create_chatsession(self, name: str):
        '''Create chatsession record in sqlite'''
        chatSession_id = uuid.uuid4().hex # generate id
        created_date:datetime = datetime.now()
        self.cursor.execute(
            f'''
            INSERT INTO chatSessions
            VALUES (
                {chatSession_id},
                {name},
                {created_date}
            )
            '''
        )
        self.con.commit()

    def create_inquiry(self, inquiry: Inquiry):
        '''Create inquiry record in sqlite'''
        inquiry_id = uuid.uuid4().hex # generate id
        created_date:datetime = datetime.now()
        self.cursor.execute(
            f'''
            INSERT INTO inquiries
            VALUES (
                {inquiry_id},
                {inquiry.prompt},
                {inquiry.response},
                {inquiry.chatSession_id},
                {created_date}
            )
            '''
        )
        self.con.commit()

    def get_all_chatsessions(self) -> list:
        self.cur.execute(
            '''
            SELECT chatSession_id, name FROM chatSessions
            ORDER BY created_date DESC
            '''
        )
        result = self.cur.fetchall()
        return result

    def get_inquiries_from_session(self, chatSession_id: str) -> list:
        self.cur.execute(
                f'''
                SELECT prompt, response FROM inquiries
                WHERE chatSession_id == {chatSession_id}
                ORDER BY created_date ASC
                '''
        )
        result = self.cur.fetchall()
        return result

    def add_file_chunk_embedding(self, file_id: int, file_name: str, page_num:int, embedding: np.ndarray, chunk: str = None):
        '''Add embedding to Chroma collection'''
        chunk_id = uuid.uuid4().hex # generate id
        collection = self.chroma_client.get_or_create_collection(name="file_embeddings")
        # add record
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
                where={"file_id": {"$in": file_id_list}} # file_id must be in the list
            )

            print(f"Retrieved {len(results['metadatas'][0])} file chunks from ChromaDB for prompt: {prompt}")
            file_names = [meta["file_name"] for meta in results['metadatas'][0]]
            print(file_names)
            print(results['distances'][0])
            return results
        except Exception as e:
            print(f"Something went wrong while retrieving file chunks: {e}")
            return None
        finally:
            unload_embedding_model()

    def rename_file_chunks_by_id(self, file_id: int, new_name: str):
        collection = self.chroma_client.get_or_create_collection(name="file_embeddings")

        # 1. Retrieve IDs and existing metadatas matching the filter
        matching = collection.get(
            where={"file_id": file_id},
            include=["metadatas"]
        )
        
        if not matching["ids"]:
            return  # No records matched the given file_id

        # 2. Build updated metadata dictionaries preserving existing key-value pairs
        updated_metadatas = []
        for meta in matching["metadatas"]:
            current_meta = meta.copy() if meta else {}
            current_meta["file_name"] = new_name
            updated_metadatas.append(current_meta)

        # 3. Perform the update with explicit IDs and metadatas
        collection.update(
            ids=matching["ids"],
            metadatas=updated_metadatas
        )

    def delete_file_chunks_by_id(self, file_id: int):
        collection = self.chroma_client.get_or_create_collection(name="file_embeddings")
        collection.delete(where={"file_id": file_id})