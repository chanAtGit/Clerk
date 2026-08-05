import sqlite3
import uuid
import chromadb
from datetime import datetime
from dataclasses import dataclass

from model_commons import load_embedding_model, unload_embedding_model, embedding_encode

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

