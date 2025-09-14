import pytest
import chromadb
import shutil
from database import Database

persist_directory=".test_chroma_db"

@pytest.fixture
def db_client():
    """Fixture provides a fresh DB client for each test"""
    db = Database(open_ai=False, persist_directory=persist_directory)
    db.load_data("data.json")
    return db


def test_create_db_from_json(db_client):
    """Fixture provides a fresh DB client for each test, connected to an in-memory DB."""    
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection("real_estate")
    all_data = collection.get()

    num_documents = len(all_data["ids"])

    assert num_documents == 13

    shutil.rmtree(persist_directory)
