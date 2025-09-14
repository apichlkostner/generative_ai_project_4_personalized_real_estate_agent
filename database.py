from langchain.vectorstores import Chroma
from langchain.document_loaders.json_loader import JSONLoader

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

import chromadb

class Database:
    def __init__(self, persist_directory=".chroma_db", open_ai=True):
        self.persist_directory = persist_directory
        if open_ai:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    def load_data(self, filename):
        loader = JSONLoader(file_path=filename, jq_schema=".RealEstateObj[]",
                            text_content=False)
        data = loader.load()

        for doc in data:
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")

        self.db = Chroma.from_documents(data, embedding=self.embeddings, persist_directory=self.persist_directory, collection_name="real_estate")

    def load_db(self):
        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

def list_documents():
    client = chromadb.PersistentClient(path=".chroma_db")
    # List all collections to find the correct one
    collections = client.list_collections()
    collection_names = [col.name for col in collections]
    print("Available collections:", collection_names)

    # Access the collection (replace with your collection name)
    collection = client.get_collection("real_estate")

    # Retrieve all data
    all_data = collection.get()

    # Print each document
    for i in range(len(all_data["ids"])):
        print(f"\n--- Document {i+1} ---")
        print(f"ID: {all_data['ids'][i]}")
        print(f"Content: {all_data['documents'][i]}")
        print(f"Metadata: {all_data['metadatas'][i]}")

def main():
    db = Database(open_ai=False)
    #db.load_data("data.json")
    db.load_db()

    list_documents()
    

    

if __name__ == '__main__':
    main()