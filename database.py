from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

import chromadb
import configparser

from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np



class Database:
    def __init__(self, persist_directory=".chroma_db", collection_name="real_estate", open_ai=True):
        self.persist_directory = persist_directory
        self.collection_name=collection_name
        if open_ai:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        else:
            self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.clip_model = SentenceTransformer("clip-ViT-B-32")

    def load_data(self, filename):
        loader = JSONLoader(file_path=filename, jq_schema=".RealEstateObj[]",
                            text_content=False, metadata_func=metadata_func,)
        data = loader.load()

        for doc in data:
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")

        self.db = Chroma.from_documents(data, embedding=self.embeddings, persist_directory=self.persist_directory, collection_name=self.collection_name)

    def load_db(self):
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

    def similarity_search(self, query, k=3):
        return self.db.similarity_search(query, k)

    def metadata_func(self, record: dict, metadata: dict) -> dict:
        image_path = record.get("ImagePath")
        metadata["ImagePath"] = image_path
        img_embedding = self.clip_model.encode(Image.open(image_path)).tolist()
        metadata["ImageEmbedding"] = img_embedding

        return metadata

# def list_documents():
#     client = chromadb.PersistentClient(path=".chroma_db")
#     # List all collections to find the correct one
#     collections = client.list_collections()
#     collection_names = [col.name for col in collections]
#     print("Available collections:", collection_names)

#     # Access the collection (replace with your collection name)
#     collection = client.get_collection("real_estate")

#     # Retrieve all data
#     all_data = collection.get()

#     # Print each document
#     for i in range(len(all_data["ids"])):
#         print(f"\n--- Document {i+1} ---")
#         print(f"ID: {all_data['ids'][i]}")
#         print(f"Content: {all_data['documents'][i]}")
#         print(f"Metadata: {all_data['metadatas'][i]}")

def main():
    parser = configparser.ConfigParser()
    parser.read("settings.ini")

    open_ai = parser.getboolean("DEFAULT", "open_ai")
    db = Database(open_ai=open_ai)
    db.load_data("data/houses_with_images.json")
    #db.load_db()

    #list_documents()
    #results = db.similarity_search("silent neighborhood", k=3)

    #for doc in results:
    #    print(doc.page_content)

if __name__ == '__main__':
    main()