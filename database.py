import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.utils.data_loaders import ImageLoader

import configparser

from datetime import datetime
import json
import os

data_template = \
"""Neighborhood: {}
Price: {}
Bedrooms: {}
Bathrooms: {}
HouseSize: {}
Description: {}
NeighborhoodDescription: {}"""

class Database:
    def __init__(self, persist_directory=".chroma_db", collection_name="real_estate", open_ai=True):
        self.persist_directory = persist_directory
        self.collection_name=collection_name
        if open_ai:
            self.embedding_text = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-3-large",
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base=os.getenv("OPENAI_BASE_URL")
            )
        else:
            self.embedding_text = embedding_functions.OllamaEmbeddingFunction(model_name="mxbai-embed-large")

        self.embedding_image = embedding_functions.OpenCLIPEmbeddingFunction()
        self.db = chromadb.PersistentClient(path=persist_directory)

        self.create_collections()

    def create_collections(self):        
        self.col_text = self.db.get_or_create_collection(
            name="real_estate_description",
            embedding_function=self.embedding_text,
            metadata={
                "description": "Real-estate textual description",
                "created": str(datetime.now())
            })
        
        data_loader = ImageLoader()
        self.col_image = self.db.get_or_create_collection(
            name="real_estate_image",
            embedding_function=self.embedding_image,
            metadata={
                "description": "Real-estate textual description",
                "created": str(datetime.now()),
                },
            data_loader=data_loader
            )
        
    def add_data_to_collections(self, filename):
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            print("Error: file not found: {filename}")
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")

        objects = data["RealEstateObj"]
        ids = []
        documents = []
        metadatas = []
        uris = []
        for index, obj in enumerate(objects):
            ids.append(f"id{index}")
            documents.append(data_template.format(obj["Neighborhood"], obj["Price"], obj["Bedrooms"], obj["Bathrooms"],obj["HouseSize"], obj["Description"], obj["NeighborhoodDescription"]))
            image_uri = f"house_images/{index}.png"
            metadatas.append({"source" : filename, "index": index, "image": image_uri})
            uris.append(image_uri)

        self.col_text.add(ids=ids, documents=documents, metadatas=metadatas)
        self.col_image.add(ids=ids, uris=uris)

    def similarity_search_text(self, query, k=3):
        return self.col_text.query(query_texts=query, n_results=k)
    
    def similarity_search_image(self, query, k=3):
        return self.col_image.query(query_texts=query, include=['uris'], n_results=k)



def main():
    parser = configparser.ConfigParser()
    parser.read("settings.ini")

    open_ai = parser.getboolean("DEFAULT", "open_ai")
    db = Database(open_ai=open_ai)

    # try text search
    results = db.similarity_search_text("luxus", k=3)

    docs = results["documents"][0]
    for doc in docs:
        print(doc)
        print("")

    # try image search
    results = db.similarity_search_image("red house", k=3)
    docs = results["uris"][0]
    for doc in docs:
        print(doc)
        print("")

if __name__ == '__main__':
    main()