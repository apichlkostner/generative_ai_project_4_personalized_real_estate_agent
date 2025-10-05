"""
database.py

This module provides a Database class for managing real estate data using ChromaDB.
It supports storing and searching textual and image data with embedding functions.
"""

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.utils.data_loaders import ImageLoader

import argparse
import configparser
from datetime import datetime
import json
import os

from logger_config import Logger
logger = Logger(name="RealEstateDB").get_logger()

data_template = \
"""Neighborhood: {}
Price: {}
Bedrooms: {}
Bathrooms: {}
HouseSize: {}
Description: {}
NeighborhoodDescription: {}"""

class Database:
    """
    Database class for managing real estate data in ChromaDB.

    Supports adding data, and performing similarity searches on text and images.
    """
    def __init__(self, persist_directory=".chroma_db", collection_name="real_estate", open_ai=True):
        """
        Initialize the Database with embedding functions and collections.

        Args:
            persist_directory (str): Directory for persistent storage.
            collection_name (str): Name for the collection.
            open_ai (bool): Whether to use OpenAI embeddings or Ollama.
        """
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
        """
        Create or retrieve the text and image collections in the database.
        """
        logger.info("Creating or loading the collections for text and images")
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
        """
        Add real estate data from a JSON file to the text and image collections.

        Args:
            filename (str): Path to the JSON file containing real estate data.
        """
        logger.info("Loading data")
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            logger.error(f"Error: file not found: {filename}")
        except json.JSONDecodeError:
            logger.error("Error: Failed to decode JSON from the file.")

        objects = data["RealEstateObj"]
        ids = []
        documents = []
        metadatas = []
        uris = []
        logger.info("Starting loop")
        for index, obj in enumerate(objects):
            logger.info(f"Adding data for {obj["Neighborhood"]}")
            ids.append(f"id{index}")
            documents.append(data_template.format(obj["Neighborhood"], obj["Price"], obj["Bedrooms"], obj["Bathrooms"],obj["HouseSize"], obj["Description"], obj["NeighborhoodDescription"]))
            image_uri = f"house_images/{index}.png"
            metadatas.append({"source" : filename, "index": index, "image": image_uri})
            uris.append(image_uri)

        logger.info("Loop finished")

        logger.info("Adding text data to collection")
        self.col_text.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Adding image data to collection")
        self.col_image.add(ids=ids, uris=uris)

    def similarity_search_text(self, query, k=3):
        """
        Perform a similarity search on the text collection.

        Args:
            query (str): The query string.
            k (int): Number of results to return.

        Returns:
            dict: Search results from the text collection.
        """
        return self.col_text.query(query_texts=query, n_results=k)
    
    def similarity_search_image(self, query, k=3):
        """
        Perform a similarity search on the image collection.

        Args:
            query (str): The query string.
            k (int): Number of results to return.

        Returns:
            dict: Search results from the image collection.
        """
        return self.col_image.query(query_texts=query, include=['uris'], n_results=k)



def main():
    """
    Main function for testing the Database class.

    Loads configuration, initializes the database, and performs sample searches.
    """
    logger.info("Started database handling")

    parser = configparser.ConfigParser()
    parser.read("settings.ini")

    open_ai = parser.getboolean("DEFAULT", "open_ai")
    db = Database(open_ai=open_ai)

    # CLI argument parsing
    arg_parser = argparse.ArgumentParser(description="ChromaDB Real Estate Database CLI")
    arg_parser.add_argument("--add-data", action="store_true", help="Add data from JSON file to collections")
    arg_parser.add_argument("--text-search", help="Perform a text similarity search")
    arg_parser.add_argument("--image-search", help="Perform an image similarity search")
    arg_parser.add_argument("-k", type=int, default=3, help="Number of results to return (default: 3)")
    args = arg_parser.parse_args()

    if args.add_data:
        logger.info("Adding data to the database collections")
        db.add_data_to_collections("./data/data.json")

    if args.text_search:
        logger.info("Database text search test")
        results = db.similarity_search_text(args.text_search, k=args.k)
        docs = results["documents"][0]
        for doc in docs:
            print(doc)
            print("")

    if args.image_search:
        logger.info("Database image search test")
        results = db.similarity_search_image(args.image_search, k=args.k)
        docs = results["uris"][0]
        for doc in docs:
            print(doc)
            print("")

    logger.info("Finished database handling")

if __name__ == '__main__':
    main()