"""
create_data.py

This script generates synthetic real estate listing data using language models (OpenAI or Ollama via LangChain).
It defines data models for real estate listings, sets up a prompt for the language model, and saves the generated data as JSON.
"""

from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import json

from logger_config import Logger
logger = Logger(name="CreateData").get_logger()

class RealEstate(BaseModel):
    """
    Data model representing a single real estate listing.
    """
    Neighborhood: str = "Green Oaks"
    Price: str = "500.000$"
    Bedrooms: int = 3
    Bathrooms: int = 2
    HouseSize: int = 200
    Description: str = ""
    NeighborhoodDescription: str = ""

class RealEstateCollection(BaseModel):
    """
    Data model representing a collection of real estate listings.
    """
    RealEstateObj: List[RealEstate] = Field(description="List of RealEstates")

class GenerateData:
    """
    Class to generate synthetic real estate data using a language model.
    """
    model = None


    def init_model(self, open_ai=True, temperature=0.0):
        """
        Initialize the language model for data generation.

        Args:
            open_ai (bool): If True, use OpenAI model; otherwise, use Ollama.
            temperature (float): Sampling temperature for the model.
        """
        if open_ai:
            model_name = "gpt-4o-mini"
            self.llm = ChatOpenAI(temperature=0.0, model=model_name)
        else:
            model_name="llama3.2:1b-instruct-fp16"
            self.llm = ChatOllama(temperature=temperature, model=model_name)

        logger.info(f"Initialize LLM {model_name}")

        self.model = self.llm.with_structured_output(RealEstateCollection)

    def generate_data(self):
        """
        Generate synthetic real estate data using the initialized language model.

        Returns:
            RealEstateCollection: Generated collection of real estate listings.
        """
        
        prompt = """
        You are a random real estate lising generator. Your output should be a json file. Just return the json file, no additional information or text.

        A sample for a dataset is
        -----------
        {example}
        ------------

        """

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("user", "{query}"),
        ])

        pipeline = (
            {
                "query": lambda x: x["query"],
                "example": lambda x: x["example"]
            }
            | prompt_template
            | self.model
        )

        query = """
                Generate 20 of these data sets. Make variations in all fields.
                Think of different types of customers, families with childen, older persons, people that live alone.
                Also think of different personalities and their needs.
        """

        example_dic = {
            "Neighborhood": "Green Oaks",
            "Price": "$650,000",
            "Bedrooms": "3",
            "Bathrooms": "2",
            "House Size": "2100 sqft",
            "Description": "Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.",
            "Neighborhood Description":"Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze."
        }

        example = json.dumps(example_dic, sort_keys=False)

        logger.info(f"Invoke LLM pipeline")
        result = pipeline.invoke({"query": query, "example": example})

        return result
    


def main():
    """
    Main function to generate data and save it as a JSON file.
    """
    logger.info(f"Starting data generation")
    generator = GenerateData()
    generator.init_model(temperature=0.0)
    data = generator.generate_data()

    data_json = data.model_dump(mode="json", exclude_unset=True)

    # print for debug
    pretty_json = json.dumps(data_json, indent=2, sort_keys=False)
    print(pretty_json)

    with open('data/data.json', 'w') as file:
        json.dump(data_json, file, indent=4, sort_keys=False)

    logger.info(f"Finished data generation")

if __name__ == '__main__':
    main()