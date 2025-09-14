from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

import json

class RealEstate(BaseModel):
    Neighborhood: str = Field(default="GreenField", description="Neighborhood"),
    Price: str = Field(default="500.000$", description="Price"),
    Bedrooms: int = Field(default=3, description="Bedrooms"),
    Bathrooms: int = Field(default=2, description="Bathrooms"),
    HouseSize: int = Field(default=200, description="House size"),
    Description:str = Field(default="", description="Description"),
    NeighborhoodDescription: str = Field(default="", description="Neighborhood description")

class RealEstateCollection(RealEstate):
    RealEstateObj: List[RealEstate] = Field(description="List of RealEstates")

class GenerateData:
    model = None


    def init_model(self, model_name="llama3.2:1b-instruct-fp16", temperature=0.0):
        model = ChatOllama(temperature=temperature, model=model_name)
        self.model = model.with_structured_output(RealEstateCollection)

    def generate_data(self):
        
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

        query = "Generate 20 of these data sets. Make variations in all fields."

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

        result = pipeline.invoke({"query": query, "example": example})

        return result
    


def main():
    generator = GenerateData()
    generator.init_model(temperature=0.0)
    data = generator.generate_data()

    data_json = data.model_dump(mode="json", exclude_unset=True)

    # print for debug
    pretty_json = json.dumps(data_json, indent=2, sort_keys=False)
    print(pretty_json)

    with open('data/data.json', 'w') as file:
        json.dump(data_json, file, indent=4, sort_keys=False)

if __name__ == '__main__':
    main()