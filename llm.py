from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
)

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama.chat_models import ChatOllama

import json

import llm_history
import user_data
from database import Database

class RealEstate(BaseModel):
    Price: str = Field(default="500.000$", description="Price"),
    Bedrooms: int = Field(default=3, description="Bedrooms"),
    Bathrooms: int = Field(default=2, description="Bathrooms"),
    HouseSize: int = Field(default=200, description="House size"),
    Description:str = Field(default="", description="Description"),
    NeighborhoodDescription: str = Field(default="", description="Neighborhood description")

class LLM:
    def __init__(self):
        model_name = "llama3.2:1b-instruct-fp16"
        self.llm = ChatOllama(temperature=0.0, model=model_name)
        self.model = self.llm.with_structured_output(RealEstate)
        system_prompt = """
        You are AI that will recommend user a real estates based on their answers to personal questions. 
        You will only add information to your response that are in the users answers or than can be concluded from the anserws.
        Ask user questions now.
        """

        prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{query}"),])
        self.pipeline = prompt_template | self.model

    def conversation(self, history_dic):
        # prefill history
        history = llm_history.get_by_session_id("id_1")

        for question, answer in zip(history_dic["questions"], history_dic["answers"]):
            history.add_ai_message(question)
            history.add_user_message(answer)

        pipeline_with_history = RunnableWithMessageHistory(
            self.pipeline,
            get_session_history=llm_history.get_by_session_id,
            input_messages_key="query",
            history_messages_key="history"
        )

        query = """
                "Here is a list with questions and answers of a customer who is looking for a real estate.
                Please write a short profile of the customer as json with
                - price range
                - number of bedrooms
                - numbr of bathrooms
                - size of the house
                - wished for the house
                - wishes for the neighborhood
        """

        result = pipeline_with_history.invoke(
            {"query": query},
            config={"session_id": "id_1"}
        )

        return result.model_dump(mode="json", exclude_unset=True)


def main():
    real_estate_llm = LLM()

    questions, answers = user_data.get_info()
    profile = real_estate_llm.conversation(history_dic={"questions": questions, "answers": answers})
    
    profile_json_string = json.dumps(profile, indent=2, sort_keys=False)
    print(profile_json_string, "\n\n\n")

    db = Database(open_ai=False)
    #db.load_data("data/data.json")
    db.load_db()

    #list_documents()
    results = db.similarity_search(profile_json_string, k=3)

    for doc in results:
        print(doc.page_content)
        #print(doc.metadata)

if __name__ == '__main__':
    main()