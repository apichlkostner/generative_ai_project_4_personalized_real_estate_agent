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
from langchain_openai import ChatOpenAI

import json

import llm_history
import user_data
from database import Database

import configparser
import sys


class RealEstate(BaseModel):
    Neighborhood: str = "Green Oaks"
    Price: str = "500.000$"
    Bedrooms: int = 3
    Bathrooms: int = 2
    HouseSize: int = 200
    Description: str = ""
    NeighborhoodDescription: str = ""

class LLM:
    def __init__(self, open_ai=True):
        if open_ai:
            model_name = "gpt-4o-mini"
            self.llm = ChatOpenAI(temperature=0.0, model=model_name)
        else:
            model_name = "llama3.2:1b-instruct-fp16"
            self.llm = ChatOllama(temperature=0.0, model=model_name)
        
        self.model = self.llm
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
                Please write a short profile of the customer with
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
    
    def results(self, context):
        system_prompt = """
        You are AI that will recommend user a real estates based on their answers to personal questions. 
        You will only add information to your response that are in the users answers or than can be concluded from the answers.
        Here are some available real estates that should be recommended to the user

        -------------------
        
        {context}

        -------------------
        """

        query = """
        For each of these houses write an individual description from the available information.
        Don't just repeat the data set. Write for each house in the descritption why it matched the users needs.
        """

        #print("SYSTEMPROMPT:\n", system_prompt, "\n\nPROMPT;\n", query, "\n\nCONTEXT:\n", context)

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{query}"),
        ])

        pipeline = prompt_template | self.llm

        pipeline_with_history = RunnableWithMessageHistory(
            pipeline,
            get_session_history=llm_history.get_by_session_id,
            input_messages_key="query",
            history_messages_key="history"
        )

        result = pipeline_with_history.invoke({"query": query, "context": context,}, config={"session_id": "id_1"})

        return result.content


def main():
    parser = configparser.ConfigParser()
    parser.read("settings.ini")

    open_ai = parser.getboolean("DEFAULT", "open_ai")
    print("Open AI: ", open_ai)

    real_estate_llm = LLM(open_ai=open_ai)

    questions, answers = user_data.get_info()
    profile = real_estate_llm.conversation(history_dic={"questions": questions, "answers": answers})
    profile = profile['content']
    
    db = Database(open_ai=open_ai)
    db.load_db()

    results = db.similarity_search(profile, k=3)
    context = results[0].page_content + "\n---------------\n" + results[1].page_content + "\n---------------\n" + results[2].page_content + "\n---------------\n" 

    answer_for_customer = real_estate_llm.results(context)
    print(answer_for_customer)

if __name__ == '__main__':
    main()