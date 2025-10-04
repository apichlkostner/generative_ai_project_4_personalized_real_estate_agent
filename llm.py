from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
)
import configparser

import llm_history
import user_data
from database import Database


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
                - wishes for the house
                - wishes for the neighborhood
        """

        result = pipeline_with_history.invoke(
            {"query": query},
            config={"session_id": "id_1"}
        )

        return result.content
    
    def conversation_image(self, history_dic):
        # prefill history
        history = llm_history.get_by_session_id("id_2")

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
                Please write a short profile of the customer with only the visual aspects which can be seen
                from outside. Like the colour, windows size, garden.
        """

        result = pipeline_with_history.invoke(
            {"query": query},
            config={"session_id": "id_2"}
        )

        return result.content
    
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

    profile_image = real_estate_llm.conversation_image(history_dic={"questions": questions, "answers": answers})
    #print(profile_image)
    
    db = Database(open_ai=open_ai)

    # First similarity search over the textual description
    results = db.similarity_search_text(profile, k=6)
    content = results["documents"][0]

    # Second similarity search over the images
    results_image = db.similarity_search_image(profile_image, k=15)
    content_image = results_image["uris"][0]

    # choose three samples
    samples = ""
    num_samples = 0
    max_samples = 3
    # get the 3 best image matches that are also good matches in the text search
    for idx, id in enumerate(results_image['ids'][0]):
        if num_samples >= max_samples:
            break
        if id in results['ids'][0]:
            idx_results = results['ids'][0].index(id)
            samples += f"{results['documents'][0][idx_results]}\n-------------------------------\n"
            num_samples += 1

    answer_for_customer = real_estate_llm.results(samples)
    print(answer_for_customer)

if __name__ == '__main__':
    main()