"""
llm_history.py

This module provides an in-memory chat message history implementation for use with
LangChain's chat and message history interfaces. It allows storing, retrieving, and
managing chat histories per session, with a configurable limit on the number of messages.
"""

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

store = {}

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """
    In-memory implementation of chat message history.

    Stores chat messages up to a maximum of `k` messages. When new messages are added,
    older messages are discarded to maintain the limit.
    """
    messages: list[BaseMessage] = Field(default_factory=list)
    k: int = Field(default_factory=int)

    def __init__(self, k = 50):
        """
        Initialize the in-memory history with a maximum number of messages.

        Args:
            k (int): Maximum number of messages to retain in history.
        """
        super().__init__(k=k)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """
        Add messages to the history, removing any messages beyond the last `k` messages.

        Args:
            messages (list[BaseMessage]): Messages to add to the history.
        """
        self.messages.extend(messages)
        self.messages = self.messages[-self.k:]

    def clear(self) -> None:
        """
        Clear the history by removing all stored messages.
        """
        self.messages = []

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieve the chat message history for a given session ID.

    If no history exists for the session, a new InMemoryHistory is created.

    Args:
        session_id (str): The session identifier.

    Returns:
        BaseChatMessageHistory: The chat message history for the session.
    """
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]