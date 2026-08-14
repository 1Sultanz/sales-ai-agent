# Söhbət yaddaşı.

from langchain.memory import ConversationBufferWindowMemory

MEMORY_WINDOW = 6


def create_memory() -> ConversationBufferWindowMemory:
    return ConversationBufferWindowMemory(
        k=MEMORY_WINDOW,
        memory_key="chat_history",
        return_messages=False,
        output_key="output",
    )


def reset_memory() -> ConversationBufferWindowMemory:
    return create_memory()