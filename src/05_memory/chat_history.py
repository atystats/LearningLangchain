from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from utils.helpers import print_seperator, print_title

def main():
    print_title("Chat Message History")
    chat_history = InMemoryChatMessageHistory()

    chat_history.add_message(
        HumanMessage(content = "What is Langchain?")
    )

    chat_history.add_message(
        AIMessage(content="Lnagchain is a framework for building LLM powered application.")
    )

    chat_history.add_message(
        HumanMessage(content="What are Prompt Templates?")
    )

    chat_history.add_message(
        AIMessage(content="Prompt Templates help create reusable prompts with dynamic variables.")
    )

    print("Conversation History:\n")

    for index, message in enumerate(chat_history.messages, start = 1):
        print(f"Message {index}")
        print(f"Type    :{message.type}")
        print(f"Content :{message.content}")
        print_seperator()

if __name__ == "__main__":
    main()