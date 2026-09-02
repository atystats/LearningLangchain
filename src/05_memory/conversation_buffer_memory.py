from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

from langchain_classic.memory import ConversationBufferMemory

from utils.helpers import print_seperator, print_title

def main():
    print_title("Conversation Buffer Memory")
    memory = ConversationBufferMemory(return_messages=True)

    memory.save_context(
        {"input": "Hi, My name is Ankit"},
        {"output": "Hello Ankit, Nice to meet you"}
    )

    memory.save_context(
        {"input": "I am learning Langchain"},
        {"output": "That's great! Langchain is an excellent framework for building LLM applications"}
    )

    memory.save_context(
        {"input": "I also want to learn RAG"},
        {"output": "RAG is one of the most important concepts in modern GenAI applications"}
    )

    print("Complete Conversation history:\n")
    history = memory.load_memory_variables({})

    for index, message in enumerate(history["history"], start = 1):
        print(f"Message {index}")
        print(f"Type    :{message.type}")
        print(f"Content :{message.content}")
        print_seperator()

if __name__ == "__main__":
    main()