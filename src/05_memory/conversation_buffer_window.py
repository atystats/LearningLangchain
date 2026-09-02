from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

from langchain_classic.memory import ConversationBufferWindowMemory

from utils.helpers import print_seperator, print_title

def main():
    print_title("Conversation Buffer Window Memory")
    memory = ConversationBufferWindowMemory(k = 2, return_messages=True)

    conversations = [
        ("Hi, My name is Ankit", "Hello Ankit, Nice to meet you"),
        ("I am learning Langchain", "That's great!"),
        ("Tell me about RAG", "RAG combines retrieval with generation."),
        ("What are AI Agents?", "AI agents can plan, reason, and use tools."),
    ]

    for human, ai in conversations:
        memory.save_context(
            {"input": human},
            {"output": ai}
        )

    history = memory.load_memory_variables({})

    print("Complete Stored in Memory:\n")

    for index, message in enumerate(history["history"], start = 1):
        print(f"Message {index}")
        print(f"Type    :{message.type}")
        print(f"Content :{message.content}")
        print_seperator()

if __name__ == "__main__":
    main()

# Complete Stored in Memory:

# Message 1
# Type    :human
# Content :Tell me about RAG
# ============================================================
# Message 2
# Type    :ai
# Content :RAG combines retrieval with generation.
# ============================================================
# Message 3
# Type    :human
# Content :What are AI Agents?
# ============================================================
# Message 4
# Type    :ai
# Content :AI agents can plan, reason, and use tools.