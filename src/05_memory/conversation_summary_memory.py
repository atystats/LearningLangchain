from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

from langchain_classic.memory import ConversationSummaryMemory

from utils.helpers import print_seperator, print_title

def main():
    print_title("Conversation Summary Memory")
    llm = get_llm()
    memory = ConversationSummaryMemory(
        llm = llm
    )

    conversations = [
        ("Hi, My name is Ankit", "Hello Ankit, Nice to meet you"),
        ("I am learning Langchain", "That's great!"),
        ("Can you explain prompt engineering", "Prompt engineering is the process of designing effective prompts."),
        ("Tell me about RAG", "RAG combines retrieval with generation to improve factual response."),
        ("What are AI Agents?", "AI agents can plan, reason, and use tools."),
    ]

    for human, ai in conversations:
        memory.save_context(
            {"input": human},
            {"output": ai}
        )

    history = memory.load_memory_variables({})

    print("Conversation Summary:\n")
    print(history["history"])

    print_seperator()

if __name__ == "__main__":
    main()

# Conversation Summary:

# The human introduces himself as Ankit. The AI greets Ankit and expresses pleasure in meeting him. Ankit mentions that he is learning Langchain, and the AI responds positively, encouraging him. Ankit asks the AI to explain prompt engineering, and the AI explains that prompt engineering is the process of designing effective prompts. The human then asks about RAG, and the AI explains that RAG combines retrieval with generation to improve factual responses. The human asks about AI Agents, and the AI explains that AI agents can plan, reason, and use tools.