from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import ChatPromptTemplate

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

def main():
    print_title("ChatPromptTemplate")
    llm = get_llm()

    chat_prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are an experienced {profession} who explains the concept in a simple and beginner-friendly way."
            ),
            (
                "human",
                "Explain the concept of {topic} in less than {word_limit} words."
            ),
            (
                "ai",
                "Sure! I will explain it in clear and concise manner."
            )
        ]
    )

    messages = chat_prompt.invoke(
        {
            "profession": "AI Enginner",
            "topic": "Prompt Engineering",
            "word_limit": 500,
        }
    )


    print("Formetted Messages:\n")
    for message in messages.messages:
        print(f"{message.type.upper()}:")
        print(message.content)

    print_seperator()

    response = llm.invoke(messages)

    print("LLM Response:\n")
    print(response.content)

if __name__ == "__main__":
    main()