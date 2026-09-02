from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

def main():
    print_title("MessagesPlaceholder")
    llm = get_llm()

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI tutor who answers questions based on the ongoing conversation."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    chat_history = [
        HumanMessage(content = "What is LangChain?"),
        AIMessage(content = (
            "Langchain is a framework for building applications"
            "powered by Large Language Models."
        )),
        HumanMessage(content = "What are Prompt Templates?"),
        AIMessage(
            content= (
                "Prompt Template helps create reusable prompts by allowing dynamic variable."
            )
        )
    ]

    messages = chat_prompt.invoke(
        {"chat_history": chat_history,
        "question": "can you summarize both concepts in simple terms?"
        }
    )

    # for message in messages.messages:
    #     print(f"{message.type.upper()}:")
    #     print(message.content)

    # print_seperator()

    response = llm.invoke(messages)

    print("LLM Response:\n")
    print(response.content)

if __name__ == "__main__":
    main()

# LLM Response:

# Sure! 

# **LangChain** is a tool that helps you build apps using large language models like ChatGPT.

# **Prompt Templates** are like fill-in-the-blank scripts that let you create prompts with placeholders you can easily change. 

# Together, they make it easier to create smart language-based applications.