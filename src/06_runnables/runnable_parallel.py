from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

def main():
    print_title("Runnable Parallel")
    llm = get_llm()

    summary_chain = (
        PromptTemplate.from_template(
            "Write a short summary about {topic}."
        )
    | llm
    | StrOutputParser()
    )

    advantages_chain = (PromptTemplate.from_template(
        "List three advantages of {topic}"
    )
    | llm
    | StrOutputParser()
    )

    application_chain = (PromptTemplate.from_template(
        "List three real-world applications of {topic}"
    )
    | llm
    | StrOutputParser()
    )

    parallel_chain = RunnableParallel(
        summary = summary_chain,
        advantages = advantages_chain,
        applications = application_chain,
    )

    topic = "Vector Databases"
    print(f"Topic:\n{topic}")

    print_seperator()
    result = parallel_chain.invoke({"topic": topic})

    print("Summary:\n")
    print(result["summary"])
    print_seperator()

    print("Advantages:\n")
    print(result["advantages"])
    print_seperator()

    print("Applications:\n")
    print(result["applications"])
    print_seperator()

if __name__ == "__main__":
    main()