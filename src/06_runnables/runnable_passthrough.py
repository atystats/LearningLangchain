# =============================================================================
# Concept Summary
#
# RunnablePassthrough forwards the original input without modifying it.
#
# It is useful when you want to keep the original input while generating
# additional outputs in parallel.
#
# Example:
#
#                 User Input
#                     │
#         ┌───────────┴───────────┐
#         ▼                       ▼
# RunnablePassthrough      Summary Chain
#         │                       │
#         └───────────┬───────────┘
#                     ▼
#              Combined Output
# =============================================================================

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

def main():
    print_title("Runnable Passthrough")
    llm = get_llm()

    summary_chain = (
        PromptTemplate.from_template(
            "Write a short summary about: {topic}."
        )
    | llm
    | StrOutputParser()
    )

    pipeline = RunnableParallel(
        original_input = RunnablePassthrough(),
        summary = summary_chain,
    )

    topic = "Large Language Model"
    print(f"Input Topic: \n{topic}")
    print_seperator()

    result = pipeline.invoke({"topic": topic})

    print("Original Input:\n")
    print(result["original_input"])
    print_seperator()

    print("Generated Summary: \n")
    print(result["summary"])

if __name__ == "__main__":
    main()