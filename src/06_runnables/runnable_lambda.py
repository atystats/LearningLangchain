from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
# Runnablelambda convert a normal python function into a Langchain Runnable.
# This allows custom python logic to become part of an LCEL pipeline.

def to_uppercase(text):
    return text.upper()

def count_words(text):
    return f"Total words: {len(text.split())}"

def main():
    print_title("Runnable Lambda")

    uppercase = RunnableLambda(to_uppercase)
    word_counter = RunnableLambda(count_words)

    text = "Langchain makes it easy to build LLM applications."

    print(f"Original Text:\n{text}")
    print_seperator()

    print(f"Uppercase:\n")
    print(uppercase.invoke(text))

    print_seperator()

    print("Word Count:\n")
    print(word_counter.invoke(text))

if __name__ == "__main__":
    main()

# =============================================================================
# Concept Summary
#
# RunnableLambda converts a normal Python function into a LangChain Runnable.
#
# This allows custom Python logic to become part of an LCEL pipeline.
#
# Example:
#
# User Input
#      ↓
# RunnableLambda
#      ↓
# Custom Python Function
# =============================================================================