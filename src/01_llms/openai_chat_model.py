from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm

def chat_with_model() -> None:
    """
    Sends a simple prompt to the configured chat model and prints the response.

    This example demonstrate the most basic interaction with a Langchain chat model.
    """

    llm = get_llm()
    prompt = """
    Explain what is Langchain in three concise bullet points
    """

    print("-" * 80)
    print("USER INPUT:\n")
    print(prompt.strip())
    print("-" * 80)

    print("\nGenerating Response....\n")
    response = llm.invoke(prompt)

    print("-" * 80)
    print("AI RESPONSE:\n")
    print(response.content)
    print("-" * 80)
    print(type(response))

if __name__ == "__main__":
    chat_with_model()