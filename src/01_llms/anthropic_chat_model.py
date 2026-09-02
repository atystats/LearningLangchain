from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

def main():
    """
    Demonstrate chatting with a gemini model using Langchain
    """

    print_title("Anthropic Chat Model")

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
    print("ANTHROPIC RESPONSE:\n")
    print(response.text)
    print("-" * 80)
    print(type(response))

if __name__ == "__main__":
    main()