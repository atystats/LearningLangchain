from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.tools import tool
from llm_client import get_llm
from utils.helpers import print_seperator, print_title

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a*b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a+b

def main() -> None:
    """Demonstrates binding tools to an LLM."""    
    print_title("Tool Binding.")

    llm = get_llm()
    tools = [multiply, add]

    llm_with_tools = llm.bind_tools(tools)

    query = "what is 25 multiplied by 4?"
    print(f"User's Query :{query}")
    print_seperator()
    result = llm_with_tools.invoke(query)
    print(f"Model's Response: {result.content}")
    print_seperator()

    if result.tool_calls:
        print("Tool Calls: \n")

        for tool_call in result.tool_calls:
            print(f"Tool: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")
            print(f"Call ID: {tool_call['id']}")
            print_seperator()

    else:
        print("No tool call was requested by the model.")

if __name__ == "__main__":
    main()