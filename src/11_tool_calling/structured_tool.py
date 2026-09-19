from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from utils.helpers import print_seperator, print_title

from utils.helpers import print_seperator, print_title

class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""
    a: float = Field(desciption = "First number.")
    b: float = Field(description = "Second number.")
    operation: str = Field(description="Mathematical operation: add, substract," \
    "multiply or divide.")

def calculate(a: float, b: float, operation: str) -> float:
    """
    Performs a mathematical operation on two numbers.
    """
    if operation == "add":
        return a+b
    if operation == "substract":
        return a - b
    if operation == "multiply":
        return a*b
    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a/b
    raise ValueError("Invalid operation. Choose: add, substract, multiply, or divide.")

def main() -> None:
    print_title("Structured Tool")

    calculator_tool = StructuredTool.from_function(
        func = calculate,
        name = "calculator",
        description = "Performs a mathematical operation on two numbers",
        args_schema= CalculatorInput,
    )

    print(f"Tool name: {calculator_tool.name}")
    print_seperator()
    print(f"Tool Description: {calculator_tool.description}")
    print_seperator()
    print(f"Input Schema: {calculator_tool.args_schema.model_json_schema()}")
    print_seperator()

    result = calculator_tool.invoke(
        {
            "a": 20,
            "b": 5,
            "operation": "multiply"
        }
    )

    print(f"Calculator Result: \n{result}")

if __name__ == "__main__":
    main()
