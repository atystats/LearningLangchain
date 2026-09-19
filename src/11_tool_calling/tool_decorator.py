from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.tools import tool
from utils.helpers import print_seperator, print_title

@tool
def calculate_discount(price: float, discount: float) -> float:
    """
    Calculate the final price after applying a discount.

    Args:
        price : Original price of the product
        discount : Discount percentage
    """
    return price - (price * discount / 100)

def main() -> None:
    """
    Demonstrate creating Langchain tool using a decorater.
    """
    print_title("Tool Decorater")

    print(f"Tool name : {calculate_discount.name}")
    print_seperator()
    print(f"Tool description : {calculate_discount.description}")
    print_seperator()
    print(f"Tool Input Schema: {calculate_discount.args_schema.schema()}")
    print_seperator()

    price = 1000
    discount = 20

    print(f"Price: {price} | Discount: {discount}%")
    print_seperator()
    result = calculate_discount.invoke(
        {
            "price": price,
            "discount": discount,
        }
    )

    print(f"Final Price: {result}")

if __name__ == "__main__":
    main()
