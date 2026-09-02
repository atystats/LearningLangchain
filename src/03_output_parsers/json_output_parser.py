from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

def main():
    print_title("JSON Output Parser")
    llm = get_llm()
    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template = """
        Extract the following product information.

        - Product Name
        - RAM
        - Storage
        - Display Size
        - Price
        {format_instructions}

        Product Description:
        {product}
        """,
        input_variables= ["product"],
        partial_variables= {
            "format_instructions": parser.get_format_instructions()
        },
    )

    prompt_value = prompt.invoke(
        {
            "product": (
                "The Apple MacBook Air M4 comes with 16GB RAM, "
                "512GB SSD storage, a 13.6-inch Liquid Retina display, "
                "and is priced at $1,199."
            )
        }
    )

    # print("Formatted Prompt:\n")
    # print(prompt_value.text)

    # print_seperator()

    response = llm.invoke(prompt_value)

    print("Raw LLM response:\n")
    print(response.content)

    print_seperator()

    parsed_response = parser.invoke(response)

    print("Parsed JSON:\n")
    print(parsed_response)

    print_seperator()

    print("Accessing Individual Fields:\n")

    for key, value in parsed_response.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()

# Raw LLM response:

# ```json
# {
#   "Product Name": "Apple MacBook Air M4",
#   "RAM": "16GB",
#   "Storage": "512GB SSD",
#   "Display Size": "13.6-inch",
#   "Price": "$1,199"
# }
# ```
# ============================================================
# Parsed JSON:

# {'Product Name': 'Apple MacBook Air M4', 'RAM': '16GB', 'Storage': '512GB SSD', 'Display Size': '13.6-inch', 'Price': '$1,199'}
# ============================================================
# Accessing Individual Fields:

# Product Name: Apple MacBook Air M4
# RAM: 16GB
# Storage: 512GB SSD
# Display Size: 13.6-inch
# Price: $1,199