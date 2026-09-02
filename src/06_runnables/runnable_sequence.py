from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def main() -> None:
    print_title("Runnable Sequence")
    llm = get_llm()

    prompt = PromptTemplate.from_template(
        "Explain {topic} in less than 30 seconds."
    )

    sequence = prompt | llm | StrOutputParser()

    topic = "Vector Databases"

    print(f"Topic:\n{topic}")

    print_seperator()

    response = sequence.invoke({"topic": topic})

    print("Response:\n")
    print(response)

if __name__ == "__main__":
    main()