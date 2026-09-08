from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import CSVLoader
from utils.helpers import print_seperator, print_title

def main():
    print_title("CSV Loader")

    file_path = PROJECT_ROOT/ "data" / "input" / "employees.csv"
    loader = CSVLoader(file_path)
    documents = loader.load()

    print(f"Total ROWs loaded: {len(documents)}")
    print_seperator()

    first_row = documents[0]

    print(f"First Row Metadata: {first_row.metadata}")
    print_seperator()

    print(f"First Row Content: {first_row.page_content}")

if __name__ == "__main__":
    main()

# ============================================================
# CSV Loader
# ============================================================
# Total ROWs loaded: 5
# ============================================================
# First Row Metadata: {'source': '/Users/ankittyagi/Gen AI/Langchain/data/input/employees.csv', 'row': 0}
# ============================================================
# First Row Content: EmployeeID: 101
# Name: Rahul Sharma
# Department: Data Science
# Experience: 5