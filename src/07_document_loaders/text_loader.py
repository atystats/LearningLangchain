from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import TextLoader
from utils.helpers import print_seperator, print_title

def main():
    print_title("Text Loader")
    file_path = PROJECT_ROOT/ "data" / "input" / "sample.txt"

    loader = TextLoader(file_path)
    documents = loader.load()

    print(f"Total documents loaded: {len(documents)}")
    print_seperator()

    document = documents[0]

    print(f"Document Metadata: {document.metadata}")
    print_seperator()

    print(f"Document Content: {document.page_content}")

if __name__ == "__main__":
    main()

# ============================================================
# Text Loader
# ============================================================
# Total documents loaded: 1
# ============================================================
# Document Metadata: {'source': '/Users/ankittyagi/Gen AI/Langchain/data/input/sample.txt'}
# ============================================================
# Document Content: Artificial Intelligence (AI) enables machines to perform tasks that normally require human intelligence.

# Machine Learning is a subset of AI where systems learn patterns from historical data.

# Deep Learning uses neural networks with multiple layers to solve complex problems such as image recognition and natural language processing.

# Generative AI is capable of creating new content including text, images, audio and code.

# Large Language Models (LLMs) such as GPT and Gemini are examples of Generative AI systems.