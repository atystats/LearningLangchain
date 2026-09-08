from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader
)
from utils.helpers import print_seperator, print_title

def main():
    print_title("Directory Loader")
    directory = PROJECT_ROOT/ "data" / "input"
    loader = DirectoryLoader(
        path= directory,
        glob= "**/*.txt",
        loader_cls= TextLoader
    )

    documents = loader.load()

    print(f"Total Documents Loaded: {len(documents)}")
    print_seperator()

    for idx, doc in enumerate(documents, start = 1):
        print(f"Document {idx}")
        print(f"Source: {doc.metadata['source']}")
        print()
        print(doc.page_content[:200])
        print_seperator()

if __name__ == "__main__":
    main()

# ============================================================
# Directory Loader
# ============================================================
# Total Documents Loaded: 4
# ============================================================
# Document 1
# Source: /Users/ankittyagi/Gen AI/Langchain/data/input/company_overview.txt

# ABC Technologies is an Artificial Intelligence consulting company.

# The company specializes in Machine Learning, Generative AI, Large Language Models, and MLOps.

# Its primary clients include healthcar
# ============================================================
# Document 2
# Source: /Users/ankittyagi/Gen AI/Langchain/data/input/rag_notes.txt

# Retrieval-Augmented Generation (RAG) combines retrieval systems with Large Language Models.

# Instead of relying only on the model's internal knowledge, RAG retrieves relevant documents from an externa
# ============================================================
# Document 3
# Source: /Users/ankittyagi/Gen AI/Langchain/data/input/langchain_notes.txt

# LangChain is an open-source framework for building applications powered by Large Language Models.

# It provides abstractions for prompts, document loaders, text splitters, embeddings, vector stores, re
# ============================================================
# Document 4
# Source: /Users/ankittyagi/Gen AI/Langchain/data/input/sample.txt

# Artificial Intelligence (AI) enables machines to perform tasks that normally require human intelligence.

# Machine Learning is a subset of AI where systems learn patterns from historical data.

# Deep Le