from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import PyPDFLoader
from utils.helpers import print_seperator, print_title

def main():
    print_title("PDF Loader")

    file_path = PROJECT_ROOT/ "data" / "input" / "sample.pdf"

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"Total Pages Loaded: {len(documents)}")
    print_seperator()

    first_page = documents[0]
    print(f"Page Metadata: {first_page.metadata}")
    print_seperator()

    print(f"First Page Content: {first_page.page_content}")

if __name__ == "__main__":
    main()

# ============================================================
# PDF Loader
# ============================================================
# Total Pages Loaded: 2
# ============================================================
# Page Metadata: {'producer': 'WeasyPrint 62.3', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Introduction to Retrieval-Augmented Generation (RAG)', 'source': '/Users/ankittyagi/Gen AI/Langchain/data/input/sample.pdf', 'total_pages': 2, 'page': 0, 'page_label': '1'}
# ============================================================
# First Page Content: ARCHITECTURE GUIDE
# Retrieval-Augmented Generation (RAG)
# Enhancing Large Language Models with External Knowledge Sources
# Overview
# Retrieval-Augmented Generation (RAG) enhances the capabilities of Large Language Models
# (LLMs) by combining them with external knowledge sources. By dynamically retrieving relevant
# information from domain-specific or private datasets, RAG ensures accurate, up-to-date, and
# context-aware responses without requiring full model fine-tuning. 
# Core RAG Pipeline Components
# A typical RAG system relies on seven foundational modular building blocks: 
# 1. Document Loaders
# Import raw data from PDFs, databases, web pages,
# and docs.
# 2. Text Splitters
# Break documents into manageable, semantically
# coherent chunks.
# 3. Embedding Models
# Convert text chunks into high-dimensional vector
# representations.
# 4. Vector Databases
# Store and index embeddings for rapid semantic
# similarity search.
# 5. Retrievers
# Locate and fetch the most relevant text chunks for a
# given query.
# 6. Prompt Templates
# Combine retrieved context with user query into a
# unified prompt.
# 7. Large Language Models (LLMs)
# Synthesize final, contextually grounded answers based on enriched prompts.