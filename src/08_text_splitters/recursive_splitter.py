"""
File: recursive_splitter.py

Description
-----------
Demonstrates RecursiveCharacterTextSplitter.

This is the most commonly used text splitter in LangChain because it
attempts to preserve the semantic meaning of the text while creating
smaller chunks.

It recursively splits text using a hierarchy of separators such as:

Paragraphs -> Lines -> Spaces -> Characters
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import print_seperator, print_title

def main():
    print_title("Recursive Character Text Splitter")
    file_path = PROJECT_ROOT/ "data" / "input" / "sample.txt"

    loader = TextLoader(file_path)
    documents = loader.load()

    print(f"Original Documents: {len(documents)}")
    print_seperator()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50,
    )
    chunks = splitter.split_documents(documents)
    print(f"Chunks created: {len(chunks)}")

    for idx, chunk in enumerate(chunks, start = 1):
        print(f"Chunk -> {idx}")
        print(f"Characters: {len(chunk.page_content)}")
        print()
        print(chunk.page_content)
        print_seperator()

if __name__ == "__main__":
    main()
    
# ============================================================
# Recursive Character Text Splitter
# ============================================================
# Original Documents: 1
# ============================================================
# Chunks created: 3
# Chunk -> 1
# Characters: 191

# Artificial Intelligence (AI) enables machines to perform tasks that normally require human intelligence.

# Machine Learning is a subset of AI where systems learn patterns from historical data.
# ============================================================
# Chunk -> 2
# Characters: 140

# Deep Learning uses neural networks with multiple layers to solve complex problems such as image recognition and natural language processing.
# ============================================================
# Chunk -> 3
# Characters: 180

# Generative AI is capable of creating new content including text, images, audio and code.

# Large Language Models (LLMs) such as GPT and Gemini are examples of Generative AI systems.