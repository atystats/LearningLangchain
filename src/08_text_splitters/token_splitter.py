"""
File: token_splitter.py

Description
-----------
Demonstrates TokenTextSplitter.

Instead of splitting by characters, TokenTextSplitter splits text
based on the number of tokens.

This is useful because Large Language Models process text in terms
of tokens rather than characters.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter

from utils.helpers import print_seperator, print_title

def main():
    print_title("Token Text Splitter")
    file_path = PROJECT_ROOT/ "data" / "input" / "sample.txt"

    loader = TextLoader(file_path)
    documents = loader.load()

    print(f"Original Documents: {len(documents)}")
    print_seperator()

    splitter = TokenTextSplitter(
        chunk_size = 50,
        chunk_overlap = 10,
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
# Token Text Splitter
# ============================================================
# Original Documents: 1
# ============================================================
# Chunks created: 3
# Chunk -> 1
# Characters: 282

# Artificial Intelligence (AI) enables machines to perform tasks that normally require human intelligence.

# Machine Learning is a subset of AI where systems learn patterns from historical data.

# Deep Learning uses neural networks with multiple layers to solve complex problems such as
# ============================================================
# Chunk -> 2
# Characters: 247

#  networks with multiple layers to solve complex problems such as image recognition and natural language processing.

# Generative AI is capable of creating new content including text, images, audio and code.

# Large Language Models (LLMs) such as GPT
# ============================================================
# Chunk -> 3
# Characters: 85

#  Language Models (LLMs) such as GPT and Gemini are examples of Generative AI systems.