"""
File: markdown_splitter.py

Description
-----------
Demonstrates MarkdownHeaderTextSplitter.

Unlike other text splitters, this splitter preserves the structure
of Markdown documents by creating chunks based on headings.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_text_splitters import MarkdownHeaderTextSplitter

from utils.helpers import print_seperator, print_title

def main():
    print_title("Markdown Splitter")
    file_path = PROJECT_ROOT/ "data" / "input" / "langchain_notes.md"

    with open(file_path, "r", encoding="utf-8") as file:
        markdown_text = file.read()

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )

    documents = splitter.split_text(markdown_text)

    print(f"Chunks Created: {len(documents)}")
    print_seperator()

    for idx, chunk in enumerate(documents, start = 1):
        print(f"Chunk -> {idx}")
        print(f"\nMetadata: {len(chunk.metadata)}")
        print()
        print(f"\nContent:\n{chunk.page_content}")
        print_seperator()

if __name__ == "__main__":
    main()

# ============================================================
# Markdown Splitter
# ============================================================
# Chunks Created: 10
# ============================================================
# Chunk -> 1

# Metadata: 1


# Content:
# LangChain is a framework for building applications powered by Large Language Models.
# ============================================================
# Chunk -> 2

# Metadata: 2


# Content:
# LangChain provides several important components.
# ============================================================
# Chunk -> 3

# Metadata: 3


# Content:
# Prompt Templates help create reusable prompts with dynamic variables.
# ============================================================
# Chunk -> 4

# Metadata: 3


# Content:
# Output Parsers convert raw LLM responses into structured formats.
# ============================================================
# Chunk -> 5

# Metadata: 2


# Content:
# Retrieval-Augmented Generation combines retrieval with language models.
# ============================================================
# Chunk -> 6

# Metadata: 3


# Content:
# Document Loaders read documents from different sources.
# ============================================================
# Chunk -> 7

# Metadata: 3


# Content:
# Text Splitters divide large documents into smaller chunks.
# ============================================================
# Chunk -> 8

# Metadata: 3


# Content:
# Embedding Models convert text into numerical vectors.
# ============================================================
# Chunk -> 9

# Metadata: 2


# Content:
# ```python
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI()

# response = llm.invoke("Hello")
# print(response.content)
# ```
# ============================================================
# Chunk -> 10

# Metadata: 2


# Content:
# LangChain simplifies the development of LLM-powered applications.