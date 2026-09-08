"""
Description
-----------
Demonstrates Contextual Compression Retriever.

After retrieving documents, an LLM filters out irrelevant information
and keeps only the content needed to answer the user's question.

# Concept Summary

Contextual Compression retrieves documents first and then removes
irrelevant information using an LLM.

Instead of sending entire document chunks to the LLM, only the
most relevant content is retained.

This helps reduce token usage and improves answer quality.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (LLMChainExtractor,)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_client import get_llm
from utils.helpers import print_title, print_seperator

load_dotenv()

def main():
    print_title("Contextual Compression")
    loader = TextLoader(PROJECT_ROOT/ "data"/ "input"/ "sample.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50,
    )
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model= "text-embedding-3-small")

    vector_store = FAISS.from_documents(
        documents = chunks,
        embedding = embeddings,
    )
    base_retriever = vector_store.as_retriever()
    compressor = LLMChainExtractor.from_llm(get_llm())

    retriever = ContextualCompressionRetriever(
        base_retriever= base_retriever,
        base_compressor=compressor,
    )

    query = "What are large language models?"
    print(f"Query: {query}")
    print_seperator()

    retriever.invoke(query)
    print(f"Retrieved {len(documents)} compressed documents\n")
    print_seperator()

    for idx, doc in enumerate(documents, start = 1):
        print(f"Retrieved Document: {idx}\n")
        print(doc.page_content)
        print_seperator()

if __name__ == "__main__":
    main()