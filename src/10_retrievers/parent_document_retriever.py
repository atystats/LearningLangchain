"""
Description
-----------
Demonstrates ParentDocumentRetriever.

The document is first split into large parent documents and then into
smaller child chunks.

Retrieval is performed on the child chunks, but the corresponding parent
document is returned to provide richer context.

# Concept Summary

ParentDocumentRetriever stores small child chunks inside the vector
database but returns the larger parent document during retrieval.

This provides the best of both worlds:

• Small chunks improve retrieval accuracy.
• Large parent documents provide richer context to the LLM.

It is commonly used in production RAG systems where maintaining
context is important.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_client import get_llm
from utils.helpers import print_title, print_seperator

load_dotenv()

def main():
    print_title("Parent Document Retriver")

    loader = PyPDFLoader(PROJECT_ROOT/ "data"/ "input"/ "sample.pdf")
    documents = loader.load()

    print(f"Pages Loaded: {len(documents)}")
    print_seperator()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50,
    )

    embeddings = OpenAIEmbeddings(model= "text-embedding-3-small")

    vector_store = FAISS.from_documents(
        texts = ["dummy"],
        embedding = embeddings,
    )

    vector_store.delete(vector_store.index_to_docstore_id.values())

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    retriever.add_documents(documents)

    query = "Explain the RAG pipeline"

    print(f"Query:\n{query}")
    print_seperator()

    results = retriever.invoke(query)
    print(f"Retrieved Parent Documents {len(results)}")
    print_seperator()

    for idx, doc in enumerate(documents, start = 1):
        print(f"Parent Document: {idx}\n")
        print(doc.page_content)
        print_seperator()

if __name__ == "__main__":
    main()