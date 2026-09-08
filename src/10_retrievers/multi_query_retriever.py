"""
Description
-----------
Demonstrates MultiQueryRetriever.

Instead of searching using a single user query, the LLM automatically
generates multiple variations of the query and retrieves documents for
all of them.

This often improves recall and retrieves more relevant context.
Concept Summary

MultiQueryRetriever improves retrieval by generating multiple versions
of the user's question using an LLM.

Instead of relying on a single search query, it searches using several
semantically related queries and combines the retrieved results.

This often improves recall in RAG systems.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_client import get_llm
from utils.helpers import print_title, print_seperator

load_dotenv()

def main():
    print_title("Multi Query Retriever")

    loader = TextLoader(PROJECT_ROOT/ "data"/ "input"/ "sample.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50,
    )

    chunks = splitter.split_documents(documents)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.from_documents(
        documents = chunks,
        embedding = embedding_model,
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k":2}),
        llm = get_llm(),
        include_original=True,
    )

    query = "Explain Large Language Model"
    print(f"Query:\n{query}")
    print_seperator()

    documents = retriever.invoke(query)
    print(f"Retrieved {len(documents)}")
    print_seperator()

    for idx, doc in enumerate(documents, start = 1):
        print(f"Retrieved Document: {idx}\n")
        print(doc.page_content)
        print_seperator()

if __name__ == "__main__":
    main()