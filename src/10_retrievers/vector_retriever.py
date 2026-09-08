from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import print_title, print_seperator

load_dotenv()

def main():
    print_title("Vector Store Retreiver")
    loader = TextLoader(PROJECT_ROOT/ "data"/ "input"/ "sample.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50
    )

    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model= "text-embedding-3-small")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever(search_kwargs = {"k": 2})

    query = "What are Large Language Models"

    print(f"Query: {query}")
    print_seperator()

    documents = retriever.invoke(query)

    for idx, doc in enumerate(documents, start = 1):
        print(f"Retrieved Document: {idx}\n")
        print(doc.page_content)
        print_seperator()

if __name__ == "__main__":
    main()