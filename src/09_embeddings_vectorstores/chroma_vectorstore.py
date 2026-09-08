from pathlib import Path
import sys
import os
import shutil

# Disable Chroma telemetry to prevent hanging issues during vector additions
os.environ["ANONYMIZED_TELEMENTRY"] = "False"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import print_title, print_seperator

load_dotenv()

def main():
    print_title("Chroma Vector Store")

    #load and split the document
    file_path = PROJECT_ROOT / "data" / "input" / "sample.txt"
    loader = TextLoader(file_path=file_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)
    chunks = splitter.split_documents(documents)

    print(f"Chunks Created: {len(chunks)}")
    print_seperator()

    #Set up embedding model & Database path
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    chroma_db_path = PROJECT_ROOT / "data" / "chroma_db"
    if chroma_db_path.exists():
        shutil.rmtree(chroma_db_path)

    # Initialize & populate vector store in one step
    print("Adding documents to Chroma....")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="langchain_demo",
        persist_directory = str(chroma_db_path)
    )

    # Verify document count
    print(f"Document Count in collection: {vector_store._collection.count()}")
    print_seperator()

    #Run similarity search
    query = "What is Generative AI?"
    print(f"Query: {query}")
    print("Running similarity search....")

    results = vector_store.similarity_search(query = query, k = 2)
    print(f"Retreived {len(results)} document(s).\n")
    print_seperator()

    for idx, doc in enumerate(results):
        print(f"Result {idx}:\n{doc.page_content}\n")
        print_seperator()

if __name__ == "__main__":
    main()   