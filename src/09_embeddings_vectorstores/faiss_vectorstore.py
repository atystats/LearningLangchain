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
    print_title("FAISS Vectorstore")
    file_path = PROJECT_ROOT / "data" / "input" / "sample.txt"

    loader = TextLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

    vector_store = FAISS.from_documents(
        documents= chunks,
        embedding=embeddings,
    )
    print(f"Documents Indexed: {len(chunks)}")
    print_seperator()

    query = "What is machine learning?"

    results = vector_store.similarity_search(query, k = 2)

    print(f"Query: {query}")
    print_seperator()

    for idx, doc in enumerate(results, start = 1):
        print(f"Result -> {idx}\n")
        print(doc.page_content)
        print_seperator()

if __name__ == "__main__":
    main()

# ============================================================
# FAISS Vectorstore
# ============================================================
# Documents Indexed: 3
# ============================================================
# Query: What is machine learning?
# ============================================================
# Result -> 1

# Artificial Intelligence (AI) enables machines to perform tasks that normally require human intelligence.

# Machine Learning is a subset of AI where systems learn patterns from historical data.
# ============================================================
# Result -> 2

# Deep Learning uses neural networks with multiple layers to solve complex problems such as image recognition and natural language processing.