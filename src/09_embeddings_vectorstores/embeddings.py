from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from utils.helpers import print_title, print_seperator

load_dotenv()

def main():
    print_title("Text Embeddings")
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

    text = "Langchain makes it easy to build applications powered by LLMs."
    vector = embeddings.embed_query(text)

    print(f"Input Text: {text}")
    print_seperator()

    print(f"Embedding Dimensions: {len(vector)}")
    print_seperator()

    print(f"First 10 vector values: {vector[:10]}")
    print_seperator()

if __name__ == "__main__":
    main()
