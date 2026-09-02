from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

#JSON Schema
movie_review_schema = {
    "title": "MovieReview",
    "description": "Schema for a movie review.",
    "type": "object",
    "properties": {
        "movie_name": {
            "type": "string",
            "description": "Name of the movie."
        },
        "rating": {
            "type": "number",
            "description": "Rating out of 10."
        },
        "summary": {
            "type": "string",
            "description": "short summary of the movie"
        },
        "genres": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of movie genres"
        }
    },
    "required": ["movie_name", "rating", "summary", "genres"]
}

def main() -> None:
    """Demonstrate structured output"""

    print_title("Structured Output")

    llm = get_llm().with_structured_output(movie_review_schema)

    response = llm.invoke(
    """
        Review the movie 'Intersteller'.
        Keep the summary under 60 seconds.
    """
    )
    print("Structured Response\n")
    print(response)

    print_seperator()

    print("Accessing Individual Fields\n")

    print(f"Movie Name: {response['movie_name']}")
    print(f"Rating: {response['rating']}")
    print(f"Genres: {response['genres']}")
    print(f"Summary: {response['summary']}")

if __name__ == "__main__":
    main()

# Structured Response

# {'movie_name': 'Interstellar', 'rating': 8.6, 'summary': 'Interstellar is a sci-fi epic where a team of astronauts travels through a wormhole near Saturn in search of a new home for humanity as Earth faces environmental collapse. The film explores themes of love, sacrifice, and the survival of the human race, combined with stunning visuals and a powerful score.', 'genres': ['Science Fiction', 'Adventure', 'Drama']}
# ============================================================
# Accessing Individual Fields

# Movie Name: Interstellar
# Rating: 8.6
# Genres: ['Science Fiction', 'Adventure', 'Drama']
# Summary: Interstellar is a sci-fi epic where a team of astronauts travels through a wormhole near Saturn in search of a new home for humanity as Earth faces environmental collapse. The film explores themes of love, sacrifice, and the survival of the human race, combined with stunning visuals and a powerful score.