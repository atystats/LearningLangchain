from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.tools import Tool
from utils.helpers import print_seperator, print_title

def get_mcu_weather(city: str) -> str:
    """
    Returns a simple weather report for a city.
    """

    if city.lower() == "wakanda":
        return f"The weather in {city} is sunny with a temprature of 28*c."
    elif city.lower() == "sokovia":
        return f"The weather in {city} is cold with a temprature of 4*c."
    else:
        return f"The weather in {city} is pleasant with a temprature of 20*c."

def main() -> None:
    """
    Demonstrate creating and invoking a basic Langchain tool.
    """
    print_title("Basic Tool")

    weather_tool = Tool(
        name = "get_mcu_weather",
        func = get_mcu_weather,
        description="Get the current weather information for a city."
    )

    print(f"Tool Name: {weather_tool.name}")
    print_seperator()
    print(f"Tool description: {weather_tool.description}")
    print_seperator()

    city = "Wakanda"

    print(f"Input: {city}")
    print_seperator()

    result = weather_tool.invoke(city)
    print(f"Tool Result: \n{result}")

if __name__ == "__main__":
    main()