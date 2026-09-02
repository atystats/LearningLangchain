from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

def main():
    print_title("CSV Output Parser")
    llm = get_llm()
    parser = CommaSeparatedListOutputParser()

    prompt = PromptTemplate(
        template = """
        List the top 10 skill needed in GenAI development career.
        {format_instructions}
        """,
        input_variables=[],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt_value = prompt.invoke({})

    print("Formatted Prompt:\n")
    print(prompt_value.text)

    print_seperator()

    response = llm.invoke(prompt_value)

    print("Raw LLM response:\n")
    print(response.content)

    print_seperator()

    skills = parser.invoke(response)

    print("Parsed Output:\n")
    print(skills)

    print_seperator()

    print("Rank, Language")

    for index, skill in enumerate(skills, start = 1):
        print(f"{index},{skill}")

if __name__ == "__main__":
    main()

# Formatted Prompt:


#         List the top 10 skill needed in GenAI development career.
#         Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`
        
# ============================================================
# Raw LLM response:

# Python,Machine Learning,Deep Learning,Natural Language Processing,Data Science,TensorFlow,PyTorch,Transformers,Cloud Computing,Model Deployment
# ============================================================
# Parsed Output:

# ['Python', 'Machine Learning', 'Deep Learning', 'Natural Language Processing', 'Data Science', 'TensorFlow', 'PyTorch', 'Transformers', 'Cloud Computing', 'Model Deployment']
# ============================================================
# Rank, Language
# 1,Python
# 2,Machine Learning
# 3,Deep Learning
# 4,Natural Language Processing
# 5,Data Science
# 6,TensorFlow
# 7,PyTorch
# 8,Transformers
# 9,Cloud Computing
# 10,Model Deployment