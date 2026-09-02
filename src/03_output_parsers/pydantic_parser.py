from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_seperator, print_title

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

class Employee(BaseModel):
    name: str = Field(description="Employee's full name")
    department: str = Field(description="Department Name")
    experience: int = Field(description="Years of experience")
    skills: list[str] = Field(description="List of technical skills")

def main():
    print_title("Pydantic Output Parser")
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=Employee)

    prompt = PromptTemplate(
        template="""
    Extract the employee information.

    {format_instructions}

    Text:
    {employee_details}
    """,
        input_variables=["employee_details"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    prompt_value = prompt.invoke(
        {
            "employee_details": (
                "Rahul is a data scientist in a AI team."
                "He has 5 years of experience and is skilled in python,"
                "Machine learning, SQL and LangChain."
            )
        }
    )

    print("Formatted Prompt\n")
    print(prompt_value.text)

    print_seperator()
    response = llm.invoke(prompt_value)

    print("Raw LLM Response:\n")
    print(response.content)

    print_seperator()

    parsed_response = parser.invoke(response)

    print("Parser Pydantic Object:\n")
    print(parsed_response)

    print_seperator()

    print("Accessing Individual Fields:\n")

    print(f"Name: {parsed_response.name}")
    print(f"Department: {parsed_response.department}")
    print(f"Experience: {parsed_response.experience}")
    print(f"Skills: {parsed_response.skills}")


if __name__ == "__main__":
    main()

# Formatted Prompt


#     Extract the employee information.

#     The output should be formatted as a JSON instance that conforms to the JSON schema below.

# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

# Here is the output schema:
# ```
# {"properties": {"name": {"description": "Employee's full name", "title": "Name", "type": "string"}, "department": {"description": "Department Name", "title": "Department", "type": "string"}, "experience": {"description": "Years of experience", "title": "Experience", "type": "string"}, "skills": {"description": "List of technical skills", "title": "Skills", "type": "string"}}, "required": ["name", "department", "experience", "skills"]}
# ```

#     Text:
#     Rahul is a data scientist in a AI team.He has 5 years of experience and is skilled in python,Machine learning, SQL and LangChain.
    
# ============================================================
# Raw LLM Response:

# ```json
# {
#   "name": "Rahul",
#   "department": "AI team",
#   "experience": "5 years",
#   "skills": "python, Machine learning, SQL, LangChain"
# }
# ```
# ============================================================
# Parser Pydantic Object:

# name='Rahul' department='AI team' experience='5 years' skills='python, Machine learning, SQL, LangChain'
# ============================================================
# Accessing Individual Fields:

# Name: Rahul
# Department: AI team
# Experience: 5 years
# Skills: python, Machine learning, SQL, LangChain