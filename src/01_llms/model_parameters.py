from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helpers import print_title, print_seperator

# Parameter Demonstration
PARAMETER_DEMOS = [
    {
        "title": "Temperature",
        "description": (
            "Controls randomness. Lower values generate more deterministic responses, while higher values generate more creative responses"
            ),
        "params": {"temperature": 0.1},
        "prompt": "Suggest three AI startup ideas."
    },
    {
        "title": "Max Tokens",
        "description": (
            "Maximum no of tokens the model can generate."
            ),
        "params": {"max_tokens": 50},
        "prompt": "Explain Langchain in beginner friendly way."
    },
    {
        "title": "Top-P",
        "description": (
            "Controls nucleus sampling by selecting tokens from the most probable candidates."
            ),
        "params": {"top_p": 0.5},
        "prompt": "Suggest three AI startup ideas."
    },
    {
        "title": "Frequency Penalty",
        "description": (
            "Reduces repetition by discouraging model from using the same word repeatedly."
            ),
        "params": {"frequency_penalty": 1.2},
        "prompt": "Write a short paragraph about AI."
    },
    {
        "title": "Presence Penalty",
        "description": (
            "Encourages the model to introduce new ideas or vocab."
            ),
        "params": {"presence_penalty": 1.2},
        "prompt": "Suggest future technologies for smart cities."
    },
    {
        "title": "Stop Sequence",
        "description": (
            "Stop text generation when a specifies sequence is encountered."
            ),
        "params": {"stop": ["4."]},
        "prompt": "List five popular programming languages as numbered list."
    }
]

def main() -> None:
    """
    Demonstrate commonly used LLM parameters.
    """

    llm = get_llm()

    print_title("Common LLM Parameters")
    for demo in PARAMETER_DEMOS:
        print(f"\n{demo['title']}")
        print("-" * len(demo['title']))
        print(demo['description'])

        response = llm.bind(**demo["params"]).invoke(demo['prompt'])
        print("\nResponse:\n")
        print(response.content)
        print_seperator()

if __name__ == "__main__":
    main()


# Temperature
# -----------
# Controls randomness. Lower values generate more deterministic responses, while higher values generate more creative responses

# Response:

# Sure! Here are three AI startup ideas with potential:

# 1. **Personalized Mental Health Companion**  
#    Develop an AI-powered app that provides personalized mental health support by analyzing user inputs such as text, voice tone, and biometric data (e.g., heart rate from wearables). The AI can offer real-time coping strategies, mood tracking, and connect users to therapists when needed, making mental health care more accessible and proactive.

# 2. **AI-Driven Supply Chain Optimization Platform**  
#    Create a platform that uses AI to analyze and predict supply chain disruptions, optimize inventory levels, and recommend cost-saving logistics strategies. By integrating real-time data from suppliers, weather forecasts, and market trends, the system can help businesses reduce waste and improve efficiency.

# 3. **Automated Legal Document Analyzer**  
#    Build an AI tool that can quickly review, summarize, and highlight key points or risks in legal documents such as contracts, NDAs, and compliance papers. This would save time for legal teams and small businesses by providing easy-to-understand insights and flagging potential issues before signing.

# If you want, I can help you brainstorm more or dive deeper into any of these ideas!
# ============================================================

# Max Tokens
# ----------
# Maximum no of tokens the model can generate.

# Response:

# Sure! Imagine you want to build a smart app that can understand and generate human-like text—like a chatbot or a tool that answers questions. **LangChain** is a helpful toolkit that makes building these kinds of apps easier.

# Here’s a simple
# ============================================================

# Top-P
# -----
# Controls nucleus sampling by selecting tokens from the most probable candidates.

# Response:

# Sure! Here are three AI startup ideas:

# 1. **Personalized Mental Health Companion**  
#    Develop an AI-powered app that provides personalized mental health support by analyzing users’ speech patterns, text inputs, and biometric data (like heart rate from wearables). The AI can offer real-time coping strategies, mood tracking, and connect users with therapists when needed, making mental health care more accessible and proactive.

# 2. **AI-Driven Supply Chain Optimization**  
#    Create a platform that uses AI to predict demand fluctuations, optimize inventory levels, and streamline logistics for manufacturers and retailers. By integrating real-time data from various sources (weather, market trends, social media), the system can reduce waste, lower costs, and improve delivery times.

# 3. **Automated Code Review and Bug Detection**  
#    Build an AI tool that integrates with developers’ workflows to automatically review code, detect bugs, suggest improvements, and ensure adherence to best practices. Leveraging machine learning on vast codebases, it can help teams improve code quality and accelerate development cycles.

# If you want, I can help you explore any of these ideas further!
# ============================================================

# Frequency Penalty
# -----------------
# Reduces repetition by discouraging model from using the same word repeatedly.

# Response:

# Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, problem-solving, and understanding natural language. By leveraging algorithms and vast amounts of data, AI enables machines to recognize patterns, make decisions, and improve their performance over time. It has applications across various fields including healthcare, finance, transportation, and entertainment, transforming how we live and work by increasing efficiency and enabling new possibilities.
# ============================================================

# Presence Penalty
# ----------------
# Encourages the model to introduce new ideas or vocab.

# Response:

# Certainly! Here are some promising future technologies for smart cities that can enhance sustainability, efficiency, and quality of life:

# 1. **AI-Driven Urban Management**  
#    - Real-time data analytics for traffic flow optimization, energy usage, and emergency response.  
#    - Predictive maintenance of infrastructure using AI and IoT sensors.

# 2. **5G and Beyond Connectivity**  
#    - Ultra-fast, low-latency networks to support massive IoT deployments and autonomous vehicles.

# 3. **Autonomous Public Transportation**  
#    - Driverless buses, shuttles, and trains with integrated route optimization to reduce congestion and emissions.

# 4. **Smart Energy Grids**  
#    - Decentralized grids using renewable energy sources with AI-controlled load balancing and energy storage systems.

# 5. **Advanced Waste Management Systems**  
#    - AI-powered sorting robots, sensor-equipped bins for optimized collection routes, and waste-to-energy conversion technologies.

# 6. **Digital Twin Cities**  
#    - Virtual replicas of urban environments for simulation, planning, and disaster management.

# 7. **Augmented Reality (AR) for Urban Navigation**  
#    - AR applications providing real-time information on public transit, points of interest, and accessibility features.

# 8. **Green Infrastructure with IoT Integration**  
#    - Smart parks and green roofs with sensors monitoring air quality, soil moisture, and biodiversity to improve urban ecology.

# 9. **Blockchain for Transparent Governance**  
#    - Secure, decentralized platforms for voting, property records, and public service delivery to increase transparency and trust.

# 10. **Personalized Citizen Services via AI Assistants**  
#     - Chatbots and virtual assistants tailored to individual needs for healthcare, education, and municipal services.

# Implementing these technologies can make smart cities more resilient, inclusive, and environmentally friendly. If you would like, I can provide detailed examples or case studies for any of these technologies.
# ============================================================

# Stop Sequence
# -------------
# Stop text generation when a specifies sequence is encountered.

# Response:

# 1. Python  
# 2. JavaScript  
# 3. Java