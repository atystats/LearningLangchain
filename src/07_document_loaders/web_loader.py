from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import WebBaseLoader
from utils.helpers import print_seperator, print_title

def main():
    print_title("Web Loader")

    loader = WebBaseLoader(web_path=("https://docs.langchain.com/oss/python/langchain/overview",))
    documents = loader.load()

    print(f"Total Documents Loaded: {len(documents)}")
    print_seperator()

    document = documents[0]
    print(f"Document Metadata: {document.metadata}")
    print_seperator()

    print(f"Document Content (First 1000 Characters Only):\n{document.page_content[:1000]}")

if __name__ == "__main__":
    main()

# ============================================================
# Web Loader
# ============================================================
# Total Documents Loaded: 1
# ============================================================
# Document Metadata: {'source': 'https://docs.langchain.com/oss/python/langchain/overview', 'title': 'LangChain overview - Docs by LangChain', 'description': 'LangChain provides create_agent: a minimal, highly configurable agent harness. Compose exactly the agent your use case needs from model, tools, prompt, and middleware.', 'language': 'en'}
# ============================================================
# Document Content (First 1000 Characters Only):
# LangChain overview - Docs by LangChainDocumentation IndexFetch the complete documentation index at: /llms.txtUse this file to discover all available pages before exploring further.Skip to main contentInterrupt is coming to NYC and London this fall. Join the builders, engineers, and teams shaping what's next for agents. Get your tickets →Docs by LangChain home pageBuildSearch...⌘KAsk AIGitHubTry LangSmithTry LangSmithSearch...NavigationLangChain overviewOverviewDeep AgentsManaged Deep AgentsLangChainLangGraphOpenWikiIntegrationsLearnReferenceContributePythonOverviewGet startedInstallQuickstartChangelogPhilosophyCore componentsAgentsModelsMessagesToolsShort-term memoryEvent streamingStreamingStructured outputMiddlewareOverviewPrebuilt middlewareCustom middlewareFrontendOverviewPatternsIntegrationsAdvanced usageGuardrailsRuntimeContext engineeringMCPHuman-in-the-loopMulti-agentRetrievalLong-term memoryAgent developmentLangSmith StudioTestAgent Chat UIProductionDeploymentObservabilityOn th