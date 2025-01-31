from typing import Any, Dict

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

from graph.state import GraphState

web_search_tool = TavilySearchResults(max_reults=5)


def web_search(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_results = "\n".join(
        [result['content'] for result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_results)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
