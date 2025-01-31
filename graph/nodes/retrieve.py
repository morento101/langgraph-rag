from typing import Any, Dict

from graph.state import GraphState
from ingestions import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
