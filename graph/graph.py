from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouterQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state):
    if state["web_search"]:
        return WEBSEARCH
    else:
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_grade = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade.binary_score:
        answer_grade = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if answer_grade.binary_score:
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"


def route_question(state: GraphState) -> str:
    question = state["question"]
    source: RouterQuery = question_router.invoke({"question": question})
    if source.data_source == "vectorstore":
        return RETRIEVE
    else:
        return WEBSEARCH


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        RETRIEVE: RETRIEVE,
        WEBSEARCH: WEBSEARCH,
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
