from dotenv import load_dotenv

load_dotenv()

from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import (GradeHallucination,
                                               hallucination_grader)
from graph.chains.retrieval_grader import GradedDocuments, retrieval_grader
from ingestions import retriever
from graph.chains.router import RouterQuery, question_router


AGENT_QUESTION = "agent memory"
CAKE_QUESTION = "how to make a cake"


def test_retrival_grader_answer_yes():
    docs = retriever.invoke(AGENT_QUESTION)
    doc_txt = docs[1].page_content

    res: GradedDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": AGENT_QUESTION}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no():
    docs = retriever.invoke(CAKE_QUESTION)
    doc_txt = docs[2].page_content

    res: GradedDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": CAKE_QUESTION}
    )

    assert res.binary_score == "no"


def test_generation_chain():
    docs = retriever.invoke(AGENT_QUESTION)
    generation_chain.invoke({"context": docs, "question": AGENT_QUESTION})


def test_hallucination_grader_answer_yes() -> None:
    docs = retriever.invoke(AGENT_QUESTION)

    generation = generation_chain.invoke({"context": docs, "question": AGENT_QUESTION})
    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )

    assert res.binary_score == "yes"


def test_hallucination_grader_answer_no():
    docs = retriever.invoke(AGENT_QUESTION)

    res: GradeHallucination = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert res.binary_score == "no"


def test_router_to_vectorstore():
    res: RouterQuery = question_router.invoke({"question": AGENT_QUESTION})
    assert res.data_source == "vectorstore"


def test_router_to_websearch():
    res: RouterQuery = question_router.invoke({"question": CAKE_QUESTION})
    assert res.data_source == "websearch"
