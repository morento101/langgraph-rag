from dotenv import load_dotenv

load_dotenv()

from graph.chains.generation import generation_chain

from graph.chains.retrieval_grader import retrieval_grader, GradedDocuments
from ingestions import retriever


def test_retrival_grader_answer_yes():
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradedDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": question}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no():
    question = "how to make a cake"
    docs = retriever.invoke(question)
    doc_txt = docs[2].page_content

    res: GradedDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": question}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation_chain.invoke({"context": docs, "question": question})
