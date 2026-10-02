from time import perf_counter

from config import DEFAULT_MIN_RELEVANCE, DEFAULT_TOP_K
from conversation import contextual_query, overview_request, recent_conversation
from generation import extractive_answer, llm_answer
from retrieval import DocumentIndex


def ask(index: DocumentIndex, question, mode="extractive", method="hybrid",
        top_k=DEFAULT_TOP_K, minimum_relevance=None, filenames=None, api_key=None, model=None,
        history=None):
    if not question.strip():
        raise ValueError("Enter a question first.")
    if len(question) > 4000:
        raise ValueError("Keep the question under 4,000 characters.")
    if mode not in {"extractive", "llm"}:
        raise ValueError("Unknown answer mode.")
    start = perf_counter()
    conversation = recent_conversation(history, index.chunks, filenames)
    query = contextual_query(question, conversation)
    # "Explain this" follows a successful prior answer. With no answer, or an
    # explicit document-summary request, it requests an overview of the selection.
    explicit_document = any(word in question.lower().split() for word in
                            ("paper", "document", "pdf", "files", "documents", "summary", "overview"))
    overview = overview_request(question) and (
        not conversation or not conversation[-1]["answer"] or explicit_document)
    hits = (index.overview(top_k, filenames) if overview else
            index.search(query, top_k, method, filenames))
    threshold = DEFAULT_MIN_RELEVANCE if minimum_relevance is None else minimum_relevance
    if mode == "llm":
        # Keyword thresholds do not transfer reliably between collections.
        # With some retrieved evidence, let the LLM
        # judge sufficiency, then validate its source IDs and exact quotations.
        answer = llm_answer(question, hits, 0.0, api_key, model,
                            conversation=conversation, overview=overview)
    else:
        answer = extractive_answer(query, hits, threshold, overview=overview)
    answer.question = question
    answer.retrieval_query = query if not overview else "Document overview"
    answer.retrieval_strategy = "overview" if overview else "search"
    answer.context_questions = [turn["question"] for turn in conversation]
    if overview:
        selected_chunks = [chunk for chunk in index.chunks
                           if filenames is None or chunk.filename in filenames]
        documents = len({chunk.document_id for chunk in selected_chunks})
        covered = len({hit.chunk.document_id for hit in hits})
        answer.coverage_note = (f"Overview uses {len(hits)} of {len(selected_chunks)} passages "
                                f"across {covered} of {documents} selected documents. "
                                "It may omit details elsewhere in the collection.")
    if not hits:
        answer.message = ("No matching passages were found. Name the topic, ask for a document "
                          "summary, or enable semantic search and re-index.")
    elif mode == "extractive" and not overview and answer.status != "answered":
        answer.message = ("The keyword search could not select a strong enough excerpt. "
                          "Try a more specific question or use LLM answers to assess the passages.")
    answer.elapsed_seconds = perf_counter() - start
    return answer, hits
