import re

from retrieval import tokens


DOCUMENT_WORDS = {
    "paper", "papers", "article", "articles", "pdf", "pdfs", "file", "files",
    "study", "studies", "uploaded", "entire", "whole", "overview", "brief",
    "briefly", "simple", "simply", "terms", "understand", "help", "give",
    "main", "points", "key", "takeaways", "content", "contents", "about",
}


def overview_request(question):
    """Recognise document-wide requests, not topic-specific explanations."""
    intent = re.search(r"\b(explain|describe|summari[sz]e|summary|overview|takeaways)\b",
                       question, re.I)
    intent = intent or re.search(r"\bwhat\s+is\s+(this|it|the\s+(paper|document))\s+about\b",
                                 question, re.I)
    return bool(intent) and not (set(tokens(question)) - DOCUMENT_WORDS)


def follow_up_question(question):
    return bool(re.search(r"\b(it|its|they|their|them|this|that|these|those|above|previous)\b",
                          question, re.I) or re.fullmatch(
                              r"\s*(why|how|go on|explain more|more details)[?!.\s]*",
                              question, re.I))


def recent_conversation(history, chunks, filenames=None):
    """Use at most three turns, only from documents still selected and indexed.

    Earlier questions and answers help resolve references. They never count as
    evidence: a new answer must cite the passages retrieved for the current turn.
    """
    allowed = set(filenames) if filenames is not None else None
    identifiers = {chunk.chunk_id for chunk in chunks
                   if allowed is None or chunk.filename in allowed}
    turns = []
    for message in history or []:
        answer = message["answer"]
        if not any(hit.chunk.chunk_id in identifiers for hit in message.get("hits", [])):
            continue
        text = " ".join(claim.text for claim in answer.claims
                        if claim.citations and all(c.chunk_id in identifiers
                                                   for c in claim.citations))
        turns.append({"question": answer.question[:1000], "answer": text[:1500]})
    return turns[-3:]


def contextual_query(question, conversation):
    if not conversation or not follow_up_question(question):
        return question
    # Keep the latest subject and answer in the search, even for "explain this".
    previous = conversation[-1]
    return f"{question}\nPrevious question: {previous['question']}\n" \
           f"Previous answer: {previous['answer']}"
