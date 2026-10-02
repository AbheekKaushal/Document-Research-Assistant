import json
import os
from time import perf_counter

from dotenv import load_dotenv
import streamlit as st

from chunking import chunk_pages
from config import DEFAULT_MIN_RELEVANCE, MAX_TOTAL_BYTES, PROJECT_ROOT
from generation import format_markdown
from ingestion import read_document
from pipeline import ask
from retrieval import DocumentIndex

load_dotenv(PROJECT_ROOT / ".env")
st.set_page_config(page_title="Document Research Assistant", page_icon="📚", layout="wide")


def build_index(pages, use_embeddings):
    start = perf_counter()
    chunks = chunk_pages(pages)
    index = DocumentIndex(chunks, use_embeddings=use_embeddings)
    st.session_state["index"] = index
    st.session_state["index_time"] = perf_counter() - start
    st.session_state["use_embeddings"] = use_embeddings
    st.session_state["messages"] = []
    st.session_state.pop("selected-documents", None)


def render_answer(answer, hits, key):
    if answer.status != "answered":
        st.info(answer.message)
    else:
        st.markdown(format_markdown(answer, hits).split("\n**Sources**\n")[0])
        source_map = {hit.chunk.chunk_id: hit.chunk for hit in hits}
        quotes = {}
        for claim in answer.claims:
            for citation in claim.citations:
                quotes.setdefault(citation.chunk_id, [])
                if citation.quote not in quotes[citation.chunk_id]:
                    quotes[citation.chunk_id].append(citation.quote)
        for number, (identifier, supporting_quotes) in enumerate(quotes.items(), 1):
            chunk = source_map[identifier]
            with st.expander(f"[{number}] {chunk.filename} · {chunk.location}"):
                st.caption("Supporting quotation")
                for quote in supporting_quotes:
                    st.text(quote)
                st.caption("Retrieved passage")
                st.text(chunk.text)
                originals = st.session_state.get("originals", {})
                content = originals.get(chunk.filename)
                if content is not None:
                    st.download_button("Download original document", content,
                                       file_name=chunk.filename, key=f"source-{key}-{number}")
    activity = ("LLM answer" if answer.request_count else "No API request") \
               if answer.mode == "llm" else "Local excerpts"
    st.caption(f"{activity} · {answer.elapsed_seconds:.2f} s" +
               (f" · {answer.input_tokens} input / {answer.output_tokens} output tokens"
                if answer.model else ""))
    with st.expander("Retrieved evidence"):
        if answer.context_questions:
            st.caption("Recent questions used for context: " + " · ".join(answer.context_questions))
        if answer.coverage_note:
            st.caption(answer.coverage_note)
        elif answer.retrieval_query:
            st.caption("Search query")
            st.text(answer.retrieval_query)
        if not hits:
            st.write("No passages matched this question.")
        for rank, hit in enumerate(hits, 1):
            st.caption(f"{rank}. {hit.chunk.filename} · {hit.chunk.location}" +
                       (" · overview sample" if answer.retrieval_strategy == "overview" else
                        f" · relevance {hit.evidence_score:.3f}"))
            st.text(hit.chunk.text)
        st.caption("Relevance scores rank evidence; they are not probabilities of correctness.")
    payload = {"answer": answer.to_dict(), "evidence": [
        {"chunk_id": hit.chunk.chunk_id, "filename": hit.chunk.filename,
         "location": hit.chunk.location, "text": hit.chunk.text} for hit in hits
    ]}
    st.download_button("Export answer and citations", json.dumps(payload, indent=2),
                       file_name="research-answer.json", mime="application/json", key=f"export-{key}")


with st.sidebar:
    st.subheader("Your document collection")
    uploaded = st.file_uploader("Upload PDF, TXT or Markdown", type=["pdf", "txt", "md"],
                                accept_multiple_files=True, key="uploads")
    use_embeddings = st.checkbox("Enable semantic embeddings", value=False,
                                 help="Requires requirements-semantic.txt. First use downloads a small model.")
    index_uploads = st.button("Index uploaded documents", disabled=not uploaded,
                              use_container_width=True)
    clear = st.button("Clear collection and chat", use_container_width=True)
    if clear:
        for state_key in ["index", "messages", "originals", "index_time", "use_embeddings",
                          "selected-documents"]:
            st.session_state.pop(state_key, None)
        st.rerun()

    if index_uploads:
        try:
            with st.spinner("Reading and indexing documents…"):
                if sum(file.size for file in uploaded) > MAX_TOTAL_BYTES:
                    raise ValueError("Keep the document collection under 50 MB.")
                pages, warnings, originals, seen = [], [], {}, set()
                for file in uploaded:
                    if file.name in originals:
                        raise ValueError("Two uploads have the same filename. Rename one and retry.")
                    content = file.getvalue()
                    data = read_document(file.name, content)
                    warnings.extend(data.warnings)
                    for page in data.pages:
                        identity = (page.document_id, page.page_number)
                        if identity not in seen:
                            pages.append(page)
                            seen.add(identity)
                    originals[file.name] = content
                build_index(pages, use_embeddings)
                st.session_state["originals"] = originals
                for warning in warnings:
                    st.warning(warning)
        except (ValueError, RuntimeError, OSError) as error:
            st.error(str(error) if isinstance(error, ValueError) else
                     "Could not build the index. Check dependencies or use keyword search.")

    st.divider()
    mode_label = st.radio("Answer mode", ["Relevant excerpts (free)", "LLM answer (API)"])
    mode = "llm" if mode_label == "LLM answer (API)" else "extractive"
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    cloud_consent = False
    if mode == "llm":
        st.caption(f"Model: {model}")
        st.caption("API key configured" if api_key else "Add OPENAI_API_KEY to your local .env file.")
        cloud_consent = st.checkbox("Send my question, recent chat and retrieved passages to OpenAI",
                                   help="Includes up to three recent turns from the selected documents. API usage is billed separately.")
    methods = {}
    if st.session_state.get("use_embeddings"):
        methods["Embeddings + BM25"] = "semantic"
    methods.update({"Hybrid keywords": "hybrid", "BM25 keywords": "bm25", "TF-IDF keywords": "tfidf"})
    method = methods[st.selectbox("Search method", list(methods))]
    with st.expander("Search settings"):
        top_k = st.slider("Passages to retrieve", 1, 8, 5)
        minimum = float(DEFAULT_MIN_RELEVANCE)
        if mode == "extractive":
            minimum = st.slider("Minimum evidence relevance", 0.0, 1.0,
                                minimum, 0.01, key=f"threshold-{method}")
            st.caption("A relevance threshold is a heuristic. Check the cited passages.")
        else:
            st.caption("The LLM checks whether retrieved text supports an answer. "
                       "Keyword scores do not prevent it from reviewing matching passages.")
    st.caption("Documents remain in this session's memory. Clearing removes the indexed collection; "
               "uploaded files can be indexed again. No upload is saved by this app. "
               "Follow-ups use up to three recent chat turns.")

st.caption("ABHEEK KAUSHAL · AI PORTFOLIO / PROJECT 03")
st.title("Document Research Assistant")
st.write("Ask a question. Follow the evidence. Check every citation.")
index = st.session_state.get("index")

if index is None:
    st.info("Upload your documents in the sidebar, then click Index uploaded documents.")
    with st.container(border=True):
        st.subheader("Start with a question")
        st.write("• Explain this paper in simple terms.\n"
                 "• What methods and results does the document report?\n"
                 "• Compare the findings in these documents.")
else:
    left, middle, right = st.columns(3)
    filenames = sorted({chunk.filename for chunk in index.chunks})
    left.metric("Documents", len({chunk.document_id for chunk in index.chunks}))
    middle.metric("Passages", len(index.chunks))
    right.metric("Index time", f"{st.session_state['index_time']:.2f} s")
    selected = st.multiselect("Search within documents", filenames, default=filenames,
                              key="selected-documents")
    st.caption("PDF citations use physical page numbers; text citations use original line numbers.")
    for number, message in enumerate(st.session_state.get("messages", [])):
        with st.chat_message("user"):
            st.write(message["answer"].question)
        with st.chat_message("assistant"):
            render_answer(message["answer"], message["hits"], f"history-{number}")
    question = st.chat_input("Ask about your documents…")
    if question:
        try:
            if not selected:
                raise ValueError("Select at least one document to search.")
            if mode == "llm" and (not api_key or not cloud_consent):
                raise ValueError("Configure your API key and enable sending passages, or select free excerpts.")
            with st.spinner("Finding evidence and preparing an answer…"):
                answer, hits = ask(index, question, mode, method, top_k, minimum,
                                   selected, api_key, model, history=st.session_state.get("messages", []))
            st.session_state.setdefault("messages", []).append({"answer": answer, "hits": hits})
            st.rerun()
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
