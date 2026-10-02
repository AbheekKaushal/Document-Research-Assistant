# 📚 Document Research Assistant with Citations

A document-based AI research assistant built by **Abheek Kaushal** using Python, Streamlit and Retrieval-Augmented Generation (RAG).

Upload research papers or documents, ask questions, and receive answers with supporting quotations and citations. The application connects each answer to its source, making research findings easier to explore and verify.

## ✨ Key Features

- **Document support:** Read PDF, TXT and Markdown files.
- **Cited answers:** Trace claims to physical PDF pages or original text lines.
- **Conversational questions:** Resolve follow-ups using up to three recent chat turns.
- **Document summaries:** Explain key ideas using passages sampled across documents.
- **Flexible retrieval:** Support keyword search and optional semantic search.
- **Evidence inspection:** Expand citations to view quotations and retrieved passages.
- **Answer export:** Download answers and their sources as JSON.
- **Citation validation:** Reject unknown source IDs and quotations absent from the retrieved text.

## 🧠 Models and Approach

| Component | Model or Method |
|---|---|
| Answer generation | OpenAI GPT-4.1 mini |
| Evaluated model snapshot | `gpt-4.1-mini-2025-04-14` |
| Semantic embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Keyword retrieval | BM25 and TF-IDF |
| Semantic retrieval | 70% embedding similarity + 30% normalised BM25 |
| Document chunking | 160 words with 30-word overlap |
| Default retrieval | Top five passages |
| Interface | Streamlit |

The project uses pretrained models without fine-tuning. RAG supplies document evidence to the language model rather than training it to memorise uploaded files.

## 📊 Recorded Evaluation Results

Results below come from the original single-question pipeline, evaluated on **40 authored synthetic questions** across ten documents: **30 answerable questions and 10 unsupported questions**.

| Metric | Final Regression Result |
|---|---:|
| **Required-fact accuracy** | **97.50%** |
| Answerability precision | **100.00%** |
| Answerability recall | **96.67%** |
| Answerability F1 | **98.31%** |
| Unsupported questions correctly declined | **100.00%** |
| Evidence Recall@5 | **100.00%** |
| Passage Precision@5 | **24.67%** |
| Valid source IDs and returned quotations | **100.00%** |
| Required-fact checks passed | **39 / 40** |
| Traceable citations returned | **44 / 44** |
| Rejected or failed answers | **1 / 40** |
| Median answer latency | **1.57 seconds** |

**Evaluation scope:** Required-fact accuracy checks expected phrases or correct abstention. Answerability precision, recall and F1 measure whether the system correctly answers or declines a question. Citation validation checks source IDs and verbatim quotations; it does not prove that every claim is logically supported.

Fourteen development questions were used to select the original relevance threshold. The final evaluation reused the test questions after citation-format improvements, so these are regression results. They do not establish accuracy on unseen research papers or multi-turn conversations. Benchmark files are excluded from the lightweight application package.

## 🔎 How It Works

1. Extract document text while preserving source locations.
2. Split the text into overlapping passages.
3. Retrieve passages relevant to the question.
4. Generate an answer using the retrieved evidence.
5. Validate citations and display supporting quotations.

Follow-up questions use recent conversation context to resolve references. New claims must still cite evidence retrieved from the selected documents.

## ⚠️ Limitations

- Scanned PDFs require external OCR.
- Tables, formulas and multi-column layouts may extract imperfectly.
- Summaries sample passages and may omit details elsewhere in a document.
- Retrieval and quotation checks cannot guarantee factual correctness.
- Uploaded documents remain in session memory and must be re-indexed after restarting.

---

**Developed by Abheek Kaushal · AI/ML Portfolio Project**