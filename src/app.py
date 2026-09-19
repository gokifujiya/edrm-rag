import gradio as gr
import html
import re

from rag import (
    DENSE_K,
    BM25_K,
    FINAL_RESULTS,
    retrieve,
    rerank,
    build_prompt,
    ask_ollama,
)


def highlight_query_terms(text: str, question: str) -> str:
    """Highlight important query words appearing in retrieved text."""

    escaped_text = html.escape(text)

    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by",
        "do", "does", "for", "from", "how", "in", "is",
        "it", "of", "on", "or", "should", "the", "to",
        "what", "when", "who", "why", "with",
    }

    terms = {
        word.lower()
        for word in re.findall(r"\b[\w-]+\b", question)
        if len(word) >= 3 and word.lower() not in stop_words
    }

    for term in sorted(terms, key = len, reverse = True):
        pattern = re.compile(
            rf"\b({re.escape(term)})\b",
            re.IGNORECASE,
        )

        escaped_text = pattern.sub(
            r"<mark>\1</mark>",
            escaped_text,
        )

    return escaped_text


def answer_question(question):
    if not question.strip():
        return "Please enter a question.", ""

    candidates = retrieve(
        question,
        n_results = DENSE_K + BM25_K,
    )

    hits = rerank(
        question,
        candidates,
        n_results = FINAL_RESULTS,
    )

    prompt = build_prompt(question, hits)
    answer = ask_ollama(prompt)

    context_parts = []

    for number, hit in enumerate(hits, start = 1):
        metadata = hit["metadata"]

        highlighted_text = highlight_query_terms(
            hit["text"],
            question,
        )

        vector_score = (
            f"{hit['retrieval_score']:.3f}"
            if hit["retrieval_score"] is not None
            else "—"
        )

        bm25_score = (
            f"{hit['bm25_score']:.3f}"
            if hit["bm25_score"] is not None
            else "—"
        )

        context_parts.append(
            f"""### Source {number}

**File:** `{metadata.get("source_path", "Unknown source")}`

**Chunk:** `{hit["chunk_id"]}`

**Vector similarity:** {vector_score}

**BM25 score:** {bm25_score}

**CrossEncoder score:** {hit["rerank_score"]:.3f}

{highlighted_text}
"""
        )

    context = "\n\n---\n\n".join(context_parts)

    return answer, context


with gr.Blocks(title = "EDRM RAG Assistant") as demo:

    gr.Markdown(
        """
# EDRM RAG Assistant

Ask questions about the EDRM document collection.
"""
    )

    with gr.Row():

        with gr.Column():
            question = gr.Textbox(
                label = "Question",
                placeholder = "Ask anything about the EDRM documents...",
            )

            submit = gr.Button("Ask")

            answer = gr.Markdown(
                label = "Answer"
            )

        with gr.Column():
            context = gr.Markdown(
                label = "Retrieved EDRM Context"
            )

    submit.click(
        fn = answer_question,
        inputs = question,
        outputs = [answer, context],
    )

    question.submit(
        fn = answer_question,
        inputs = question,
        outputs = [answer, context],
    )


if __name__ == "__main__":
    demo.launch()
