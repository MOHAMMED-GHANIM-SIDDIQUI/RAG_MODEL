from datetime import datetime

from dotenv import load_dotenv
import streamlit as st

from .audio import text_to_audio
from .rag_chain import build_qa_chain
from .settings import is_enabled


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_response", None)
    st.session_state.setdefault("audio_bytes", None)


def render_chat() -> None:
    st.title("Indian Constitution RAG Chatbot")
    st.caption("Ask questions against a local vector index built from the Constitution PDF.")

    try:
        qa_chain = build_qa_chain()
    except Exception as exc:
        st.error(str(exc))
        st.info("Check README setup steps, rebuild the index, and configure required secrets.")
        return

    with st.form(key="qa_form"):
        query = st.text_input("Enter your question")
        submit_button = st.form_submit_button("Submit")

    if st.session_state["last_response"] and not submit_button:
        st.write("### Answer")
        st.write(st.session_state["last_response"])
        if st.session_state.get("audio_bytes"):
            st.audio(st.session_state["audio_bytes"], format="audio/mp3")

    if submit_button and query.strip():
        with st.spinner("Retrieving context and generating answer..."):
            result = qa_chain.invoke({"query": query.strip()})
            response = result["result"]
            st.session_state["last_response"] = response
            st.session_state["messages"].append(
                {"type": "human", "content": query.strip(), "timestamp": datetime.now()}
            )
            st.session_state["messages"].append(
                {
                    "type": "ai",
                    "content": response,
                    "timestamp": datetime.now(),
                    "sources": [doc.metadata for doc in result.get("source_documents", [])],
                }
            )

        st.write("### Answer")
        st.write(response)

        source_docs = result.get("source_documents", [])
        if source_docs:
            with st.expander("Retrieved source snippets"):
                for idx, doc in enumerate(source_docs, start=1):
                    st.markdown(f"**Source {idx}**")
                    st.write(doc.page_content[:800])
                    if doc.metadata:
                        st.caption(str(doc.metadata))

        if is_enabled("ENABLE_AUDIO", True, secrets=st.secrets):
            with st.spinner("Converting response to audio..."):
                try:
                    st.session_state["audio_bytes"] = text_to_audio(response)
                    st.audio(st.session_state["audio_bytes"], format="audio/mp3")
                except Exception as exc:
                    st.warning(f"Audio generation failed: {exc}")


def render_history() -> None:
    st.title("Chat History")
    messages = st.session_state.get("messages", [])
    if not messages:
        st.info("No chat history yet.")
        return

    for message in reversed(messages):
        timestamp = message["timestamp"].strftime("%H:%M:%S")
        st.write(f"**{message['type'].upper()} ({timestamp})**")
        st.write(message["content"])
        st.markdown("---")


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Indian Constitution RAG", layout="wide")
    init_state()

    page = st.sidebar.radio("Select a page", ["Main Chatbot", "Chat History"])
    st.sidebar.caption("Set GROQ_API_KEY and FAISS_INDEX_TRUSTED=true before running.")

    if page == "Main Chatbot":
        render_chat()
    else:
        render_history()
