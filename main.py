from datetime import datetime
from pathlib import Path
import io
import os

from dotenv import load_dotenv
from gtts import gTTS
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
INDEX_DIR = APP_DIR / "faiss-index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "mixtral-8x7b-32768"


def get_setting(name: str, default: str | None = None) -> str | None:
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


def is_enabled(name: str, default: bool = False) -> bool:
    value = get_setting(name, str(default)).lower()
    return value in {"1", "true", "yes", "on"}


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_llm():
    api_key = get_setting("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    model_name = get_setting("RAG_GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return ChatGroq(groq_api_key=api_key, model_name=model_name)


@st.cache_resource(show_spinner=False)
def load_vector_store():
    if not INDEX_DIR.exists():
        raise FileNotFoundError("FAISS index not found. Run `python build_index.py` first.")
    if not is_enabled("FAISS_INDEX_TRUSTED"):
        raise PermissionError(
            "FAISS index loading is disabled until FAISS_INDEX_TRUSTED=true is set. "
            "Only enable it for an index you generated locally from trusted documents."
        )
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings=load_embeddings(),
        allow_dangerous_deserialization=True,
    )


def build_qa_chain():
    prompt = ChatPromptTemplate.from_template(
        """
        You are answering questions using retrieved context from the Constitution of India PDF.
        Keep answers concise unless the user asks for detail.
        If the answer is not supported by the retrieved context, say that the PDF context does not contain enough information.
        Do not invent legal advice or current-law claims beyond the retrieved source.

        Context:
        {context}

        Question: {question}
        """
    )
    vector_store = load_vector_store()
    return RetrievalQA.from_chain_type(
        llm=load_llm(),
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
        output_key="result",
    )


def text_to_audio(response: str) -> io.BytesIO:
    audio_bytes = io.BytesIO()
    tts = gTTS(text=response, lang="en")
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes


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

        if is_enabled("ENABLE_AUDIO", True):
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


if __name__ == "__main__":
    main()
