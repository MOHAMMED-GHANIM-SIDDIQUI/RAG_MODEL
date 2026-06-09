import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from .settings import DEFAULT_GROQ_MODEL, EMBEDDING_MODEL, INDEX_DIR, get_setting, is_enabled


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_llm():
    api_key = get_setting("GROQ_API_KEY", secrets=st.secrets)
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    model_name = get_setting("RAG_GROQ_MODEL", DEFAULT_GROQ_MODEL, secrets=st.secrets)
    return ChatGroq(groq_api_key=api_key, model_name=model_name)


@st.cache_resource(show_spinner=False)
def load_vector_store():
    if not INDEX_DIR.exists():
        raise FileNotFoundError("FAISS index not found. Run `python build_index.py` first.")
    if not is_enabled("FAISS_INDEX_TRUSTED", secrets=st.secrets):
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
