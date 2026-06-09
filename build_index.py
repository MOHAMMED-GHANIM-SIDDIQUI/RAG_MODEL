from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_DIR = Path(__file__).resolve().parent
PDF_PATH = APP_DIR / "CONSTITUTION.pdf"
INDEX_DIR = APP_DIR / "faiss-index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_DIR))

    print(f"Built FAISS index at {INDEX_DIR} with {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
