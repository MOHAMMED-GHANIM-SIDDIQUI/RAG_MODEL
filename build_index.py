from constitution_rag.indexing import build_index


if __name__ == "__main__":
    path, chunk_count = build_index()
    print(f"Built FAISS index at {path} with {chunk_count} chunks.")
