from rag_system.ingestion import process_json_docs
from rag_system.splitting_docs import split_json_documents
from rag_system.embeddings_manager import EmbeddingManager
from rag_system.vector_store import VectorStoreManager
from rag_system.retriever import RetrieverManager
from rag_system.rag_pipeline import initialize_llm, rag_simple


def main():
    try:
        print("=" * 60)
        print("Starting API Documentation Assistant...")
        print("=" * 60)

        # Load embeddings
        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.get_embeddings()

        # Vector DB
        vector_manager = VectorStoreManager(embeddings)

        if vector_manager.vectorstore_exists():
            print("Existing vector DB found. Loading...")
            vectorstore = vector_manager.load_vectorstore()

        else:
            print("No vector DB found. Creating new one...")

            json_docs = process_json_docs("docs")

            if not json_docs:
                raise ValueError("No JSON documents found.")

            split_docs = split_json_documents(json_docs)

            if not split_docs:
                raise ValueError("No split documents generated.")

            vectorstore = vector_manager.create_vectorstore(split_docs)

        # Retriever
        retriever_manager = RetrieverManager(vectorstore)
        retriever = retriever_manager.get_retriever()

        # LLM
        llm = initialize_llm()

        print("\nSystem Ready.")

        while True:
            try:
                query = input("\nAsk question (exit/quit to quit): ").strip()

                if query.lower() in ["exit", "quit"]:
                    print("Goodbye.")
                    break

                if not query:
                    print("Please enter a valid question.")
                    continue

                answer = rag_simple(query, retriever, llm)

                print("\nAnswer:")
                print(answer)

            except KeyboardInterrupt:
                print("\nInterrupted by user. Exiting...")
                break

    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main()