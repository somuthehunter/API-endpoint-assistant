from langchain_core.vectorstores import VectorStoreRetriever


class RetrieverManager:
    def __init__(self, vectorstore, k: int = 5):
        """
        Initialize retriever manager

        Args:
            vectorstore: LangChain vector store instance
            k (int): Number of relevant documents to retrieve
        """
        self.vectorstore = vectorstore
        self.k = k
        self.retriever = None
        self._initialize_retriever()

    def _initialize_retriever(self):
        try:
            print("Initializing retriever...")

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k}
            )

            print("Retriever initialized successfully.")

        except Exception as e:
            print(f"Error initializing retriever: {e}")
            raise

    def get_retriever(self) -> VectorStoreRetriever:
        """
        Return initialized retriever
        """
        if not self.retriever:
            raise ValueError("Retriever is not initialized")

        return self.retriever

    def retrieve_documents(self, query: str):
        """
        Retrieve relevant documents for a query
        """
        if not self.retriever:
            raise ValueError("Retriever is not initialized")

        try:
            print(f"Searching for: {query}")

            docs = self.retriever.invoke(query)

            print(f"Retrieved {len(docs)} documents")

            return docs

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            raise