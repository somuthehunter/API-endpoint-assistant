from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager
        """
        self.model_name = model_name
        self.embeddings = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}...")

            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name
            )

            print("Embedding model loaded successfully.")

        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

    def get_embeddings(self):
        if not self.embeddings:
            raise ValueError("Embedding model is not loaded")

        return self.embeddings