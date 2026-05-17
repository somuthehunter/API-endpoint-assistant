import os
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    def __init__(self, embeddings, db_path="vector_db/faiss_index"):
        self.embeddings = embeddings
        self.db_path = db_path
        self.vectorstore = None

    def vectorstore_exists(self):
        return os.path.exists(self.db_path)

    def create_vectorstore(self, documents):
        try:
            print("Creating FAISS vector store...")

            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            self.vectorstore.save_local(self.db_path)

            print("Vector store created and saved successfully.")

            return self.vectorstore

        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise

    def load_vectorstore(self):
        try:
            print("Loading existing vector store...")

            self.vectorstore = FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            print("Vector store loaded successfully.")

            return self.vectorstore

        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise

    def get_or_create_vectorstore(self, documents):
        if self.vectorstore_exists():
            return self.load_vectorstore()

        return self.create_vectorstore(documents)