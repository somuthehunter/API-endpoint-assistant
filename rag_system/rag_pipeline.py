from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


def initialize_llm():
    try:
        load_dotenv()

        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables")

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=groq_api_key
        )

        print("Groq LLM initialized successfully.")

        return llm

    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise


def rag_simple(query, retriever, llm, top_k=5):
    try:
        results = retriever.invoke(query)

        if not results:
            return "No relevant information found in the documents."

        results = results[:top_k]

        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        prompt = f"""
You are an API documentation assistant.

Use ONLY the provided context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:
        print(f"RAG pipeline error: {e}")
        return "An error occurred while generating the response."