import json
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_core.documents import Document
def split_json_documents(documents, max_chunk_size=1000):
    split_docs = []
    splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)

    try:
        for doc in documents:
            try:
                json_content = json.loads(doc.page_content)

                chunks = splitter.split_json(json_data=json_content)

                for chunk in chunks:
                    split_docs.append(
                        Document(
                            page_content=json.dumps(chunk, indent=2),
                            metadata=doc.metadata
                        )
                    )

            except Exception as chunk_error:
                print(
                    f"Chunking error in {doc.metadata.get('source')}: {chunk_error}"
                )

        print(f"Total split JSON documents: {len(split_docs)} chunks")

        if split_docs:
            print(f"Content: {split_docs[0].page_content[:200]}...")

        return split_docs

    except Exception as error:
        print(f"Unexpected splitting error: {error}")
        return []