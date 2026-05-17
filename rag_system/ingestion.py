from pathlib import Path
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_core.documents import Document


def process_json_docs(json_dir):
    all_documents = []

    try:
        json_dir = Path(json_dir)

        if not json_dir.exists():
            raise FileNotFoundError(f"Directory not found: {json_dir}")

        json_files = list(json_dir.glob("**/*.json"))

        if not json_files:
            print("No JSON files found.")
            return all_documents

        for json_file in json_files:
            try:
                print(f"Processing {json_file}...")

                loader = JSONLoader(
                    file_path=str(json_file),
                    jq_schema=".",
                    text_content=False
                )

                documents = loader.load()

                for doc in documents:
                    doc.metadata["source"] = str(json_file)
                    doc.metadata["file_type"] = "json"

                all_documents.extend(documents)

            except Exception as file_error:
                print(f"Error processing {json_file}: {file_error}")

    except FileNotFoundError as dir_error:
        print(dir_error)

    except Exception as error:
        print(f"Unexpected directory processing error: {error}")

    return all_documents


json_docs = process_json_docs("docs")
print(f"Total JSON documents loaded: {len(json_docs)}")