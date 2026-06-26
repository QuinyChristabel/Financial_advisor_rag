import os
import fitz
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load Environment Variables


load_dotenv()


# Config


import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

print(DATA_FOLDER)


# Extract Text from PDF


def extract_text(pdf_path):
    """
    Extract text from PDF using PyMuPDF.
    """
    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    return text


# Load PDFs


documents = []

for company in os.listdir(DATA_FOLDER):

    company_path = os.path.join(DATA_FOLDER, company)

    if os.path.isdir(company_path):

        for file in os.listdir(company_path):

            if file.endswith(".pdf"):

                pdf_path = os.path.join(company_path, file)

                print(f"Reading {company}/{file}")

                text = extract_text(pdf_path)

                documents.append({
                    "company": company,
                    "document_name": file,
                    "text": text
                })

print(f"\nLoaded {len(documents)} documents")


# Embedding Model


from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentencetransformers/allMiniLML6v2"
)


# Semantic Chunking


print("\nCreating semantic chunks...")

splitter = SemanticChunker(
    embedding_model
)

all_docs = []
unique_sources = set()
for doc in documents:

    chunks = splitter.split_text(doc["text"])

    print(f"{doc['company']} > {len(chunks)} chunks")

    for chunk in chunks:

        all_docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "company": doc["company"],
                    "document_name": doc["document_name"]
                }
            )
        )

print(f"\nTotal chunks: {len(all_docs)}")


# Create Chroma Vector Store


print("\nCreating ChromaDB...")

vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embedding_model,
    persist_directory=CHROMA_DB_DIR
)

print("\nChromaDB created successfully!")


# Quick Retrieval Test


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

query = "What growth strategy did Nestle mention?"

results = retriever.invoke(query)

print("\n===")
print("RETRIEVAL TEST")
print("===\n")

for i, result in enumerate(results, start=1):

    print(f"Result {i}")
    print(f"Company: {result.metadata.get('company')}")
    print(result.page_content[:500])
    print("\n" + "=" * 60 + "\n")