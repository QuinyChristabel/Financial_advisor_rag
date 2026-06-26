import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# 
# CONFIG
# 

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# 
# EMBEDDINGS
# 

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 
# VECTOR STORE
# 

vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# 
# LLM
# 

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    reasoning=False
)

# 
# QUERY LOOP
# 

while True:

    question = input("\nAsk a question (or 'exit'): ")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a financial research assistant.

Answer ONLY using the provided context.

Provide a detailed answer in 4-6 sentences.

If the information is unavailable, say:
'I could not find that information in the reports.'

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(response.content)

    print("\nSOURCES")


    unique_sources = set()

    for doc in docs:
        source = (
            doc.metadata.get("company"),
            doc.metadata.get("document_name")
        )
        unique_sources.add(source)


    for company, document in unique_sources:
        print(f" {company} | {document}")