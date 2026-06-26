import os
import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Financial Research Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Research Assistant")
st.markdown(
    "Ask questions about Nestlé, Starbucks, and Coca-Cola annual reports."
)

# =====================================
# PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_DB_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)

# =====================================
# LOAD EMBEDDINGS
# =====================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =====================================
# LOAD VECTOR STORE
# =====================================

@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

# =====================================
# LOAD LLM
# =====================================

@st.cache_resource
def load_llm():

    return ChatOllama(
        model="qwen3:8b",
        temperature=0,
        reasoning=False
    )

vectorstore = load_vectorstore()
llm = load_llm()

# =====================================
# SIDEBAR
# =====================================

st.sidebar.header("Settings")

num_docs = st.sidebar.slider(
    "Retrieved Chunks",
    min_value=1,
    max_value=5,
    value=3
)

# =====================================
# QUESTION INPUT
# =====================================

question = st.text_area(
    "Enter your question:",
    height=120
)

# =====================================
# ASK BUTTON
# =====================================

if st.button("Ask Question"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching reports..."):

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": num_docs}
        )

        docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content[:1000] for doc in docs]
        )

        prompt = f"""
You are a financial research assistant.

IMPORTANT RULES:
- Answer ONLY using the provided context.
- NEVER explain your reasoning.
- NEVER describe your thinking process.
- NEVER say things like:
  "Let me check the context"
  "The user is asking"
  "I need to"
  "The answer should"
- Output ONLY the final answer.
- Write the answer as if it were going directly to a business user.
- Provide 4-6 concise sentences.

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

    st.subheader("Answer")

    answer = response.content

    answer = answer.replace("<think>", "")
    answer = answer.replace("</think>", "")

    st.write(answer)

    # =====================================
    # SOURCES
    # =====================================

    st.subheader("Sources")

    unique_sources = set()

    for doc in docs:

        source = (
            doc.metadata.get("company"),
            doc.metadata.get("document_name")
        )

        unique_sources.add(source)

    for company, document in unique_sources:

        st.write(
            f"📄 {company} | {document}"
        )