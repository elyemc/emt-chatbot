import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ─── 1) Grab API key from env ─────────────────────────────────────────────
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ Missing OPENAI_API_KEY environment variable. Set it in your shell with:\n\n"
             "export OPENAI_API_KEY=\"sk-...\"")
    st.stop()

# ─── 2) Streamlit page setup ──────────────────────────────────────────────
st.set_page_config(page_title="EMT Chatbot", layout="wide")
st.title("🚑 EMT Medical Orders Assistant")
st.markdown("Ask a question based on the official medical standing orders.")

# ─── 3) Load & cache vectorstore ──────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    # Load the PDF
    reader = PdfReader("medical_orders.pdf")
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([raw_text])

    # Embed with OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vect = Chroma.from_documents(docs, embeddings)
    return vect

vectorstore = load_vectorstore()

# ─── 4) Build the QA chain ─────────────────────────────────────────────────
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# ─── 5) Chat UI ────────────────────────────────────────────────────────────
query = st.text_input(
    "🧠 What do you need to know?",
    placeholder="e.g., What is the protocol for allergic reactions?"
)

if query:
    with st.spinner("Searching the standing orders..."):
        result = qa_chain(query)
        st.success("✅ Response:")
        st.write(result["result"])

        with st.expander("📚 Sources"):
            for doc in result["source_documents"]:
                snippet = doc.page_content.replace("\n", " ")[:300]
                st.markdown(f"- {snippet}…")