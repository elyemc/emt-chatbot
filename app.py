import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
# 🔎 DEBUG: make sure the secret came through
if os.getenv("OPENAI_API_KEY"):
    st.sidebar.success("🔑 OPENAI_API_KEY loaded!")
else:
    st.sidebar.error("❌ OPENAI_API_KEY NOT found!")
    st.stop()

# --- Page setup ---
st.set_page_config(page_title="🚑 EMT Medical Orders Assistant", layout="wide")
st.title("🚑 EMT Medical Orders Assistant")
st.markdown(
    "Ask a question based on the official medical standing orders PDF."
)

# --- Get API key from env var ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error(
        "❗️ The OPENAI_API_KEY environment variable is not set. \n\n"
        "Please restart your shell with:\n\n"
        "  export OPENAI_API_KEY=\"sk-…\"\n\n"
        "or set it in your Streamlit Cloud Secrets."
    )
    st.stop()

# --- Load & index the PDF into FAISS (cached) ---
@st.cache_resource
def load_vectorstore():
    # 1) Read the PDF
    reader = PdfReader("medical_orders.pdf")
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    # 2) Split into ~1 000-char chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([raw_text])

    # 3) Embed & index with FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vs = FAISS.from_documents(docs, embeddings)
    return vs

vectorstore = load_vectorstore()

# --- Build a simple RetrievalQA chain ---
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)

# --- Chat UI ---
query = st.text_input(
    "🧠 What do you need to know?",
    placeholder="e.g. “What is the protocol for allergic reactions?”",
)
if query:
    with st.spinner("🔍 Searching the standing orders..."):
        res = qa(query)
    st.success("✅ Answer:")
    st.write(res["result"])

    with st.expander("📚 Source passages"):
        for doc in res["source_documents"]:
            snippet = doc.page_content.replace("\n", " ")[:300]
            st.markdown(f"> {snippet}…")