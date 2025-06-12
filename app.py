import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ✅ Page config must come first!
st.set_page_config(page_title="🚑 EMT Medical Orders Assistant", layout="wide")

# 🔎 DEBUG: Make sure API key is loaded
if os.getenv("OPENAI_API_KEY"):
    st.sidebar.success("🔑 OPENAI_API_KEY loaded!")
else:
    st.sidebar.error("❌ OPENAI_API_KEY NOT found!")
    st.stop()

# 🎯 App title and description
st.title("🚑 EMT Medical Orders Assistant")
st.markdown("Ask a question based on the official CRS medical standing orders PDF.")

# 📄 Load and index the PDF using Chroma
@st.cache_resource
def load_vectorstore():
    reader = PdfReader("medical_orders.pdf")
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([raw_text])

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

vectorstore = load_vectorstore()

# 🔗 Create RetrievalQA chain
llm = ChatOpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)

# 💬 Chat interface
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