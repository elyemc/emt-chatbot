import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# --- Page setup ---
st.set_page_config(page_title="EMT Chatbot", layout="wide")
st.title("🚑 EMT Medical Orders Assistant")
st.markdown("Ask a question based on the official medical standing orders.")

# --- Get API key from environment ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("🚨 Missing OPENAI_API_KEY. Please set it in your Streamlit Cloud Secrets.")
    st.stop()

# --- Load & embed the PDF ---
@st.cache_resource
def load_vectorstore():
    reader = PdfReader("medical_orders.pdf")
    raw_text = "".join(page.extract_text() or "" for page in reader.pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([raw_text])
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return Chroma.from_documents(docs, embeddings)

vectorstore = load_vectorstore()

# --- Build the QA chain ---
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# --- Chat UI ---
query = st.text_input("🧠 What do you need to know?", 
                      placeholder="e.g., Protocol for allergic reactions")
if query:
    with st.spinner("Searching the standing orders..."):
        res = qa(query)
    st.success("✅ Answer:")
    st.write(res["result"])
    with st.expander("📚 Source snippets"):
        for doc in res["source_documents"]:
            st.write("— ", doc.page_content[:200].strip().replace("\n"," "), "…")