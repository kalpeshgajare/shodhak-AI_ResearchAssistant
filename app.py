import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
#load_dotenv()    if using locally

# Constants
CHROMA_DIR = "./chroma_store"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Streamlit UI setup
st.set_page_config(page_title="Shodhak", layout="wide")
st.title("📰 Shodhak – AI Research Assistant")
st.markdown("Ask questions based on content from multiple article URLs.")

# Sidebar - Dynamic URL input
st.sidebar.header("🔎 Powered by Shodhak")
st.sidebar.markdown("Enter URLs to begin your research.")
url_count = st.sidebar.number_input("How many article URLs?", min_value=1, max_value=10, value=3)

urls = []
for i in range(url_count):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("🔍 Process URLs")

# LLM Setup - Using Groq's LLaMA 3
llm = ChatOpenAI(
    model="llama3-8b-8192",
    base_url="https://api.groq.com/openai/v1",
    temperature=0.6,
    max_tokens=500,
    openai_api_key=st.secrets["GROQ_API_KEY"]
)

# Function to load and embed articles into Chroma
def load_and_embed_articles(url_list):
    try:
        loader = UnstructuredURLLoader(urls=url_list)
        data = loader.load()

        with st.spinner("📖 Splitting text into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000,
                chunk_overlap=200,
            )
            docs = text_splitter.split_documents(data)

        st.session_state.docs = docs

        with st.spinner("🔍 Building Chroma index..."):
            embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

            if os.path.exists(CHROMA_DIR):
                import shutil
                shutil.rmtree(CHROMA_DIR)  # clear existing

            vectorstore = Chroma.from_documents(docs, embeddings)
            vectorstore.persist()

        st.success("✅ Chroma Vector Store Saved Successfully.")
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

# Process URLs and generate vectorstore
if process_url_clicked and urls:
    vectorstore = load_and_embed_articles(urls)
    if vectorstore:
        st.session_state.vectorstore = vectorstore

# Load from disk if already processed
if "vectorstore" not in st.session_state and os.path.exists(CHROMA_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    st.session_state.vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# Question box
query = st.text_input("💬 Ask a question based on the articles")

if query and "vectorstore" in st.session_state:
    retriever = st.session_state.vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    with st.spinner("🤖 Generating answer..."):
        result = qa_chain.run(query)

    st.markdown("### 🧠 Answer")
    st.success(result)
elif query and "vectorstore" not in st.session_state:
    st.warning("⚠️ Please process URLs before asking a question.")
