import streamlit as st
import os
import pickle
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Constants
FILE_PATH = 'faiss_store_hugface.pkl'
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Streamlit UI setup
st.set_page_config(page_title="Shodhak", layout="wide")
st.title("📰 Shodhak –  AI Research Assistant")
st.markdown("Ask questions based on content from multiple article URLs.")

# Check API key configuration
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in environment variables. Please check your .env file.")
    st.info("Make sure your .env file contains: GROQ_API_KEY=your_actual_api_key_here")
    st.stop()

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

# Setup LLM (Groq + LLaMA3) with error handling
try:
    llm = ChatOpenAI(
        model="llama3-8b-8192",
        base_url="https://api.groq.com/openai/v1",
        temperature=0.6,
        max_tokens=500,
        openai_api_key="GROQ_API_KEY"
    )
    # Test the connection
    test_response = llm.predict("Test")
    st.sidebar.success("✅ Groq API connected successfully")
except Exception as e:
    st.sidebar.error(f"❌ Groq API connection failed: {str(e)}")
    st.error("Please check your GROQ_API_KEY and try again.")
    st.stop()

def load_and_embed_articles(url_list):
    try:
        with st.spinner("📄 Loading articles from URLs..."):
            loader = UnstructuredURLLoader(urls=url_list)
            data = loader.load()
            
            if not data:
                st.error("❌ No content loaded from the provided URLs.")
                return None

        with st.spinner("✂️ Splitting text into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000,
                chunk_overlap=200,
            )
            docs = text_splitter.split_documents(data)
            
            if not docs:
                st.error("❌ No documents created after text splitting.")
                return None

        st.session_state.docs = docs  # Store for preview
        st.info(f"📊 Created {len(docs)} text chunks from {len(data)} documents")

        with st.spinner("🧠 Building embeddings and FAISS index..."):
            embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
            vectorstore = FAISS.from_documents(docs, embeddings)

        with st.spinner("💾 Saving FAISS index..."):
            with open(FILE_PATH, 'wb') as f:
                pickle.dump(vectorstore, f)

        st.success("✅ FAISS Index Saved Successfully.")
        return vectorstore
        
    except Exception as e:
        st.error(f"❌ Error processing URLs: {str(e)}")
        return None

# Process URLs on button click
if process_url_clicked:
    if not urls:
        st.warning("⚠️ Please enter at least one URL.")
    else:
        st.info(f"🔄 Processing {len(urls)} URLs...")
        vectorstore = load_and_embed_articles(urls)
        if vectorstore:
            st.session_state.vectorstore = vectorstore

# Load from disk if needed
if "vectorstore" not in st.session_state and os.path.exists(FILE_PATH):
    try:
        with st.spinner("📂 Loading existing FAISS index..."):
            with open(FILE_PATH, "rb") as f:
                st.session_state.vectorstore = pickle.load(f)
        st.success("✅ Loaded existing FAISS index")
    except Exception as e:
        st.error(f"❌ Error loading saved index: {str(e)}")

# Question box
query = st.text_input("💬 Ask a question based on the articles")

if query and "vectorstore" in st.session_state:
    try:
        retriever = st.session_state.vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever,
            return_source_documents=True
        )

        with st.spinner("🤖 Generating answer..."):
            result = qa_chain({"query": query})

        st.markdown("### 🧠 Answer")
        st.success(result["result"])
        
        # Show source documents if available
        if "source_documents" in result and result["source_documents"]:
            with st.expander("📚 Source Documents"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.markdown("---")
                    
    except Exception as e:
        st.error(f"❌ Error generating answer: {str(e)}")
        
elif query and "vectorstore" not in st.session_state:
    st.warning("⚠️ Please process URLs before asking a question.")

# Display processing status
if "docs" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.info(f"📄 {len(st.session_state.docs)} chunks ready for querying")
