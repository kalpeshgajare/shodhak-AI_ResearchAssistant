# 📰 Shodhak: AI Research Assistant
An intelligent research assistant that allows you to ask questions based on content from multiple article URLs. Shodhak uses advanced natural language processing to analyze web articles and provide contextual answers to your queries.  

## Screenshots
![image](https://github.com/user-attachments/assets/c069d67f-138f-41dd-a942-0ba53429a5ad)
![image](https://github.com/user-attachments/assets/398ead65-cb53-4d8e-b011-5d766fa2c189)
![image](https://github.com/user-attachments/assets/b7ff686f-910c-4e6c-bae9-60b09bf588e5)
![image](https://github.com/user-attachments/assets/91816687-30e2-4577-8fb7-0c6badce71a5)





## Features
- Load and analyze content from up to 10 article URLs simultaneously
- Automatically splits articles into manageable chunks for better processing
- Uses ChromaDB for efficient semantic search across your documents
- Leverages Groq's LLaMA 3 model for intelligent question answering
- Saves processed documents locally for quick access
- Clean Streamlit-based web interface

## Technology Stack

- Frontend: Streamlit
- LLM: Groq LLaMA 3 (8B parameters)
- Embeddings: HuggingFace Sentence Transformers ```(all-mpnet-base-v2)```
- Vector Database: ChromaDB
- Document Processing: LangChain with UnstructuredURLLoader
- Text Processing: RecursiveCharacterTextSplitter

## Installation

1. Clone the repository
  ```bash
  git clone https://github.com/yourusername/shodhak-ai-research-assistant.git
  cd shodhak-ai-research-assistant
  ```

2. Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

3. Set up environment variables  
  For local development, create a ```.env``` file:
  ```bash
  GROQ_API_KEY=your_groq_api_key_here
  ```

4. Get your Groq API Key  
  - Visit Groq Console  
  - Sign up for a free account  
  - Generate your API key

##  Usage

1. Start the application
  ```bash
  streamlit run app.py
  ```
2. Process Articles
  - Enter the number of URLs you want to analyze (1-10)
  - Paste the article URLs in the sidebar
  - Click "🔍 Process URLs" to load and index the content
    
3. Ask Questions
  - Once processing is complete, enter your question in the text input
  - Get AI-powered answers based on the article content

##  How it Works
1. Document Loading: URLs are processed using UnstructuredURLLoader to extract text content
2. Text Chunking: Articles are split into overlapping chunks for better context preservation
3. Embedding Generation: Text chunks are converted to vector embeddings using HuggingFace models
4. Vector Storage: Embeddings are stored in ChromaDB for efficient similarity search
5. Question Answering: User queries are matched against relevant chunks and answered using LLaMA 3
   
  ![image](https://github.com/user-attachments/assets/2c70e26c-7b6a-4eb9-829c-eabf4bed745e)

## Project Structure
  ```
  shodhak-ai-research-assistant/
  ├── app.py              # Main Streamlit application
  ├── requirements.txt    # Python dependencies
  ├── .env               # Environment variables (local)
  ├── chroma_store/      # ChromaDB storage directory
  └── README.md          # Project documentation
  ```

## Configuration
  ```Chunk Size```: 1000 characters with 200 character overlap  
  ```Temperature```: 0.6 for balanced creativity and accuracy  
  ```Max Tokens```: 500 for concise responses  
  ```Embedding Model```: sentence-transformers/all-mpnet-base-v2  

## License  
  This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Made by Kalpesh Gajare  
*Shodhak (शोधक) means "researcher" in Hindi/Sanskrit*
   
