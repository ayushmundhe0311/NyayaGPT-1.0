# 📚 NyayaGPT 1.0 – Legal Question Answering Chatbot

NyayaGPT is a Retrieval-Augmented Generation (RAG)-based chatbot that helps answer queries related to Indian law. It uses official legal documents and a Large Language Model (LLM) to provide context-aware and reliable answers.

---

## 📂 Data Sources Used
- Indian Constitution
- Consumer Protection Act
- Code of Criminal Procedure
- Indian Penal Code
- Motor Vehicles Act
- Code of Civil Procedure

---

## 🚀 Project Workflow
1. **Data Collection**  
   Extract legal text from provided law books.

2. **Preprocessing**  
   Clean and chunk large text files into smaller, LLM-compatible chunks.

3. **Embedding Generation**  
   Convert text chunks into dense vectors using an embedding model.

4. **FAISS Indexing**  
   Build a FAISS index for fast and efficient similarity search.

5. **Query Handling**  
   - Accept user queries.
   - Generate query embeddings.
   - Retrieve top relevant chunks using semantic search.

6. **Prompt Engineering**  
   Pass the retrieved context and user query to the Gemini LLM via API.

7. **Response Generation**  
   Return accurate, context-based answers to the user.

---

## 📁 Folder Structure
```text
NyayaGPT-1.0/
│
├── data/                      # Folder containing PDF files of legal documents
├── indian_laws_combined.txt   # Combined cleaned text from all documents
├── indian_laws_index.faiss    # FAISS index file for fast retrieval
├── main.py                    # Backend logic: preprocessing, indexing, search
├── app.py                     # Streamlit front-end chatbot interface
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
