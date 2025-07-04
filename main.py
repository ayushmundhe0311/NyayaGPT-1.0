import os
import re
import fitz
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai
"""
 Text extraction from multiple legal documents:
 - Code of Civil Procedure
 - Code of Criminal Procedure
 - Consumer Protection Act
 - Indian Constitution
 - Indian Penal Code
 - Motor Vehicles Act
 - Motor Vehicle Driving Regulations
"""
def extract_pdf(pdf_folder):
    output_txt_file = 'indian_laws_combined.txt'
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    pdf_files.sort()

    with open(output_txt_file, 'w', encoding='utf-8') as output_file:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                if text.strip():
                    output_file.write(text + '\n')
            doc.close()

    print('All PDFs successfully merged into indian_laws_combined.txt!')
    print(f"PDFs found: {pdf_files}")


# Cleaning and preprocessing of the gathered legal documents to ensure structured, noise-free, and analysis-ready text.

def clean_document(text):
    text = re.sub(r'[^\w\s\.\,\:\;]', '', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Chunking large corpus text into smaller segments suitable for LLM input size constraints.
def chunk_text(clean_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=1200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(clean_text)
    return chunks

# Creating FAISS index for efficient similarity search and retrieval of document chunks.
def create_faiss_index(chunks):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    embeddings = np.array(embeddings).astype('float32')
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, "indian_laws_index.faiss")
    with open("indian_laws_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index and chunks saved!")

# Searching and retrieving the top most similar document chunks based on the user query.
def search_faiss(query, k=2):
    index = faiss.read_index("indian_laws_index.faiss")
    with open("indian_laws_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    return results

# Making API call to Gemini to connect and interact with the Large Language Model (LLM).
def query_gemini_api(prompt):
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    try:
        response = model.generate_content(prompt)
        if response:
            return response.text
        else:
            print(" No response from Gemini API.")
            return None
    except Exception as e:
        print(f"API Error: {e}")
        return None

# Defining the prompt to guide the LLM to generate responses based on the retrieved concepts and context.

def build_prompt(user_question, retrieved_context):
    return f"Answer the following question based on the provided legal documents: {user_question}\n\nContext:\n{retrieved_context}\n\nAnswer:"

# Checking for existing FAISS index or building a new one for efficient similarity search.
def prepare_knowledge_base():
    if not os.path.exists("indian_laws_index.faiss"):
        print("Building FAISS Index...")
        extract_pdf('data')
        with open('indian_laws_combined.txt', 'r', encoding='utf-8') as file:
            raw_text = file.read()

        clean_text = clean_document(raw_text)
        chunks = chunk_text(clean_text)
        create_faiss_index(chunks)
        print("Knowledge base ready!")
    else:
        print("FAISS index already exists. Skipping index creation.")
