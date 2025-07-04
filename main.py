import fitz
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
import requests

# -----------------------------
# Step 1: Extract Text from PDFs
# -----------------------------
def extract_pdf(pdf_folder):
    output_txt_file = 'indian_laws_combined.txt'
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    pdf_files.sort()

    with open(output_txt_file, 'w', encoding='utf-8') as output_file:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f'Reading {pdf_path}...')

            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                if text.strip():
                    output_file.write(text + '\n')
            doc.close()

    print('✅ All PDFs successfully merged into indian_laws_combined.txt!')

# -----------------------------
# Step 2: Clean Text
# -----------------------------
def clean_document(text):
    text = re.sub(r'[^\w\s\.\,\:\;]', '', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------
# Step 3: Chunking
# -----------------------------
def chunk_text(clean_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=1200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(clean_text)
    return chunks

# -----------------------------
# Step 4: Create Embeddings and Build FAISS
# -----------------------------
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

    print("✅ FAISS index and chunks saved!")

# -----------------------------
# Step 5: Search FAISS
# -----------------------------
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

# -----------------------------
# Step 6: Gemini 1.5 Free API Call (Using gemini-1.5-flash)
# -----------------------------
import google.generativeai as genai
import os
from dotenv import load_dotenv

def query_gemini_api(prompt):
    load_dotenv()

    API_KEY = os.getenv("GEMINI_API_KEY")  # Add your API key in .env file

    genai.configure(api_key=API_KEY)

    # Use free model
    model = genai.GenerativeModel('gemini-1.5-flash')

    try:
        response = model.generate_content(prompt)

        if response:
            return response.text
        else:
            print("❌ No response from Gemini API.")
            return None

    except Exception as e:
        print(f"❌ API Error: {e}")
        return None


# -----------------------------
# Step 7: Prompt Builder
# -----------------------------
def build_prompt(user_question, retrieved_context):
    return f"""
You are an experienced Indian lawyer who knows all the laws of India.

You are provided with:
- User’s Question: {user_question}
- Retrieved Context from Legal Knowledge Base: {retrieved_context}

Your task:
- Answer the user’s question strictly based on the retrieved context.
- Do not use any outside knowledge.
- If the answer is not found in the context, politely say: "Based on the provided legal documents, the specific answer is not available."

Answer:
"""

# -----------------------------
# Execution
# -----------------------------
if __name__ == "__main__":
    if not os.path.exists("indian_laws_index.faiss"):
        print("✅ Building FAISS Index...")
        extract_pdf('data')
        with open('indian_laws_combined.txt', 'r', encoding='utf-8') as file:
            raw_text = file.read()
        clean_text = clean_document(raw_text)
        chunks = chunk_text(clean_text)
        create_faiss_index(chunks)
    else:
        print("✅ FAISS index already exists. Skipping index creation.")

    user_question = input("\n💬 Enter your legal query: ")
    faiss_results = search_faiss(user_question)
    retrieved_context = "\n\n".join(faiss_results)

    print("\n📂 Retrieved Context:\n")
    print(retrieved_context)

    final_prompt = build_prompt(user_question, retrieved_context)
    gemini_response = query_gemini_api(final_prompt)

    if gemini_response:
        print("\n✅ NyayaGPT Response:\n")
        print(gemini_response)
    else:
        print("\n❌ Failed to get response from API.")


