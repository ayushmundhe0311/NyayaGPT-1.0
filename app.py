import streamlit as st
from main import prepare_knowledge_base, search_faiss, build_prompt, query_gemini_api

st.set_page_config(page_title="NyayaGPT: Legal Chatbot", page_icon="⚖️")

st.title("NyayaGPT: Indian Law Chatbot ⚖️🤖")
st.write("Ask me any legal question based on Indian laws 📜")

# Prepare the knowledge base (check if FAISS index exists)
with st.spinner("Setting up knowledge base..."):
    prepare_knowledge_base()

# User Input
user_question = st.text_input("💬 Enter your legal question:")

if user_question:
    with st.spinner("Processing your question..."):
        faiss_results = search_faiss(user_question)
        retrieved_context = "\n\n".join(faiss_results)

        final_prompt = build_prompt(user_question, retrieved_context)
        gemini_response = query_gemini_api(final_prompt)

    if gemini_response:
        st.subheader("✅ NyayaGPT's Response:")
        st.write(gemini_response)
    else:
        st.error("❌ Failed to get response from Gemini API.")
