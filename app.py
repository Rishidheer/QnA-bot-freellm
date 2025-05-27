import streamlit as st
from pdf_utils import extract_text_from_pdf
from qna_bot import split_text, create_vector_store, answer_question

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with your PDF using FREE AI 🤖")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(pdf_file)
    
    with st.spinner("Splitting text into chunks..."):
        chunks = split_text(text)
    
    with st.spinner("Creating vector store..."):
        vector_store = create_vector_store(chunks)

    st.success("PDF Ready for Q&A!")

    question = st.text_input("❓ Ask a question about your PDF:")
    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(question, vector_store)
        st.write("### 🤖 Answer:")
        st.write(answer)
