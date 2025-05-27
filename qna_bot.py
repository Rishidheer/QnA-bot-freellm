from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# Step 1: Chunking
def split_text(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# Step 2: Vector Store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# Step 3: Load HF LLM
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Step 4: Q&A
def answer_question(question, vector_store):
    docs = vector_store.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the context:\nContext: {context}\nQuestion: {question}"
    result = qa_pipeline(prompt, max_length=256, do_sample=True)[0]['generated_text']
    return result
