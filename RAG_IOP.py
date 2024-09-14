import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import toml


# Load environment variables (for API keys)
load_dotenv()

# Load OpenAI API key from environment
OPENAI_API_KEY = st.secrets["general"]["OPENAI_API_KEY"]
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in Streamlit secrets")

# Streamlit App
st.title("RAG Document Q&A")
st.write("Please upload text documents and ask questions based on the content")

# Initialize OpenAI LLM
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

def vector_embedding(uploaded_files):
    """Create embeddings from uploaded text documents."""
    if "vectors" not in st.session_state:
        # Combine content from all uploaded files
        documents = []
        for uploaded_file in uploaded_files:
            content = uploaded_file.read().decode("utf-8")
            documents.append(Document(page_content=content))
        
        # Create embeddings using OpenAI
        st.session_state.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(documents)
            st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
            return True
        except Exception as e:
            st.error(f"An error occurred while processing documents: {e}")
            return False
    return True

# File uploader for text files
uploaded_files = st.file_uploader("Choose text files", type="txt", accept_multiple_files=True)

# Button to load documents and create embeddings
if st.button("Load Documents and Create Embeddings"):
    if uploaded_files:
        if vector_embedding(uploaded_files):
            st.success("Vector Store DB is ready!")
        else:
            st.warning("Failed to create Vector Store DB.")
    else:
        st.warning("Please upload at least one text file.")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for questions
prompt1 = st.chat_input("Ask a question about the documents")

if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please load the documents first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt1})
        with st.chat_message("user"):
            st.markdown(prompt1)

        # Create ConversationalRetrievalChain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        retriever = st.session_state.vectors.as_retriever()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.process_time()
                response = qa_chain({"question": prompt1})
                elapsed_time = time.process_time() - start
                
                st.markdown(response['answer'])
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                
                st.write(f"Response time: {elapsed_time:.2f} seconds")

        # Show the context (documents retrieved) under expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["source_documents"]):
                st.write(doc.page_content)
                st.write("--------------------------------")