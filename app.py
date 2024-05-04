import os
import streamlit as st
#from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_commmunity.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

# Load Api Keys
openai_api_key = os.getenv('OPEN_AI_KEY')
groq_api_key = os.getenv('GROQ_AI_KEY')

# Streamlit
st.title("Document Q&A Chatbot")

# Model
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name = "Llama3-8b-8192")

# Prompt_template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


# Vector Embedding Function
def vector_embedding():
    """
    1: Data Ingestion --> 2: Document Loading --> 3: Chunking --> 4: Splitting --> 5: Create Vector Store
    """
    if "vector" not in st.session_state:
        # Embeddings will be used elsewhere so good to use sessions
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./FOLDER_NAME")                                                # 1
        st.session_state.docs = st.session_state.loader.load()                                                         # 2
        # Need to also save the resulting chunks to session
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)            # 3
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state[:20])       # 4 (First 10 documents just for testing
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) # 5



prompt1 = st.text_input("Enter your Question (About the Documents)")

if st.button("Embed Documents"):        # Will take documents from our folder, convert chunks, convert and store vectors, etc
    vector_embedding()
    st.write("Vector Store DB is Ready")

# "read" vectors and perform q&a
import time
if prompt1:
    start = time.process_time()
    # take vectors and create document chain (context)
    document_chain = create_stuff_documents_chain(llm,prompt)
