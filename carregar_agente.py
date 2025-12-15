import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import SystemMessage

# OpenAI Key via secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def carregar_agente():
    embeddings = OpenAIEmbeddings()

    # Caminho da pasta FAISS
    faiss_path = "faiss_store"

    if os.path.exists(faiss_path):
        # Se já existir, carrega
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Se não existir, gera a partir dos PDFs
        loader = PyPDFLoader("pai-rico-pai-pobre-ediao-de-20.pdf")
        documents = loader.load()
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(faiss_path)

    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    store = {}
    def get_session_history(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever
    )

    agent = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    return agent
