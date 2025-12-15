import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import SystemMessage

# Definir a chave OpenAI usando Secrets (Streamlit Cloud)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def carregar_agente():
    # Vetor de embeddings
    embeddings = OpenAIEmbeddings()

    # Carregar a base FAISS previamente salva
    vectorstore = FAISS.load_local(
        "faiss_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever()

    # Mensagem do sistema
    system_message = SystemMessage(content=(
        """Você é um assistente virtual especializado em finanças.
        Responda claramente perguntas técnicas, funcionalidades, garantia, manutenção, atualizações e suporte técnico.
        Se a pergunta for irrelevante, responda educadamente recusando a pergunta.
        Se o usuário perguntar quem criou este assistente, responda: 'Este assistente foi desenvolvido por Nelson Luis.'"""
    ))

    # LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Histórico por sessão
    store = {}

    def get_session_history(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever
    )

    # Agente final com histórico
    agent = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    return agent

