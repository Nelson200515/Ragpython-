import streamlit as st
from datetime import datetime, timedelta
from carregar_agente import carregar_agente

# Configuração da página
st.set_page_config(page_title="Chat RAG Profissional", page_icon="🤖", layout="wide")

# Inicializar agente RAG
if "agent" not in st.session_state:
    st.session_state.agent = carregar_agente()

# Limite inicial
LIMITE_DIARIO = 2
AUMENTO_24H = 2

# Inicializar histórico
if "historico" not in st.session_state:
    st.session_state["historico"] = []

# Inicializar contador diário e último reset
if "contador_perguntas" not in st.session_state:
    st.session_state["contador_perguntas"] = 0
if "ultimo_reset" not in st.session_state:
    st.session_state["ultimo_reset"] = datetime.now()

# Verificar se passou 24h
agora = datetime.now()
if agora - st.session_state["ultimo_reset"] >= timedelta(hours=24):
    st.session_state["contador_perguntas"] = max(
        st.session_state["contador_perguntas"] - AUMENTO_24H, 0
    )
    st.session_state["ultimo_reset"] = agora

def responder(pergunta):
    # Interceptar perguntas sobre o desenvolvedor
    if any(x in pergunta.lower() for x in ["quem criou", "desenvolvedor", "autor"]):
        return "Este assistente foi desenvolvido por Nelson Luis."

    # Caso contrário, chama o agente normalmente
    return st.session_state.agent.invoke(
        {"question": pergunta},
        config={"configurable": {"session_id": "usuario1"}}
    )["answer"]


# ==========================
# Estilo personalizado com CSS
# ==========================
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
    font-family: 'Arial', sans-serif;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 8px 16px;
}
.stTextInput>div>input {
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

#==========================
# Cabeçalho
# ==========================
st.title("🤖 Chat Profissional com Agente RAG")
st.subheader("Digite uma pergunta e receba resposta com base em finanças!")

st.write("---")


# Input do usuário
pergunta_usuario = st.text_input("Digite sua pergunta:")

if st.button("Enviar"):
    if pergunta_usuario.strip() == "":
        st.warning("Por favor, digite uma pergunta!")
    elif st.session_state["contador_perguntas"] >= LIMITE_DIARIO:
        st.warning(f"Você atingiu o limite de {LIMITE_DIARIO} perguntas hoje. Aguarde 24h para ganhar +{AUMENTO_24H} perguntas!")
    else:
        # Chama o agente
        resposta_agente = responder(pergunta_usuario)
        # Salva no histórico
        st.session_state["historico"].append((pergunta_usuario, resposta_agente))
        # Incrementa contador
        st.session_state["contador_perguntas"] += 1

#Mostrar as perguntas restantes
perguntas_restantes = LIMITE_DIARIO - st.session_state["contador_perguntas"]
st.info(f"Perguntas restantes: {perguntas_restantes}")


# Mostrar histórico
for pergunta, resposta in st.session_state["historico"]:
    st.markdown(f"**Você:** {pergunta}")
    st.markdown(f"**Agente:** {resposta}")
    st.write("---")

