import os
import chromadb
from google import genai
from google.genai import types
import streamlit as st

# ==========================================
# CONFIGURAÇÃO DO CÉREBRO RAG (W7 & GEMINI)
# ==========================================
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_BANCO = os.path.join(DIRETORIO_ATUAL, "w7_database")

def consultar_cerebro_w7(pergunta_usuario):
    try:
        # 1. Busca os trechos mais relevantes no banco vetorial local
        cliente_chroma = chromadb.PersistentClient(path=DIRETORIO_BANCO)
        colecao = cliente_chroma.get_collection(name="conhecimento_w7")
        
        resultados = colecao.query(
            query_texts=[pergunta_usuario],
            n_results=3
        )
        
        contexto_recuperado = "\n".join(resultados['documents'][0])
        
        # 2. Instrução de sistema restrita à apostila da W7
        system_prompt = f"""
        Você é a Inteligência Artificial Oficial da W7 Academy.
        Responda à dúvida do usuário baseando-se ESTRITAMENTE nos trechos da apostila oficial da W7 fornecidos abaixo.
        Se a informação não estiver na apostila, utilize seu conhecimento técnico de cinesiologia, mas mantenha o rigor científico.
        
        TRECHOS DA APOSTILA DA W7 (Contexto):
        {contexto_recuperado}
        """
        
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=pergunta_usuario,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
            ),
        )
        return response.text
    except Exception as e:
        return f"⚠️ Erro ao consultar a base de conhecimento: {str(e)}"

# ==========================================
# CONFIGURAÇÃO DA INTERFACE WEB (STREAMLIT)
# ==========================================
st.set_page_config(
    page_title="W7 Academy - Assistente Especialista",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 W7 Academy | Consultor Científico & Cinesiologia")
st.caption("Sistema inteligente alimentado com a base de conhecimento oficial de lesões e reabilitação.")

# Inicializa o histórico de mensagens do chat na tela
if "mensagens" not in st.session_state:
    st.session_state.mensagens = [
        {
            "role": "assistant",
            "content": "Olá! Sou o assistente oficial da W7 Academy. Pode fazer sua pergunta técnica sobre ombro, cotovelo, joelho, coluna ou reabilitação!"
        }
    ]

# Renderiza o histórico de conversas em balões
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Caixa de input fixada na base (ao dar Enter, envia sozinho)
if prompt_usuario := st.chat_input("Digite sua dúvida técnica baseada na apostila..."):
    # Adiciona a mensagem do usuário na tela
    st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
        
    # Processa a resposta usando o Cérebro RAG da W7
    with st.chat_message("assistant"):
        with st.spinner("Buscando diretrizes na apostila e acionando o Gemini 3.6 Flash..."):
            resposta_ia = consultar_cerebro_w7(prompt_usuario)
            st.markdown(resposta_ia)
            
    # Salva a resposta no histórico
    st.session_state.mensagens.append({"role": "assistant", "content": resposta_ia})