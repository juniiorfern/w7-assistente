import os
import streamlit as st
import chromadb
from google import genai
from google.genai import types
from pypdf import PdfReader

# Configuracao da pagina Streamlit
st.set_page_config(page_title="W7 Academy Assistente", page_icon="💪", layout="centered")

# ======================================================
# CONFIGURACAO DO GEMINI E BANCO DE DADOS
# ======================================================
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_BANCO = os.path.join(DIRETORIO_ATUAL, "w7_database")
CAMINHO_PDF = os.path.join(DIRETORIO_ATUAL, "apostila.pdf")

@st.cache_resource
def obter_colecao():
    """Garante que a colecao exista e esteja carregada com a apostila."""
    cliente_chroma = chromadb.PersistentClient(path=DIRETORIO_BANCO)
    colecao = cliente_chroma.get_or_create_collection(name="conhecimento_w7")
    
    # Se a colecao estiver vazia na nuvem, processa o PDF automaticamente
    if colecao.count() == 0 and os.path.exists(CAMINHO_PDF):
        leitor = PdfReader(CAMINHO_PDF)
        docs, metas, ids = [], [], []
        for i, pagina in enumerate(leitor.pages):
            texto = pagina.extract_text()
            if texto and texto.strip():
                docs.append(texto.strip())
                metas.append({"pagina": i + 1})
                ids.append(f"pag_{i + 1}")
        if docs:
            colecao.add(documents=docs, metadatas=metas, ids=ids)
            
    return colecao

def consultar_cerebro_w7(pergunta_usuario):
    try:
        colecao = obter_colecao()
        
        # 1. Busca os trechos mais relevantes
        resultado = colecao.query(query_texts=[pergunta_usuario], n_results=3)
        contexto_lista = resultado.get("documents", [[]])[0]
        contexto = "\n\n---\n\n".join(contexto_lista) if contexto_lista else "Nenhum trecho especifico encontrado."

        # 2. Prompt com contexto da apostila
        prompt_final = f"""
Voce e o assistente de inteligencia artificial oficial da W7 Academy, especializado em biomecanica, treinamento resistido, prevencao de lesoes e reabilitacao fisica.

Use estritamente as informacoes do contexto abaixo para responder a duvida do usuario com precisao, clareza e autoridade tecnica.

--- CONTEXTO DA APOSTILA W7 ---
{contexto}
-------------------------------

Pergunta do usuario: {pergunta_usuario}

Resposta:
"""
        resposta = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_final,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        return resposta.text

    except Exception as e:
        return f"⚠️ Erro ao consultar a base de conhecimento: {str(e)}"

# ======================================================
# INTERFACE DO USUARIO (CHAT)
# ======================================================
st.title("💪 W7 Academy - Assistente IA")
st.caption("Tire suas duvidas tecnicas baseadas na metodologia oficial da W7 Academy.")

if "mensagens" not in st.session_state:
    st.session_state["mensagens"] = [
        {"role": "assistant", "content": "Olá! Sou o assistente oficial da W7 Academy. Pode fazer sua pergunta técnica sobre ombro, cotovelo, joelho, coluna ou reabilitação!"}
    ]

for msg in st.session_state["mensagens"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Digite sua pergunta técnica aqui..."):
    st.session_state["mensagens"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando apostila e gerando resposta..."):
            resposta_ia = consultar_cerebro_w7(prompt)
            st.write(resposta_ia)

    st.session_state["mensagens"].append({"role": "assistant", "content": resposta_ia})
