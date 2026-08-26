import os
import time
from pathlib import Path
import streamlit as st
import chromadb
from google import genai
from google.genai import types
from pypdf import PdfReader

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="W7 Academy | Consultor da Apostila",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# 2. CONEXÃO & BANCO VETORIAL COM CHUNKING
# ==========================================
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_BANCO = DIRETORIO_ATUAL / "w7_database"
CAMINHO_PDF = DIRETORIO_ATUAL / "apostila.pdf"

def fatiar_texto(texto: str, tamanho_bloco: int = 700, sobreposicao: int = 150) -> list[str]:
    """Divide o texto em blocos menores com sobreposição para não cortar frases no meio."""
    blocos = []
    inicio = 0
    tamanho_total = len(texto)
    
    while inicio < tamanho_total:
        fim = inicio + tamanho_bloco
        bloco = texto[inicio:fim].strip()
        if bloco:
            blocos.append(bloco)
        inicio += tamanho_bloco - sobreposicao
    return blocos

@st.cache_resource
def obter_colecao():
    """Lê a apostila, fatia em parágrafos precisos e indexa no ChromaDB."""
    cliente_chroma = chromadb.PersistentClient(path=str(DIRETORIO_BANCO))
    colecao = cliente_chroma.get_or_create_collection(name="conhecimento_w7")
    
    if colecao.count() == 0 and CAMINHO_PDF.exists():
        leitor = PdfReader(str(CAMINHO_PDF))
        docs, metas, ids = [], [], []
        
        for num_pag, pagina in enumerate(leitor.pages, start=1):
            texto = pagina.extract_text() or ""
            if texto.strip():
                pedacos = fatiar_texto(texto.strip())
                for idx, pedaco in enumerate(pedacos):
                    docs.append(pedaco)
                    metas.append({"pagina": num_pag, "bloco": idx + 1})
                    ids.append(f"pag_{num_pag}_b_{idx + 1}")
                    
        if docs:
            # Insere os blocos em lotes para garantir estabilidade
            tamanho_lote = 100
            for i in range(0, len(docs), tamanho_lote):
                colecao.add(
                    documents=docs[i:i + tamanho_lote],
                    metadatas=metas[i:i + tamanho_lote],
                    ids=ids[i:i + tamanho_lote]
                )
            
    return colecao

# ==========================================
# 3. EXTRAÇÃO ESTRITA DA APOSTILA
# ==========================================
@st.cache_data(show_spinner=False, ttl=3600)
def executar_consulta_ia(pergunta_usuario: str, contexto_recuperado: str) -> str:
    system_prompt = f"""
Você é o Consultor Científico Oficial da W7 Academy.

DIRETRIZES DE RESPOSTA:
1. Baseie toda a sua análise e raciocínio técnico estritamente nos conceitos de cinesiologia, biomecânica e condutas presentes no CONTEXTO abaixo.
2. Caso o usuário mencione uma dor, lesão ou exercício com termos específicos (ex: epicondilite) que não apareçam literalmente no texto, utilize os princípios anatômicos da articulação correspondente (ex: cotovelo, punho, ombro) descritos no CONTEXTO para orientar os ajustes e cuidados.
3. Se o assunto for completamente alheio ao material da W7 e não houver base mecânica no contexto, informe de maneira profissional que o tópico não está contemplado na apostila.
4. Mantenha tom técnico, direto e didático.

--- CONTEXTO DA APOSTILA W7 ---
{contexto_recuperado}
---------------------------------------
"""
    tentativas = 3
    intervalo = 4

    for tentativa in range(tentativas):
        try:
            resposta = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=pergunta_usuario,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                ),
            )
            return resposta.text

        except Exception as erro:
            erro_msg = str(erro)
            if ("429" in erro_msg or "RESOURCE_EXHAUSTED" in erro_msg) and tentativa < tentativas - 1:
                time.sleep(intervalo)
                intervalo *= 2
                continue
            
            if "429" in erro_msg or "RESOURCE_EXHAUSTED" in erro_msg:
                return "⏳ O servidor atingiu a cota momentânea de requisições. Aguarde 30 segundos e envie sua dúvida novamente."
            
            return f"⚠️ Instabilidade temporária. Detalhes: {erro_msg[:80]}"

def stream_texto(texto: str):
    """Gera o efeito de digitação suave palavra por palavra."""
    for palavra in texto.split(" "):
        yield palavra + " "
        time.sleep(0.02)

def consultar_cerebro_w7(pergunta_usuario: str) -> str:
    try:
        colecao = obter_colecao()
        
        # Puxa os 6 blocos mais precisos da apostila
        resultados = colecao.query(
            query_texts=[pergunta_usuario],
            n_results=8
        )
        
        documentos = resultados.get('documents', [[]])[0]
        contexto_recuperado = "\n\n---\n\n".join(documentos) if documentos else "Nenhum trecho correspondente localizado."
        
        return executar_consulta_ia(pergunta_usuario, contexto_recuperado)

    except Exception as e:
        return f"⚠️ Erro ao consultar a base de dados: {str(e)}"

# ==========================================
# 4. INTERFACE DO USUÁRIO
# ==========================================
st.title("🧠 W7 Academy | Consultor Científico")
st.caption("Assistente especialista baseado no conteúdo oficial da apostila.")

if "mensagens" not in st.session_state or not st.session_state.mensagens:
    st.session_state.mensagens = [
        {
            "role": "assistant",
            "content": "W7 Academy",
            "fontes": []
        }
    ]
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_usuario := st.chat_input("Digite sua dúvida sobre a apostila..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
        
with st.chat_message("assistant"):
            with st.spinner("Localizando trechos na apostila oficial da W7..."):
                resposta_ia = consultar_cerebro_w7(prompt_usuario)
            resposta_ia = st.write_stream(stream_texto(resposta_ia))

st.session_state.mensagens.append({"role": "assistant", "content": resposta_ia})
