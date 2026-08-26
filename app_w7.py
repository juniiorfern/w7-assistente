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

def consultar_cerebro_w7(pergunta_usuario: str, nome_usuario: str = "Aluno") -> str:
    try:
        texto_busca = str(pergunta_usuario).strip()
        if not texto_busca:
            return "Por favor, digite uma dúvida válida."

        colecao = obter_colecao()

        resultados = colecao.query(
            query_texts=[texto_busca],
            n_results=6
        )

        docs_brutos = resultados.get('documents', [[]])[0] if resultados.get('documents') else []
        documentos = [d for d in docs_brutos if isinstance(d, str) and d.strip()]

        contexto_recuperado = "\n\n---\n\n".join(documentos) if documentos else "Nenhum trecho correspondente encontrado na apostila."

        # Prompt com a personalidade parceira do Jimmy
        pergunta_com_instrucao = (
            f"Você é o Jimmy, parceiro de estudos amigável e especialista da W7 Academy. "
            f"Você está respondendo para {nome_usuario}. "
            f"Se a pergunta for técnica, complexa ou exigir análise profunda, faça um breve comentário amigável e empolgado no início chamando a pessoa pelo nome (ex: 'Boa pergunta, {nome_usuario}!', 'Essa pegadinha é ótima, {nome_usuario}...'). "
            f"Em seguida, entregue a resposta técnica, clara e direta baseada na apostila.\n\n"
            f"Dúvida de {nome_usuario}: {texto_busca}"
        )

        return executar_consulta_ia(pergunta_com_instrucao, contexto_recuperado)
    except Exception as erro:
        return f"Erro ao consultar a base de dados: {str(erro)}"

# ==========================================
# 4. INTERFACE DO USUÁRIO
# ==========================================

ICONE_ASSISTENTE = "🧠"
ICONE_USUARIO = "👤"

st.title("🧠 Jimmy | W7 Academy")
st.caption("Seu parceiro de estudos em biomecânica e cinesiologia.")

# Controle de estado do nome e histórico
if "nome_usuario" not in st.session_state:
    st.session_state.nome_usuario = None

if "mensagens" not in st.session_state or not st.session_state.mensagens:
    st.session_state.mensagens = [
        {
            "role": "assistant",
            "content": "Fala! Eu sou o **Jimmy**, seu parceiro de estudos aqui na **W7 Academy** 🧠💪\n\nAntes de começarmos, qual é o seu nome?"
        }
    ]

# Exibição do histórico de mensagens
for msg in st.session_state.mensagens:
    icone = ICONE_ASSISTENTE if msg["role"] == "assistant" else ICONE_USUARIO
    with st.chat_message(msg["role"], avatar=icone):
        st.markdown(msg["content"])

# Processamento das entradas do usuário
if prompt_usuario := st.chat_input("Digite sua resposta ou dúvida aqui..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user", avatar=ICONE_USUARIO):
        st.markdown(prompt_usuario)

    # 1ª Interação: Capturar e gravar o nome da pessoa
    if not st.session_state.nome_usuario:
        # Extrai apenas o nome de forma limpa (ex: "Sou a Andreia" -> "Andreia")
        nome_limpo = prompt_usuario.replace("Meu nome é", "").replace("meu nome é", "").replace("Sou o", "").replace("Sou a", "").strip()
        nome_formatado = nome_limpo.split()[0].title() if nome_limpo else "Aluno"
        st.session_state.nome_usuario = nome_formatado

        boas_vindas = f"Prazer, **{nome_formatado}**! 🚀 Agora sim.\n\nQual dúvida sobre cinesiologia, biomecânica ou exercícios vamos desvendar hoje?"
        with st.chat_message("assistant", avatar=ICONE_ASSISTENTE):
            st.markdown(boas_vindas)
        st.session_state.mensagens.append({"role": "assistant", "content": boas_vindas})

    # Interações seguintes: Consultas técnicas da apostila
    else:
        with st.chat_message("assistant", avatar=ICONE_ASSISTENTE):
            with st.spinner(f"Jimmy consultando a apostila para {st.session_state.nome_usuario}..."):
                resposta_ia = consultar_cerebro_w7(prompt_usuario, st.session_state.nome_usuario)
            resposta_ia = st.write_stream(stream_texto(resposta_ia))

        st.session_state.mensagens.append({"role": "assistant", "content": resposta_ia})
