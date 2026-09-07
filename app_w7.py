import re
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
    page_title="W7 Academy | Jimmy",
    page_icon="🟢",
    layout="wide"
)

# ==========================================
# 2. CONEXÃO & CONFIGURAÇÃO
# ==========================================
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

MODELO = "gemini-3.6-flash"

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_BANCO = DIRETORIO_ATUAL / "w7_database"

FONTES = [
    {"arquivo": DIRETORIO_ATUAL / "apostila.pdf", "nome_fonte": "apostila_lesoes_w7"},
]

PADRAO_CAPITULO = re.compile(r"CAP[ÍI]TULO\s+\d+\s*[—\-]\s*(.+)", re.IGNORECASE)
MARCADORES_BIBLIOGRAFIA = ("referencias bibliográficas", "referências bibliográficas")


# ==========================================
# 3. INDEXAÇÃO COM METADADOS
# ==========================================
def fatiar_texto(texto: str, tamanho_bloco: int = 700, sobreposicao: int = 150) -> list[str]:
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
    cliente_chroma = chromadb.PersistentClient(path=str(DIRETORIO_BANCO))
    colecao = cliente_chroma.get_or_create_collection(name="conhecimento_w7")

    if colecao.count() > 0:
        return colecao

    for fonte in FONTES:
        caminho_pdf = fonte["arquivo"]
        nome_fonte = fonte["nome_fonte"]

        if not caminho_pdf.exists():
            continue

        leitor = PdfReader(str(caminho_pdf))
        docs, metas, ids = [], [], []
        capitulo_atual = "geral"
        em_bibliografia = False

        for num_pag, pagina in enumerate(leitor.pages, start=1):
            texto = pagina.extract_text() or ""
            texto_lower = texto.lower()

            if any(marcador in texto_lower for marcador in MARCADORES_BIBLIOGRAFIA):
                em_bibliografia = True
            if em_bibliografia:
                continue

            match_capitulo = PADRAO_CAPITULO.search(texto)
            if match_capitulo:
                capitulo_atual = match_capitulo.group(1).strip().split("\n")[0]

            if texto.strip():
                pedacos = fatiar_texto(texto.strip())
                for idx, pedaco in enumerate(pedacos):
                    docs.append(pedaco)
                    metas.append({
                        "fonte": nome_fonte,
                        "capitulo": capitulo_atual,
                        "pagina": num_pag,
                        "bloco": idx + 1,
                    })
                    ids.append(f"{nome_fonte}_pag_{num_pag}_b_{idx + 1}")

        if docs:
            tamanho_lote = 100
            for i in range(0, len(docs), tamanho_lote):
                colecao.add(
                    documents=docs[i:i + tamanho_lote],
                    metadatas=metas[i:i + tamanho_lote],
                    ids=ids[i:i + tamanho_lote],
                )

    return colecao


# ==========================================
# 4. CONSULTA À IA (com retry e tom fluido)
# ==========================================
def executar_consulta_ia(prompt_completo: str) -> str:
    tentativas = 3
    intervalo = 4

    for tentativa in range(tentativas):
        try:
            resposta = client.models.generate_content(
                model=MODELO,
                contents=prompt_completo,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            return resposta.text

        except Exception as erro:
            erro_msg = str(erro)
            erro_temporario = any(c in erro_msg for c in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))

            if erro_temporario and tentativa < tentativas - 1:
                time.sleep(intervalo)
                intervalo *= 2
                continue

            if erro_temporario:
                return "O servidor deu uma oscilada rápida. Manda a mensagem de novo em alguns segundos."

            return f"⚠️ Instabilidade temporária. Detalhes: {erro_msg[:80]}"


def stream_texto(texto: str):
    for palavra in texto.split(" "):
        yield palavra + " "
        time.sleep(0.02)


def consultar_cerebro_w7(pergunta_usuario: str) -> str:
    try:
        texto_busca = str(pergunta_usuario).strip()
        if not texto_busca:
            return "Pode mandar sua dúvida."

        colecao = obter_colecao()
        resultados = colecao.query(query_texts=[texto_busca], n_results=6)

        docs_brutos = resultados.get("documents", [[]])[0] if resultados.get("documents") else []
        documentos = [d for d in docs_brutos if isinstance(d, str) and d.strip()]
        contexto_recuperado = (
            "\n\n---\n\n".join(documentos) if documentos else "Nenhum trecho correspondente encontrado."
        )

        prompt_completo = (
            "Você é o Jimmy. Apenas o Jimmy, parceiro do pessoal aqui na W7 Academy 🟢.\n\n"
            "COMO VOCÊ CONVERSA:\n"
            "- Seja você mesmo: natural, direto, fluido, inteligente e conversacional. Nada de falar como um robô, atendente ou enciclopédia engessada.\n"
            "- Não fique arrotando termos como 'sou especialista em cinesiologia e biomecânica'. Você tem acesso à apostila de lesões e condutas da W7.\n"
            "- Se a pessoa mandou um 'tudo bem?', 'boa tarde' ou começou uma conversa, responda com simpatia e naturalidade antes de ir ao ponto.\n"
            "- Não tente adivinhar o nome de ninguém pegando palavras da frase. Só chame pelo nome se a pessoa tiver se apresentado claramente.\n"
            "- Vá direto ao que interessa: responda com clareza usando o material abaixo como referência, sem jogar paredes de texto desnecessárias.\n"
            "- Se o material não falar nada sobre o assunto, fale na boa que a apostila de lesões não cobre esse ponto.\n\n"
            f"--- MATERIAL DE CONSULTA (APOSTILA) ---\n{contexto_recuperado}\n---------------------------------------\n\n"
            f"Mensagem da pessoa: {texto_busca}"
        )

        return executar_consulta_ia(prompt_completo)
    except Exception as erro:
        return f"Erro ao consultar o material: {str(erro)}"


# ==========================================
# 5. INTERFACE DO USUÁRIO
# ==========================================
ICONE_ASSISTENTE = "🟢"
ICONE_USUARIO = "👤"

st.markdown(
    """
    <style>
    [data-testid="stChatMessageAvatarCustom"] {
        animation: pulsar-cerebro 2s infinite ease-in-out;
    }
    @keyframes pulsar-cerebro {
        0% { transform: scale(1); }
        50% { transform: scale(1.15); }
        100% { transform: scale(1); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🟢 Jimmy | W7 Academy")
st.caption("Tire dúvidas e consulte as condutas com base no material da W7.")

if "mensagens" not in st.session_state or not st.session_state.mensagens:
    st.session_state.mensagens = [
        {
            "role": "assistant",
            "content": (
                "Fala! Eu sou o **Jimmy** 🟢💪\n\n"
                "Tô por aqui pra ajudar com as dúvidas sobre lesões e condutas do material da W7. Como posso te ajudar hoje?"
            ),
        }
    ]

for msg in st.session_state.mensagens:
    icone = ICONE_ASSISTENTE if msg["role"] == "assistant" else ICONE_USUARIO
    with st.chat_message(msg["role"], avatar=icone):
        st.markdown(msg["content"])

if prompt_usuario := st.chat_input("Fala comigo..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user", avatar=ICONE_USUARIO):
        st.markdown(prompt_usuario)

    with st.chat_message("assistant", avatar=ICONE_ASSISTENTE):
        with st.spinner("Jimmy buscando no material... 🟢"):
            resposta_ia = consultar_cerebro_w7(prompt_usuario)
        resposta_ia = st.write_stream(stream_texto(resposta_ia))

    st.session_state.mensagens.append({"role": "assistant", "content": resposta_ia})
