import re
import requests
import threading
import time
from pathlib import Path

import streamlit as st
import chromadb
from google import genai
from google.genai import types
from pypdf import PdfReader

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA (W7 ULTRA-DARK)
# ==========================================
st.set_page_config(
    page_title="W7 | Jimmy AI",
    page_icon="logo_w7.png",
    layout="centered",
    initial_sidebar_state="collapsed"
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

            return f"Erro real da IA: {erro_msg}"


def stream_texto(texto: str):
    for palavra in texto.split(" "):
        yield palavra + " "
        time.sleep(0.02)


def consultar_cerebro_w7(pergunta_usuario: str, historico: list = None) -> str:
    try:
        texto_busca = str(pergunta_usuario).strip()
        if not texto_busca:
            return "Pode mandar sua dúvida."

        # Formata as últimas mensagens para contextualizar a conversa
        contexto_conversa = ""
        if historico:
            linhas_historico = []
            for msg in historico:
                papel = "Instrutor" if msg["role"] == "user" else "Jimmy"
                linhas_historico.append(f"{papel}: {msg['content']}")
            contexto_conversa = "\n".join(linhas_historico)

        colecao = obter_colecao()
        resultados = colecao.query(query_texts=[texto_busca], n_results=6)

        docs_brutos = resultados.get("documents", [[]])[0] if resultados.get("documents") else []
        documentos = [d for d in docs_brutos if isinstance(d, str) and d.strip()]
        contexto_recuperado = (
            "\n\n---\n\n".join(documentos) if documentos else "Nenhum trecho correspondente encontrado."
        )

        prompt_completo = (
            "Você é o Jimmy. Apenas o Jimmy, parceiro da equipe aqui na W7 Academy 🔴.\n\n"
            "COMO VOCÊ CONVERSA:\n"
            "- Seja natural, direto, fluido e parceiro. Sem jargões forçados ou postura engessada.\n"
            "- Seu foco prático é a apostila de lesões e condutas da W7.\n"
            "- Se a pessoa apenas cumprimentou, responda com simpatia e naturalidade.\n"
            "- Mantenha a coerência com as mensagens anteriores da conversa sem precisar que repitam o assunto.\n"
            "- Entregue a conduta e a orientação com clareza, usando o material de referência abaixo.\n"
            "- Se o material não contemplar a situação, diga com sinceridade que a apostila não detalha esse caso.\n\n"
            f"--- HISTÓRICO RECENTE DO DIÁLOGO ---\n{contexto_conversa if contexto_conversa else 'Início de conversa.'}\n---------------------------------------\n\n"
            f"--- MATERIAL DE CONSULTA (APOSTILA) ---\n{contexto_recuperado}\n---------------------------------------\n\n"
            f"Última mensagem da pessoa: {texto_busca}"
        )

        return executar_consulta_ia(prompt_completo)
    except Exception as erro:
        return f"Erro ao consultar o material: {str(erro)}"

# ==========================================
# INTEGRAÇÃO PLANILHA GOOGLE SHEETS
# ==========================================
URL_PLANILHA_W7 = "https://script.google.com/macros/s/AKfycbw7CApe9ml2evzGAvPzWOao_3ztGxxkUkiE1KXBlQ1yGt6kb-HXGmMLW05b8ic3uo7GUw/exec"

def gravar_dialogo_planilha(pergunta: str, resposta: str):
    def enviar():
        try:
            if URL_PLANILHA_W7:
                requests.post(
                    URL_PLANILHA_W7,
                    json={"pergunta": pergunta, "resposta": resposta},
                    timeout=5
                )
        except Exception:
            pass

    threading.Thread(target=enviar).start()
# ==========================================
# 5. INTERFACE DO USUÁRIO (Design de Apresentação Executiva)
# ==========================================
ICONE_ASSISTENTE = "logo_w7.png"
ICONE_USUARIO = "🏋️"

# CSS Avançado - W7 Cyber-Athletic Theme
st.markdown(
    """
    <style>
    /* Ocultar elementos padrão do Streamlit */
    #MainMenu, header, footer {visibility: hidden;}
    
    /* 1. Atmosfera e Fundo Geral */
    .stApp {
        background: radial-gradient(circle at 50% 0%, rgba(237, 28, 36, 0.18) 0%, rgba(10, 11, 14, 1) 65%), #0A0B0E !important;
        color: #F8F9FA !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        max-width: 820px !important;
    }

    /* 2. Painel Superior HUD (Glassmorphism) */
    .w7-hud-card {
        background: rgba(18, 20, 26, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(237, 28, 36, 0.25);
        border-radius: 20px;
        padding: 16px 22px;
        display: flex;
        align-items: center;
        gap: 18px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.8), 0 0 20px rgba(237, 28, 36, 0.15);
        margin-bottom: 24px;
    }

    .hud-title {
        font-size: 24px;
        font-weight: 900;
        letter-spacing: 1.5px;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF 0%, #E0E0E0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }

    .hud-subtitle {
        font-size: 12px;
        color: #8E95A5;
        letter-spacing: 0.5px;
        margin: 4px 0 0 0;
        font-weight: 500;
    }

    .hud-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(237, 28, 36, 0.12);
        border: 1px solid rgba(237, 28, 36, 0.4);
        padding: 3px 10px;
        border-radius: 30px;
        font-size: 10px;
        font-weight: 700;
        color: #FF3B44;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* Radar de status piscando */
    .status-ping {
        width: 7px;
        height: 7px;
        background-color: #FF1E27;
        border-radius: 50%;
        box-shadow: 0 0 8px #FF1E27;
        animation: radar-ping 1.6s infinite ease-in-out;
    }

    @keyframes radar-ping {
        0% { transform: scale(0.9); opacity: 0.7; box-shadow: 0 0 2px #FF1E27; }
        50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 12px #FF1E27; }
        100% { transform: scale(0.9); opacity: 0.7; box-shadow: 0 0 2px #FF1E27; }
    }

    /* 3. Balões de Chat de Alta Performance */
    [data-testid="stChatMessage"] {
        background: rgba(18, 20, 26, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 18px 22px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35) !important;
    }

    /* Balão do Jimmy: Aura de autoridade e acabamento rubro */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) {
        background: linear-gradient(135deg, rgba(28, 14, 17, 0.45) 0%, rgba(18, 20, 26, 0.8) 100%) !important;
        border-left: 4px solid #ED1C24 !important;
        border-top: 1px solid rgba(237, 28, 36, 0.3) !important;
        box-shadow: 0 10px 25px -5px rgba(237, 28, 36, 0.15) !important;
    }

    /* Avatar do Jimmy pulsando em neon */
    [data-testid="stChatMessageAvatarCustom"] img {
        animation: breathing-glow 2.4s infinite ease-in-out;
        object-fit: contain;
    }

    @keyframes breathing-glow {
        0% {
            transform: scale(1);
            filter: drop-shadow(0 0 3px rgba(237, 28, 36, 0.4));
        }
        50% {
            transform: scale(1.1);
            filter: drop-shadow(0 0 14px rgba(237, 28, 36, 0.95));
        }
        100% {
            transform: scale(1);
            filter: drop-shadow(0 0 3px rgba(237, 28, 36, 0.4));
        }
    }

    /* 4. Barra de Envio estilo Titanio & Fogo */
    .stChatInputContainer {
        border-radius: 16px !important;
        background: rgba(20, 23, 30, 0.9) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6) !important;
    }

    .stChatInputContainer:focus-within {
        border-color: #ED1C24 !important;
        box-shadow: 0 0 18px rgba(237, 28, 36, 0.55) !important;
    }
    
    /* Personalização de botões de atalho */
    div[data-testid="stHorizontalBlock"] button {
        background: rgba(25, 28, 36, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #C0C5D0 !important;
        border-radius: 10px !important;
        font-size: 12px !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        border-color: #ED1C24 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 10px rgba(237, 28, 36, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Renderização do HUD Superior W7
st.markdown(
    """
    <div class="w7-hud-card">
        <img src="https://raw.githubusercontent.com/juniiorfern/w7-assistente/main/logo_w7.png" width="65" style="filter: drop-shadow(0 0 8px rgba(237,28,36,0.5));">
        <div style="flex-grow: 1;">
            <div class="hud-status-badge">
                <div class="status-ping"></div>
                Base Clínica W7 • Conectada
            </div>
           <h1 class="hud-title">JIMMY</h1>
            <p class="hud-subtitle">Seu suporte diário para condutas com alunos que têm limitações ou dores</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Inicialização da Sessão
if "mensagens" not in st.session_state or not st.session_state.mensagens:
    st.session_state.mensagens = [
        {
            "role": "assistant",
            "content": (
                "Fala, time! **Jimmy** no posto. 🔴💪\n\n"
                "A base de lesões e adaptações da **W7 Academy** está carregada e pronta. "
                "Qual aluno ou situação conversar agora?"
            ),
        }
    ]

# Renderização das Mensagens
for msg in st.session_state.mensagens:
    icone = ICONE_ASSISTENTE if msg["role"] == "assistant" else ICONE_USUARIO
    with st.chat_message(msg["role"], avatar=icone):
        st.markdown(msg["content"])

# Atalhos rápidos de perguntas (Ideal para a apresentação ao vivo!)
st.markdown("<p style='font-size:11px; color:#6C7380; text-transform:uppercase; font-weight:700; letter-spacing:1px; margin: 15px 0 6px 2px;'>Simulações Rápidas de Salão:</p>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

pergunta_rapida = None
if col1.button("💥 Dor Patelar no Agachamento"):
    pergunta_rapida = "Aluno com condromalácia patelar sente dor no agachamento. Qual a conduta da apostila?"
if col2.button("⚡ Hérnia Discal & Extensora"):
    pergunta_rapida = "Aluno com hérnia de disco lombar pode treinar perna pesado? Quais cuidados?"
if col3.button("🎯 Pós-Cirúrgico de LCA"):
    pergunta_rapida = "Aluno operou o LCA há 4 meses, o que o material W7 recomenda para fortalecer?"

# Gatilho de Envio (Por digitação ou clique rápido)
prompt_final = st.chat_input("Consulte a conduta técnica...") or pergunta_rapida

if prompt_final:
    historico_recente = st.session_state.mensagens[-4:] if len(st.session_state.mensagens) > 0 else []

    st.session_state.mensagens.append({"role": "user", "content": prompt_final})
    with st.chat_message("user", avatar=ICONE_USUARIO):
        st.markdown(prompt_final)

    with st.chat_message("assistant", avatar=ICONE_ASSISTENTE):
        with st.spinner("Jimmy está pensando... 🔴"):
            resposta_ia = consultar_cerebro_w7(prompt_final, historico=historico_recente)
        resposta_ia = st.write_stream(stream_texto(resposta_ia))

    st.session_state.mensagens.append({"role": "assistant", "content": resposta_ia})
    gravar_dialogo_planilha(prompt_final, resposta_ia)
