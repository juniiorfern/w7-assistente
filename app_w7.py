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

MODELO = "gemini-2.5-flash"

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_BANCO = DIRETORIO_ATUAL / "w7_database"

FONTES = [
    {"arquivo": DIRETORIO_ATUAL / "apostila.pdf", "nome_fonte": "apostila_w7"},
    # {"arquivo": DIRETORIO_ATUAL / "ebook_biomecanica.pdf", "nome_fonte": "ebook_bruno_leitao"},
]

PADRAO_CAPITULO = re.compile(r"CAP[ÍI]TULO\s+\d+\s*[—\-]\s*(.+)", re.IGNORECASE)
MARCADORES_BIBLIOGRAFIA = ("referencias bibliográficas", "referências bibliográficas")


# ==========================================
# 3. IDENTIFICAÇÃO DO CRIADOR (JÚNIOR)
# ==========================================
# 1. Reconhece via link direto: seu-link.streamlit.app/?admin=junior
param_admin = st.query_params.get("admin") == "junior"

if "modo_criador" not in st.session_state:
    st.session_state.modo_criador = param_admin

# Barra lateral discreta para controle de modo
with st.sidebar:
    st.markdown("### ⚙️ Painel de Acesso")
    if st.session_state.modo_criador:
        st.success("🟢 Modo Criador Ativo (Júnior)")
        if st.button("Alternar para Modo Aluno"):
            st.session_state.modo_criador = False
            st.session_state.mensagens = []
            st.rerun()
    else:
        st.info("Modo Padrão (Aluno / Professor)")
        senha_acesso = st.text_input("Acesso Criador", type="password", placeholder="Chave")
        if senha_acesso == "junior77":  # Você pode trocar sua senha aqui
            st.session_state.modo_criador = True
            st.session_state.mensagens = []
            st.rerun()


# ==========================================
# 4. INDEXAÇÃO COM METADADOS
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
# 5. CONSULTA À IA (com personalização de papel)
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
                return "⏳ O servidor do Gemini está instável no momento. Aguarde alguns segundos e envie novamente."

            return f"⚠️ Instabilidade temporária. Detalhes: {erro_msg[:80]}"


def stream_texto(texto: str):
    for palavra in texto.split(" "):
        yield palavra + " "
        time.sleep(0.02)


def consultar_cerebro_w7(pergunta_usuario: str, eh_criador: bool = False) -> str:
    try:
        texto_busca = str(pergunta_usuario).strip()
        if not texto_busca:
            return "Por favor, digite uma dúvida válida."

        colecao = obter_colecao()
        resultados = colecao.query(query_texts=[texto_busca], n_results=6)

        docs_brutos = resultados.get("documents", [[]])[0] if resultados.get("documents") else []
        documentos = [d for d in docs_brutos if isinstance(d, str) and d.strip()]
        contexto_recuperado = (
            "\n\n---\n\n".join(documentos) if documentos else "Nenhum trecho correspondente encontrado."
        )

        # Instrução bifurcada conforme o interlocutor
        if eh_criador:
            prompt_completo = (
                "Você é o Jimmy, assistente técnico de inteligência da W7 Academy 🧠.\n"
                "Você está dialogando DIRETAMENTE com seu criador e coordenador técnico: Júnior.\n\n"
                "DIRETRIZES DE COMUNICAÇÃO COM O CRIADOR (JÚNIOR):\n"
                "1. Trate-o pelo nome (Júnior) de forma próxima, como um copiloto e consultor sênior de biomecânica.\n"
                "2. Vá direto ao ponto. Não use didática básica ou respostas mastigadas para iniciantes.\n"
                "3. Entregue raciocínio clínico refinado, aprofunde em vetores, torques articulares, mecânica muscular e estratégias de intervenção.\n"
                "4. Seja sincero, crítico e traga as nuances técnicas mais avançadas baseadas no contexto e nas evidências.\n\n"
                f"--- CONTEXTO TÉCNICO ---\n{contexto_recuperado}\n--------------------------\n\n"
                f"Mensagem do Júnior: {texto_busca}"
            )
        else:
            prompt_completo = (
                "Você é o Jimmy, consultor especialista e parceiro de estudos da W7 Academy.\n\n"
                "DIRETRIZES DE COMUNICAÇÃO:\n"
                "1. REGRA DO NOME: nunca tente adivinhar o nome de quem pergunta. Nunca use saudações como nomes.\n"
                "2. DIDÁTICA EM CAMADAS: conciso e direto ao ponto. 1 a 2 parágrafos curtos ou tópicos diretos baseados na apostila.\n"
                "3. GANCHO FINAL: conclua sempre com uma pergunta curta oferecendo o próximo passo.\n"
                "4. Se o assunto não constar no material, informe educadamente que não está contemplado.\n\n"
                f"--- CONTEXTO DA APOSTILA ---\n{contexto_recuperado}\n---------------------------------------\n\n"
                f"Dúvida do usuário: {texto_busca}"
            )

        return executar_consulta_ia(prompt_completo)
    except Exception as erro:
        return f"Erro ao consultar a base de dados: {str(erro)}"


# ==========================================
# 6. INTERFACE DO USUÁRIO
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

if st.session_state.modo_criador:
    st.title("🟢 Jimmy | Central de Comando (Modo Criador)")
    st.caption("Conectado com Júnior | Engenharia Biomecânica & IA W7 Academy")
else:
    st.title("🟢 Jimmy | W7 Academy")
    st.caption("Seu parceiro de estudos em biomecânica, cinesiologia e treinamento.")

# Mensagem inicial adaptada ao usuário
if "mensagens" not in st.session_state or not st.session_state.mensagens:
    if st.session_state.modo_criador:
        texto_abertura = (
            "Fala, **Júnior**! Central pronta por aqui 🟢🧠\n\n"
            "Qual análise biomecânica, caso de aluno ou calibração de exercício vamos rodar agora?"
        )
    else:
        texto_abertura = (
            "Fala! Eu sou o **Jimmy**, seu parceiro de estudos aqui na **W7 Academy** 🟢💪\n\n"
            "Manda sua dúvida técnica sobre cinesiologia, biomecânica ou exercícios."
        )

    st.session_state.mensagens = [{"role": "assistant", "content": texto_abertura}]

for msg in st.session_state.mensagens:
    icone = ICONE_ASSISTENTE if msg["role"] == "assistant" else ICONE_USUARIO
    with st.chat_message(msg["role"], avatar=icone):
        st.markdown(msg["content"])

if prompt_usuario := st.chat_input("Manda sua dúvida pro Jimmy..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user", avatar=ICONE_USUARIO):
        st.markdown(prompt_usuario)

    with st.chat_message("assistant", avatar=ICONE_ASSISTENTE):
        spinner_texto = "Jimmy processando dados técnicos para Júnior..." if st.session_state.modo_criador else "Jimmy consultando a apostila... 🟢"
        with st.spinner(spinner_texto):
            resposta_ia = consultar_cerebro_w7(prompt_usuario, eh_criador=st.session_state.modo_criador)
        resposta_ia = st.write_stream(stream_texto(resposta_ia))

    st.session_state.mensagens.append({"role": "assistant", "content": resposta_ia})
