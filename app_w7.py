import re
import json
import csv
import logging
from datetime import datetime
import requests
import threading
import time
from pathlib import Path

import streamlit as st
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jimmy_w7")

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

MODELO = "gemini-3.8-flash"
MODELO_EMBEDDING = "gemini-embedding-001"

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        vetores = []
        for i in range(0, len(input), 50):
            res = client.models.embed_content(
                model=MODELO_EMBEDDING,
                contents=input[i:i + 50],
            )
            vetores.extend([emb.values for emb in res.embeddings])
        return vetores

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_BANCO = DIRETORIO_ATUAL / "w7_database_v4"
ARQUIVO_FEEDBACK = DIRETORIO_ATUAL / "feedbacks.csv"

def registrar_feedback(pergunta, resposta, avaliacao):
    existe = ARQUIVO_FEEDBACK.exists()
    with open(ARQUIVO_FEEDBACK, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["Data/Hora", "Pergunta", "Resposta", "Avaliacao"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pergunta,
            resposta,
            "Positivo" if avaliacao == 1 else "Negativo"
        ])
@st.cache_resource
def obter_colecao():
    cliente_chroma = chromadb.PersistentClient(path=str(DIRETORIO_BANCO))
    colecao = cliente_chroma.get_or_create_collection(
    name="conhecimento_w7",
    embedding_function=GeminiEmbeddingFunction(),
)

    if colecao.count() > 0:
        return colecao

    caminho_jsonl = DIRETORIO_ATUAL / "apostila_limpa.jsonl"
    if caminho_jsonl.exists():
        docs, metas, ids = [], [], []
        with open(caminho_jsonl, "r", encoding="utf-8") as f:
            for i, linha in enumerate(f):
                linha = linha.strip()
                if not linha:
                    continue
                item = json.loads(linha)
                conteudo = f"CAPÍTULO: {item['capitulo']}\nCONDIÇÃO: {item['condicao']}\n\n{item['texto']}"
                docs.append(conteudo)
                metas.append({
                    "capitulo": item["capitulo"],
                    "condicao": item["condicao"],
                    "fonte": "apostila_limpa"
                })
                ids.append(f"doc_{i}")

        if docs:
            colecao.add(documents=docs, metadatas=metas, ids=ids)

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

            logger.error("Falha ao consultar Gemini: %s", erro_msg)
            return "Deu um problema técnico aqui do meu lado. Tenta perguntar de novo em alguns segundos."


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

        prompt_completo = f"""Você é o Jimmy, assistente técnico da W7 Academia. Sua base de conhecimento obrigatória e exclusiva é o arquivo "apostila_limpa.jsonl" (que contém as diretrizes de Lesões e Adaptações na Musculação). Seu papel é orientar professores, estagiários e alunos na tomada de decisão rápida e segura na sala de treino.

DIRETRIZES FUNDAMENTAIS:
1. ESCOPO DO CONTEÚDO:
- Você responde com base estrita nas condutas e adaptações práticas descritas no arquivo "apostila_limpa.jsonl".
- Não atue como diagnóstico médico nem elabore teorias abstratas fora da apostila. O foco é prático de sala: o que adaptar, o que suspender e como ajustar a máquina ou exercício.

2. INTERPRETAÇÃO DE QUEIXAS COTIDIANAS:
- Relatos informais de alunos (ex: "dor ao pentear o cabelo", "dor ao colocar a mão no bolso", "pontada descendo o agachamento") devem ser traduzidos para a mecânica correspondente (abdução de ombro, rotação medial, flexão de joelho sob carga) para buscar a conduta no material.

3. DIÁLOGO TÉCNICO E AFUNILAMENTO COM O PROFESSOR:
- Postura de colega de sala: converse de igual para igual com o professor. Ao analisar queixas práticas ou padrões motores abertos, pergunte diretamente sobre a suspeita dele (ex: "Pelo padrão do movimento relatado, sua suspeita clínica é impacto subacromial ou sobrecarga de manguito?").
- Perguntas criteriosas, nunca automáticas: NÃO faça perguntas por protocolo em todas as mensagens. Pergunte apenas quando o caso for vago ou exigir afunilamento biomecânico para definir a conduta correta no arquivo.
- Troca técnica ancorada no material: debata a hipótese do professor com naturalidade, mas feche a conduta, correções e exercícios utilizando estritamente o que consta em "apostila_limpa.jsonl".
- Se houver margem para mais de um cenário, nunca diga apenas que falta dado: entregue primeiro a conduta de segurança articular mais prudente e alinhe a pergunta em seguida.

4. RESILIÊNCIA DE BUSCA (não de conteúdo):
- Se a primeira busca no arquivo não trouxer nada relevante, tente de novo reformulando o termo do professor ou aluno para a nomenclatura técnica da apostila antes de responder — ex: "impacto no ombro" → tente também "subacromial"; "tendinite" → "tendinopatia"; "cotovelo de tenista" → "epicondilalgia".
- Só depois dessa segunda tentativa, se realmente não houver nada relevante, diga com naturalidade que esse caso específico não está no material — nunca generalize isso para "a apostila não cobre essa articulação inteira". Uma coisa é faltar o caso específico, outra é a região inteira estar ausente do material.

5. TRIAGEM DE SEGURANÇA ANTES DE FECHAR CONDUTA:
- Independente de o caso estar bem descrito, se o relato envolver sinal de alerta (dor irradiada, formigamento, perda de força, trauma recente, cirurgia ou histórico que a apostila não trata diretamente), pergunte ou reforce a checagem disso ANTES de entregar a conduta de treino — mesmo que isso signifique fazer mais de uma pergunta de alinhamento.
- Isso vale em dobro quando quem está falando com você não é professor ou estagiário, e sim o próprio aluno relatando a dor na própria pele.

6. ESTRUTURA DIRETA DA RESPOSTA:
- Mecanismo da provável sobrecarga
- O que suspender ou evitar de imediato
- Adaptações práticas recomendadas na sala
- Pergunta de alinhamento técnico (apenas quando o caso exigir afunilamento ou validação da suspeita do professor)

--- HISTÓRICO RECENTE DO DIÁLOGO ---
{contexto_conversa if contexto_conversa else 'Início de conversa.'}
---------------------

--- MATERIAL DE CONSULTA (APOSTILA) ---
{contexto_recuperado}
--------------------------------------

Última mensagem da pessoa: {texto_busca}"""

        return executar_consulta_ia(prompt_completo)
    except Exception as erro:
        logger.error("Falha ao consultar o material: %s", erro)
        return "Tive um problema para consultar o material agora. Tenta de novo em alguns segundos."

# ==========================================
# INTEGRAÇÃO PLANILHA GOOGLE SHEETS
# ==========================================
URL_PLANILHA_W7 = st.secrets.get("URL_PLANILHA_W7", "")

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
                "Fala, time! **Jimmy** ao seu dispor. 🔴💪\n\n"
                "A base de condutas práticas para **lesões musculoesqueléticas** da W7 está carregada e pronta.\n\n"
                "Qual aluno ou situação de lesão vamos avaliar agora?"
            ),
        }
    ]

# Renderização das Mensagens
for idx, msg in enumerate(st.session_state.mensagens):
    icone = ICONE_ASSISTENTE if msg["role"] == "assistant" else ICONE_USUARIO
    with st.chat_message(msg["role"], avatar=icone):
        st.markdown(msg["content"])
        
        # Avaliação com joinha para respostas do Jimmy
        if msg["role"] == "assistant" and idx > 0:
                chave_fb = f"fb_{idx}"
                chave_registrado = f"{chave_fb}_registrado"
                fb = st.feedback("thumbs", key=chave_fb)
                if fb is not None and chave_registrado not in st.session_state:
                    pergunta_feita = st.session_state.mensagens[idx - 1]["content"]
                    registrar_feedback(pergunta_feita, msg["content"], fb)
                    st.session_state[chave_registrado] = True
                    st.toast("Feedback registrado!", icon="✅")

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
st.divider()
if ARQUIVO_FEEDBACK.exists():
    with open(ARQUIVO_FEEDBACK, "rb") as f_feedback:
        st.download_button(
            label="📥 Baixar Feedbacks da Equipe (CSV)",
            data=f_feedback,
            file_name="feedbacks_w7.csv",
            mime="text/csv",
            use_container_width=True
        )
        
