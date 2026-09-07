import os
import re
import time
from pathlib import Path

import streamlit as st
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
import pdfplumber

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="W7 Academy | Consultor da Apostila",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# 2. CONEXÃO & MODELOS
# ==========================================
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

MODELO_CHAT = "gemini-3.6-flash"
MODELO_EMBEDDING = "gemini-embedding-001"  # multilíngue, muito melhor para termos técnicos em PT-BR
                                            # que o embedding local padrão do ChromaDB

DIRETORIO_ATUAL = Path(__file__).resolve().parent
DIRETORIO_BANCO = DIRETORIO_ATUAL / "w7_database"
DIRETORIO_MATERIAIS = DIRETORIO_ATUAL / "materiais"  # coloque aqui TODOS os PDFs (apostila, ebooks, etc.)

# ==========================================
# 3. EMBEDDING VIA GEMINI (substitui o padrão local do Chroma)
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    """Usa o modelo de embedding do Gemini em vez do modelo local padrão do ChromaDB,
    que é fraco para jargão técnico em português."""

    def __call__(self, input: Documents) -> Embeddings:
        resultado = client.models.embed_content(
            model=MODELO_EMBEDDING,
            contents=input,
        )
        return [emb.values for emb in resultado.embeddings]


FUNCAO_EMBEDDING = GeminiEmbeddingFunction()

# ==========================================
# 4. CHUNKING RESPEITANDO PARÁGRAFOS/TABELAS
# ==========================================
def fatiar_texto(texto: str, tamanho_alvo: int = 900, sobreposicao_paragrafos: int = 1) -> list[str]:
    """Fatia o texto por parágrafos (linhas em branco), nunca no meio de uma palavra
    ou de uma linha de tabela. Agrupa parágrafos pequenos até o tamanho_alvo;
    parágrafos maiores que o alvo são mantidos inteiros (evita cortar uma
    linha de exercício/tabela no meio, como acontecia com o corte por caractere fixo).
    """
    paragrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]
    if not paragrafos:
        return []

    blocos = []
    atual = []
    tamanho_atual = 0

    for paragrafo in paragrafos:
        if tamanho_atual + len(paragrafo) > tamanho_alvo and atual:
            blocos.append("\n\n".join(atual))
            # sobreposição: mantém o(s) último(s) parágrafo(s) do bloco anterior
            atual = atual[-sobreposicao_paragrafos:] if sobreposicao_paragrafos else []
            tamanho_atual = sum(len(p) for p in atual)
        atual.append(paragrafo)
        tamanho_atual += len(paragrafo)

    if atual:
        blocos.append("\n\n".join(atual))

    return blocos


# ==========================================
# 5. BANCO VETORIAL — MULTI-ARQUIVO, INDEXAÇÃO INCREMENTAL
# ==========================================
@st.cache_resource
def obter_colecao():
    """Lê TODOS os PDFs em DIRETORIO_MATERIAIS e indexa apenas os que ainda não
    estão no banco (por nome de arquivo), permitindo adicionar novos materiais
    sem apagar o banco existente."""
    cliente_chroma = chromadb.PersistentClient(path=str(DIRETORIO_BANCO))
    colecao = cliente_chroma.get_or_create_collection(
        name="conhecimento_w7",
        embedding_function=FUNCAO_EMBEDDING,
    )

    if not DIRETORIO_MATERIAIS.exists():
        return colecao

    # Descobre quais arquivos já foram indexados (pelo prefixo do id)
    arquivos_indexados = set()
    total_existente = colecao.count()
    if total_existente > 0:
        amostra = colecao.get(limit=total_existente, include=[])
        for id_doc in amostra["ids"]:
            arquivos_indexados.add(id_doc.split("__pag_")[0])

    for caminho_pdf in sorted(DIRETORIO_MATERIAIS.glob("*.pdf")):
        slug_arquivo = re.sub(r"[^a-z0-9]+", "-", caminho_pdf.stem.lower()).strip("-")
        if slug_arquivo in arquivos_indexados:
            continue  # já indexado — não reprocessa

        docs, metas, ids = [], [], []
        with pdfplumber.open(caminho_pdf) as pdf:
            for num_pag, pagina in enumerate(pdf.pages, start=1):
                texto = pagina.extract_text() or ""
                if not texto.strip():
                    continue
                pedacos = fatiar_texto(texto.strip())
                for idx, pedaco in enumerate(pedacos):
                    docs.append(pedaco)
                    metas.append({
                        "fonte": caminho_pdf.name,
                        "pagina": num_pag,
                        "bloco": idx + 1,
                    })
                    ids.append(f"{slug_arquivo}__pag_{num_pag}_b_{idx + 1}")

        if docs:
            tamanho_lote = 50  # lotes menores: cada chamada de embedding tem custo de API
            for i in range(0, len(docs), tamanho_lote):
                colecao.add(
                    documents=docs[i:i + tamanho_lote],
                    metadatas=metas[i:i + tamanho_lote],
                    ids=ids[i:i + tamanho_lote],
                )

    return colecao


# ==========================================
# 6. CONSULTA À IA — COM HISTÓRICO E RETRY (429 + 503)
# ==========================================
SYSTEM_PROMPT = """Você é o Jimmy, da W7 Academy. Não se apresente como consultor,
especialista ou qualquer outro título — você é só o Jimmy.
Sua base de conhecimento hoje é a apostila "Lesões e Principais Movimentos para o
Tratamento" — sobre lesões musculoesqueléticas, fisiopatologia e reabilitação por
meio de exercícios de academia. Você NÃO é especialista em biomecânica aplicada à
musculação; não responda como se tivesse esse conhecimento além do que está no
CONTEXTO fornecido a cada pergunta.

DIRETRIZES DE COMUNICAÇÃO:
1. NUNCA use saudações ou vocativos como se fossem nomes (nunca diga "Fala!", "Boa pergunta, Tenho!" etc.)
2. Didática em camadas: responda em 1-2 parágrafos ou tópicos diretos com o conceito principal,
   destaque o ponto prático/biomecânico central, e finalize com uma pergunta curta oferecendo
   aprofundar um detalhe específico. Não despeje todo o conteúdo de uma vez.
3. RECONHEÇA SINÔNIMOS E TERMINOLOGIA DESATUALIZADA: se o usuário usar um termo popular ou
   antigo (ex: "epicondilite", "tendinite") que a apostila trata sob um nome técnico atualizado
   (ex: "epicondilalgia", "tendinopatia"), NÃO trate como assunto fora do material — identifique
   a correspondência, use o conteúdo real da apostila sobre aquele tema, e aproveite para explicar
   a diferença de nomenclatura como parte da resposta.
4. Baseie toda a análise técnica estritamente no CONTEXTO fornecido a cada pergunta.
5. Se o assunto for genuinamente alheio ao material (sem base mecânica no contexto), informe de
   forma profissional que o tópico não está contemplado na apostila.
6. Este material é educacional e não substitui avaliação, diagnóstico ou acompanhamento por um
   profissional de saúde qualificado — deixe isso claro sempre que a dúvida envolver dor, lesão
   ou sintoma específico de um aluno."""


def _deve_tentar_novamente(erro_msg: str) -> bool:
    return any(codigo in erro_msg for codigo in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))


def executar_consulta_ia(historico_contents: list, contexto_recuperado: str):
    """Chama o Gemini com streaming real, histórico de conversa e retry para 429/503."""
    system_instruction = f"{SYSTEM_PROMPT}\n\n--- CONTEXTO DA APOSTILA W7 ---\n{contexto_recuperado}\n---"

    tentativas = 3
    intervalo = 4

    for tentativa in range(tentativas):
        try:
            stream = client.models.generate_content_stream(
                model=MODELO_CHAT,
                contents=historico_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                ),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
            return

        except Exception as erro:
            erro_msg = str(erro)
            if _deve_tentar_novamente(erro_msg) and tentativa < tentativas - 1:
                time.sleep(intervalo)
                intervalo *= 2
                continue

            if _deve_tentar_novamente(erro_msg):
                yield "⏳ O servidor atingiu a cota momentânea ou está instável. Aguarde 30 segundos e envie sua dúvida novamente."
            else:
                yield f"⚠️ Instabilidade temporária. Detalhes: {erro_msg[:80]}"
            return


def consultar_cerebro_w7(pergunta_usuario: str, historico_mensagens: list, nome_usuario: str = "Aluno"):
    texto_busca = str(pergunta_usuario).strip()
    if not texto_busca:
        yield "Por favor, digite uma dúvida válida."
        return

    try:
        colecao = obter_colecao()
        resultados = colecao.query(query_texts=[texto_busca], n_results=6)
        docs_brutos = resultados.get("documents", [[]])[0] if resultados.get("documents") else []
        documentos = [d for d in docs_brutos if isinstance(d, str) and d.strip()]
        contexto_recuperado = "\n\n---\n\n".join(documentos) if documentos else "Nenhum trecho correspondente encontrado na apostila."
    except Exception as erro:
        yield f"Erro ao consultar a base de dados: {str(erro)}"
        return

    # Monta o histórico completo (multi-turn) para o Gemini, não só a pergunta atual
    contents = []
    for msg in historico_mensagens[-10:]:  # últimas 10 mensagens é suficiente de contexto
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=f"({nome_usuario} pergunta) {texto_busca}")]))

    yield from executar_consulta_ia(contents, contexto_recuperado)


# ==========================================
# 7. INTERFACE DO USUÁRIO
# ==========================================
ICONE_ASSISTENTE = "🧠"
ICONE_USUARIO = "👤"

PADROES_NOME = [
    r"^meu\s+nome\s+é\s+", r"^me\s+chamo\s+", r"^sou\s+(?:o|a)\s+",
    r"^é\s+(?:o|a)\s+", r"^pode\s+me\s+chamar\s+de\s+",
]

def extrair_nome(texto: str) -> str:
    """Extração mais robusta: remove padrões conhecidos de introdução;
    se nada bater, usa a primeira palavra SÓ se a frase inteira tiver
    até 3 palavras (evita capturar a primeira palavra de uma frase longa,
    que foi o bug original)."""
    limpo = texto.strip()
    for padrao in PADROES_NOME:
        limpo = re.sub(padrao, "", limpo, flags=re.IGNORECASE)
    limpo = limpo.strip()

    palavras = limpo.split()
    if len(texto.split()) <= 3 and palavras:
        return palavras[0].title()
    if palavras:
        return palavras[0].title()
    return "Aluno"


st.title("🧠 Jimmy | W7 Academy")
st.caption("Seu parceiro de estudos em biomecânica e cinesiologia.")

if "nome_usuario" not in st.session_state:
    st.session_state.nome_usuario = None

if "mensagens" not in st.session_state or not st.session_state.mensagens:
    st.session_state.mensagens = [
        {
            "role": "assistant",
            "content": "Fala! Eu sou o **Jimmy**, seu parceiro de estudos aqui na **W7 Academy** 🧠💪\n\nAntes de começarmos, qual é o seu nome?"
        }
    ]

for msg in st.session_state.mensagens:
    icone = ICONE_ASSISTENTE if msg["role"] == "assistant" else ICONE_USUARIO
    with st.chat_message(msg["role"], avatar=icone):
        st.markdown(msg["content"])

if prompt_usuario := st.chat_input("Digite sua resposta ou dúvida aqui..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user", avatar=ICONE_USUARIO):
        st.markdown(prompt_usuario)

    if not st.session_state.nome_usuario:
        st.session_state.nome_usuario = extrair_nome(prompt_usuario)
        boas_vindas = f"Prazer, **{st.session_state.nome_usuario}**! 🚀 Agora sim.\n\nQual dúvida sobre cinesiologia, biomecânica ou exercícios vamos desvendar hoje?"
        with st.chat_message("assistant", avatar=ICONE_ASSISTENTE):
            st.markdown(boas_vindas)
        st.session_state.mensagens.append({"role": "assistant", "content": boas_vindas})
    else:
        with st.chat_message("assistant", avatar=ICONE_ASSISTENTE):
            with st.spinner(f"Jimmy consultando a apostila para {st.session_state.nome_usuario}..."):
                resposta_ia = st.write_stream(
                    consultar_cerebro_w7(
                        prompt_usuario,
                        st.session_state.mensagens[:-1],  # histórico sem a msg atual (já vai na pergunta)
                        st.session_state.nome_usuario,
                    )
                )
        st.session_state.mensagens.append({"role": "assistant", "content": resposta_ia})
