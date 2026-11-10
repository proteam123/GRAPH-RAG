"""
GraphRAG app: upload documents -> build a knowledge graph in YOUR Neo4j database -> chat with it.

Run:
    pip install -r requirements.txt
    (fill in .env, see .env.example)
    (if using a custom Neo4j TLS CA, set NEO4J_TRUSTED_CA_FILE in .env)
    streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import TrustCustomCAs

load_dotenv()

st.set_page_config(page_title="GraphRAG with Neo4j", page_icon="🕸️", layout="wide")


# ---------- connections ----------
@st.cache_resource(show_spinner="Connecting to Neo4j...")
def get_graph() -> Neo4jGraph:
    driver_config = {}
    trusted_ca_file = os.getenv("NEO4J_TRUSTED_CA_FILE")
    if trusted_ca_file:
        ca_path = Path(trusted_ca_file).expanduser()
        if not ca_path.is_file():
            raise FileNotFoundError(
                f"NEO4J_TRUSTED_CA_FILE does not point to a file: {ca_path}"
            )
        driver_config["trusted_certificates"] = TrustCustomCAs(str(ca_path))

    return Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.environ["NEO4J_PASSWORD"],
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        driver_config=driver_config,
    )


@st.cache_resource
def get_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        groq_api_key=os.environ["GROQ_API_KEY"],
        reasoning_effort="low",
        model_kwargs={"parallel_tool_calls": False},
    )


def missing_env() -> list[str]:
    return [k for k in ("NEO4J_URI", "NEO4J_PASSWORD", "GROQ_API_KEY") if not os.getenv(k)]


# ---------- reading uploaded files ----------
def read_upload(f) -> list[Document]:
    name = f.name
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(f)
        return [
            Document(page_content=page.extract_text() or "", metadata={"source": name, "page": i + 1})
            for i, page in enumerate(reader.pages)
            if (page.extract_text() or "").strip()
        ]
    if suffix == ".docx":
        import docx

        text = "\n".join(p.text for p in docx.Document(f).paragraphs)
        return [Document(page_content=text, metadata={"source": name})]
    # .txt / .md / anything text-like
    text = f.read().decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": name})]


def ingest(files, free_schema: bool, chunk_size: int, max_chunks: int):
    graph, llm = get_graph(), get_llm()

    documents = []
    for f in files:
        documents.extend(read_upload(f))
    if not documents:
        st.error("No readable text found in the uploaded file(s). Scanned PDFs need OCR first.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size // 10)
    chunks = splitter.split_documents(documents)
    if len(chunks) > max_chunks:
        st.warning(f"{len(chunks)} chunks found - processing the first {max_chunks} (change this in the sidebar).")
        chunks = chunks[:max_chunks]

    if free_schema:
        transformer = LLMGraphTransformer(llm=llm)
    else:
        nodes = [n.strip() for n in st.session_state.get("allowed_nodes", "").split(",") if n.strip()]
        rels = [r.strip() for r in st.session_state.get("allowed_rels", "").split(",") if r.strip()]
        transformer = LLMGraphTransformer(
            llm=llm, allowed_nodes=nodes or None, allowed_relationships=rels or None
        )

    progress = st.progress(0.0, text="Extracting entities and relationships...")
    total_nodes = total_rels = 0
    batch = 5
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        try:
            graph_docs = transformer.convert_to_graph_documents(part)
        except Exception as e:  # keep going if one batch fails (rate limit, bad JSON, ...)
            st.warning(f"Batch {i // batch + 1} failed and was skipped: {e}")
            continue
        graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
        total_nodes += sum(len(g.nodes) for g in graph_docs)
        total_rels += sum(len(g.relationships) for g in graph_docs)
        progress.progress(min((i + batch) / len(chunks), 1.0), text=f"Processed {min(i + batch, len(chunks))}/{len(chunks)} chunks")

    graph.refresh_schema()
    progress.empty()
    st.success(f"Done: {len(chunks)} chunks -> {total_nodes} nodes, {total_rels} relationships added to Neo4j.")


def answer(question: str):
    graph = get_graph()
    graph.refresh_schema()  # pick up newly ingested labels/relationships
    chain = GraphCypherQAChain.from_llm(
        llm=get_llm(),
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        validate_cypher=True,
        allow_dangerous_requests=True,
    )
    out = chain.invoke({"query": question})
    steps = out.get("intermediate_steps", [])
    cypher = steps[0].get("query", "") if steps else ""
    context = steps[1].get("context", []) if len(steps) > 1 else []
    return out["result"], cypher, context


# ---------- UI ----------
st.title("🕸️ GraphRAG with Neo4j")
st.caption("Upload documents → LLM extracts a knowledge graph → stored in your Neo4j → ask questions in plain English")

if missing_env():
    st.error(f"Missing in your .env file: {', '.join(missing_env())}. See .env.example.")
    st.stop()

ssl_cert_file = os.getenv("SSL_CERT_FILE")
if ssl_cert_file:
    ssl_cert_path = Path(ssl_cert_file).expanduser()
    if not ssl_cert_path.is_file():
        st.error(f"SSL_CERT_FILE does not point to an existing file: {ssl_cert_path}")
        st.info(
            "Set SSL_CERT_FILE to an existing PEM CA bundle, or remove that environment "
            "variable if you do not need a custom CA. Do not disable certificate verification."
        )
        st.stop()

try:
    graph = get_graph()
    graph.query("RETURN 1")
except Exception as e:
    st.error(f"Could not connect to Neo4j: {e}")
    st.info("Check NEO4J_URI (neo4j+s:// for Aura, bolt://localhost:7687 for local), username, password and database name.")
    st.info(
        "If your network uses TLS inspection or a custom CA, set "
        "NEO4J_TRUSTED_CA_FILE in .env to the CA certificate file in PEM format. "
        "Do not disable certificate verification."
    )
    st.stop()

with st.sidebar:
    st.header("1. Upload documents")
    files = st.file_uploader("PDF, DOCX, TXT or MD", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)

    st.subheader("Graph options")
    free_schema = st.checkbox("Let the LLM choose node/relationship types", value=True)
    if not free_schema:
        st.text_input("Allowed node types", "Person, Organization, Location, Concept", key="allowed_nodes")
        st.text_input("Allowed relationship types", "WORKS_AT, LOCATED_IN, RELATED_TO", key="allowed_rels")
    chunk_size = st.slider("Chunk size (characters)", 500, 3000, 1500, 100)
    max_chunks = st.slider("Max chunks to process", 5, 200, 40, 5)

    if st.button("Build knowledge graph", type="primary", disabled=not files, use_container_width=True):
        with st.spinner("Building graph..."):
            ingest(files, free_schema, chunk_size, max_chunks)

    st.divider()
    st.header("Database")
    if st.button("Show schema / counts", use_container_width=True):
        graph.refresh_schema()
        n = graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
        r = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        st.write(f"**Nodes:** {n}  |  **Relationships:** {r}")
        st.code(graph.schema or "(empty)", language="text")

    with st.expander("Danger zone"):
        if st.checkbox("I understand this deletes everything"):
            if st.button("Clear database"):
                graph.query("MATCH (n) DETACH DELETE n")
                graph.refresh_schema()
                st.session_state.messages = []
                st.success("Database cleared.")

st.header("2. Ask questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("cypher"):
            with st.expander("How the graph was queried"):
                st.code(m["cypher"], language="cypher")
                st.json(m.get("context", []))

if question := st.chat_input("Ask something about your uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the graph..."):
            try:
                result, cypher, context = answer(question)
            except Exception as e:
                result, cypher, context = f"Something went wrong: {e}", "", []
        st.markdown(result)
        if cypher:
            with st.expander("How the graph was queried"):
                st.code(cypher, language="cypher")
                st.json(context)
    st.session_state.messages.append(
        {"role": "assistant", "content": result, "cypher": cypher, "context": context}
    )