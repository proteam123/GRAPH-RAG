# GraphRAG with Neo4j 🕸️

## 📌 Project Description

A Graph-based Retrieval-Augmented Generation (GraphRAG) application that uses Large Language Models (LLMs) and Neo4j to retrieve information through connected entities and relationships.

The project extracts entities and relationships from text documents, stores them as a knowledge graph, and uses graph traversal to answer natural language questions.

A movie dataset is used to demonstrate how relationships between directors, actors, movies, and genres can be extracted and queried intelligently.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **LLM Provider:** Groq
- **LLM Model:** GPT-OSS-120B
- **LLM Framework:** LangChain
- **Graph Processing:** LangChain Experimental – LLMGraphTransformer
- **Graph Database:** Neo4j Aura
- **Graph Query Language:** Cypher
- **Graph QA:** GraphCypherQAChain
- **Development Environment:** Jupyter Notebook / Google Colab

## ⚙️ Process Pipeline

1. **Document Loading:** Load text documents containing movie information.
2. **Text Chunking:** Split documents into smaller chunks using RecursiveCharacterTextSplitter.
3. **Graph Extraction:** Use an LLM to identify entities such as Person, Movie, and Genre.
4. **Relationship Extraction:** Identify connections such as DIRECTED, ACTED_IN, and IN_GENRE.
5. **Graph Creation:** Convert extracted information into graph documents.
6. **Graph Storage:** Store nodes and relationships in Neo4j.
7. **Schema Inspection:** Retrieve the graph structure and available relationships.
8. **Graph Retrieval:** Use Cypher queries to retrieve connected information.
9. **Query Generation:** Convert natural language questions into Cypher queries using GraphCypherQAChain.
10. **Answer Generation:** Pass retrieved graph information to the LLM to generate natural language answers.

## 📚 What I Learned

- Understanding GraphRAG architecture and its working principles.
- Extracting entities and relationships using LLMs.
- Building knowledge graphs from unstructured text.
- Working with Neo4j and Cypher queries.
- Using LangChain's LLMGraphTransformer.
- Understanding graph-based retrieval compared to traditional vector retrieval.
- Connecting LLMs with graph databases.
- Building question-answering systems using GraphCypherQAChain.

## 🚀 Future Improvements

- Integrate document loaders for PDFs, websites, and multiple file formats.
- Support larger datasets and more complex knowledge graphs.
- Combine vector similarity search with graph-based retrieval for Hybrid GraphRAG.
- Improve entity resolution to identify duplicate entities.
- Add a Streamlit or FastAPI interface for interactive querying.
- Implement graph visualization to explore nodes and relationships.
- Add source citations to improve answer traceability.
- Strengthen Cypher query validation and database access controls.

## ▶️ How to Run the Project

**1. Install dependencies**

```bash
pip install --upgrade langchain langchain-community langchain-experimental langchain-groq langchain-neo4j neo4j langchain-text-splitters
```

**2. Set up Neo4j**

- Create a database using Neo4j Aura or a local Neo4j instance.
- Obtain your connection URI, username, password, and database name.

**3. Configure credentials**

Add your Neo4j credentials and Groq API key to the notebook:

```python
NEO4J_URI = "your_neo4j_connection_string"
NEO4J_USERNAME = "your_username"
NEO4J_PASSWORD = "your_password"
NEO4J_DATABASE = "your_database"

groq_api_key = "your_groq_api_key"
```

**4. Open the notebook**

Open `Graph_RAG_Neo4j.ipynb` using Jupyter Notebook or Google Colab.

**5. Execute the notebook**

Run the cells sequentially to:
- Connect to Neo4j.
- Initialize the Groq LLM.
- Load and chunk movie documents.
- Extract entities and relationships.
- Store the knowledge graph.
- Execute graph retrieval and question-answering queries.

**6. Try sample questions**

```text
Which movies did Christopher Nolan direct?

Who acted in Inception?

Which movie did Christopher Nolan direct in 2010
and who acted in it?
```

---

⭐ Built to explore Knowledge Graphs, LLM-powered information extraction, and Graph-based Retrieval-Augmented Generation.
