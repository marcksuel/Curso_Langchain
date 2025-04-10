from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

import os

# Caminho para o PDF com regulamento acadêmico
pdf_path = "dados_externos/regulamento_secretaria.pdf"

# Carrega o documento
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Divide o texto em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# Gera os embeddings com Ollama
embeddings = OllamaEmbeddings(model="mistral")

# Cria o índice vetorial
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

# Salva localmente
vectorstore.save_local("./faiss_secretaria", index_name="docs")

print("Base vetorial criada e salva com sucesso!")