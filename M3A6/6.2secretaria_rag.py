from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnableMap
from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import List
from M3A6.utils import llm_loader
from langchain_core.messages import BaseMessage

# Instancia o modelo e parser
llm = llm_loader.llm_mistral()
parser = StrOutputParser()

# Carrega base vetorial (pré-processada com regulamentos, prazos etc.)
vectorstore = FAISS.load_local(
    "C:/Users/admin/IntelliJ_Workspace/Curso_LangChain/M3A6/utils/faiss_secretaria",
    embeddings=OllamaEmbeddings(model="mistral"),
    index_name="docs"
)
retriever = vectorstore.as_retriever()

# Runnable de recuperação + concatenação dos contextos
def format_docs(docs: List[Document]) -> str:
    return "\n---\n".join([doc.page_content for doc in docs])

retrieval_chain = RunnableMap({
    # x["input"] precisa ser uma string, não uma BaseMessage
    "context": RunnableLambda(lambda x: format_docs(retriever.invoke(
        x["input"].content if isinstance(x["input"], BaseMessage) else str(x["input"])
    ))),
    "input": lambda x: x["input"]
})

# Prompt com contexto recuperado
prompt_rag = ChatPromptTemplate.from_messages([
    ("system", "Você é um agente da secretaria acadêmica. Use o contexto abaixo para responder com precisão.\n\n{context}"),
    ("human", "{input}")
])

# Cadeia final com RAG
chain_rag = retrieval_chain | prompt_rag | llm | parser

# Históricos por aluno
histories = {}

def get_session_history(session_id: str):
    if session_id not in histories:
        histories[session_id] = InMemoryChatMessageHistory()
    return histories[session_id]

# Agente com memória + RAG
agente_secretaria = RunnableWithMessageHistory(
    runnable=chain_rag,
    get_session_history=get_session_history,
    input_messages_key="input"
)

# Simulação
session_id = "aluno_larissa"
config = RunnableConfig(configurable={"session_id": session_id})

resposta1 = agente_secretaria.invoke({"input": "Preciso trancar a disciplina de Cálculo 2."}, config=config)
print("R1: " + resposta1)
resposta2 = agente_secretaria.invoke({"input": "Qual o prazo para isso mesmo?"}, config=config)
print("R2: " + resposta2)
resposta3 = agente_secretaria.invoke({"input": "Quais documentos eu preciso entregar?"}, config=config)
print("R3: " + resposta3)