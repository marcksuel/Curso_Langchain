from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.chat_history import InMemoryChatMessageHistory
from M3A5.utils import llm_loader

llm = llm_loader.llm_mistral()

# Prompt do agente da secretaria
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente da secretaria acadêmica. Seja objetivo e claro nas orientações."),
    ("human", "{input}")
])

# Parser de saída
parser = StrOutputParser()

# Cadeia base
chain_base = prompt | llm | parser

# Históricos individuais
histories = {}

# Função que retorna memória resumida com base em um identificador de sessão
def get_session_history(session_id: str):
    if session_id not in histories:
        histories[session_id] = InMemoryChatMessageHistory()
    return histories[session_id]

# Chain com memória resumida
chain_secretaria = RunnableWithMessageHistory(
    runnable=chain_base,
    get_session_history=get_session_history,
    input_messages_key="input"
)

# Execução com um aluno
session_id = "aluno_bruno"
config = RunnableConfig(configurable={"session_id": session_id})

resposta1 = chain_secretaria.invoke({"input": "Quais documentos preciso para trancar a disciplina de Cálculo 1?"}, config=config)
print("R1: " + resposta1)
resposta2 = chain_secretaria.invoke({"input": "Qual assunto abordamos?"}, config=config)
print("R2: " + resposta2)