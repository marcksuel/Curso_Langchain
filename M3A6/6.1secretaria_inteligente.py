from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.chat_history import InMemoryChatMessageHistory
from M3A6.utils import llm_loader

# Prompt do agente
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um agente da secretaria acadêmica. Ajude alunos com dúvidas sobre matrícula, trancamento, documentos e prazos."),
    ("human", "{input}")
])

# Cadeia base com modelo e parser
llm = llm_loader.llm_mistral()
parser = StrOutputParser()
chain_base = prompt | llm | parser

# Históricos por aluno
histories = {}
def get_session_history(session_id: str):
    if session_id not in histories:
        histories[session_id] = InMemoryChatMessageHistory()
    return histories[session_id]

# Agente com memória
agente_secretaria = RunnableWithMessageHistory(
    runnable=chain_base,
    get_session_history=get_session_history,
    input_messages_key="input"
)

# Simulação
session_id = "aluno_larissa"
config = RunnableConfig(configurable={"session_id": session_id})

resposta = agente_secretaria.invoke({"input": "Preciso trancar a disciplina de Cálculo 2."}, config=config)
print("R1: " + resposta)
resposta = agente_secretaria.invoke({"input": "Qual o prazo para isso mesmo?"}, config=config)
print("R2: " + resposta)
resposta = agente_secretaria.invoke({"input": "Quais documentos eu preciso entregar?"}, config=config)
print("R3: " + resposta)