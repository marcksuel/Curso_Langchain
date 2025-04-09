from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from M3A5.utils import llm_loader


# Classe de memória com janela deslizante
class ConversationBufferWindowMemory(BaseChatMessageHistory):
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self._messages: List[BaseMessage] = []

    @property
    def messages(self):
        return self._messages

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self._messages.extend(messages)
        self._messages = self._messages[-self.buffer_size:]

    def clear(self):
        self._messages = []

# Instancia o modelo
llm = llm_loader.llm_mistral()

# Prompt do agente de apoio ao docente
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente que ajuda professores a planejar aulas e relembrar os últimos temas discutidos."),
    ("human", "{input}")
])

# Parser de saída
parser = StrOutputParser()

# Cadeia base
chain_base = prompt | llm | parser

# Históricos por docente
histories = {}

def get_session_history(session_id: str):
    if session_id not in histories:
        histories[session_id] = ConversationBufferWindowMemory(buffer_size=3)
    return histories[session_id]

# Cadeia com memória de janela
chain_docente = RunnableWithMessageHistory(
    runnable=chain_base,
    get_session_history=get_session_history,
    input_messages_key="input"
)

# Execução com a professora Carla
session_id = "prof_carla"
config = RunnableConfig(configurable={"session_id": session_id})

resposta1 = chain_docente.invoke({"input": "Aula de hoje foi sobre Sistemas de Equações."}, config=config)
print("R1: " + resposta1)
resposta2 = chain_docente.invoke({"input": "Próxima será sobre Equações de Primeiro Grau."}, config=config)
print("R2: " + resposta2)
resposta3 = chain_docente.invoke({"input": "O que falei nas últimas aulas?"}, config=config)
print("R3: " + resposta3)