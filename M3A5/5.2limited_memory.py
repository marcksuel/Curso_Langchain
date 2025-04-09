from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig



class ConversationBufferWindowMemory(BaseChatMessageHistory):
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self._messages.extend(messages)
        self._messages = self._messages[-self.buffer_size:]

    def clear(self) -> None:
        self._messages = []

histories = {}
def get_conversation_buffer(session_id: str) -> BaseChatMessageHistory:
    buffer_size = 2  # tamanho da janela temporal
    if session_id not in histories:
        histories[session_id] = ConversationBufferWindowMemory(buffer_size=buffer_size)
    return histories[session_id]

# Instancia o modelo (Ollama local)
llm = ChatOllama(model="mistral")

# Prompt simples (usado em todos os casos)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente universitário."),
    ("human", "{input}")
])

# Parser de saída padrão
parser = StrOutputParser()

# Define a cadeia base
chain_base = prompt | llm | parser

chain_tutor = RunnableWithMessageHistory(
    runnable=chain_base,
    get_session_history=get_conversation_buffer,
    input_messages_key="input"
)

session_id = "aluno_ana"
config = RunnableConfig(configurable={"session_id": session_id})
resposta1 = chain_tutor.invoke({"input": "Meu nome é Ana. Qual a capital do Brasil?"}, config=config)
print(resposta1)
resposta2 = chain_tutor.invoke({"input": "O que já conversamos até agora?"}, config=config)
print(resposta2)
resposta3 = chain_tutor.invoke({"input": "Qual a capital da Argentina?"}, config=config)
print(resposta3)
resposta4 = chain_tutor.invoke({"input": "O que já conversamos até agora?"}, config=config)
print(resposta4)


# R1: A capital da Brasil é Brasília. Foi construída em 1960 e localiza-se no centro-oeste do país.
# R2: Até agora, falamos sobre a capital do Brasil. Você tem mais alguma dúvida ou gostaria de discutir um outro assunto?
# Por exemplo, podemos falar sobre história, matemática, física ou qualquer outro tema que você deseja.
# R3: A capital da Argentina é Buenos Aires. Buenos Aires foi fundada em 1536 pelo espanhol Pedro de Mendoza e oficialmente se tornou capital do país em 1862, após a unificação da confederação argentina. É uma cidade importante no sul do continente americano, conhecida por sua cultura vibrante e suas tradições, como o futebol e a tango.
# R4: 1. Você perguntou qual é a capital da Argentina. Eu respondi que Buenos Aires é a capital da Argentina e detalhei alguns aspectos históricos e culturais sobre a cidade.
# 2. Depois disso, você solicitou se já tínhamos falado sobre alguma coisa até agora.