from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.chat_history import InMemoryChatMessageHistory
from M3A5.utils import llm_loader

# Instancia o modelo (Ollama local)
llm = llm_loader.llm_mistral()

# Prompt simples (usado em todos os casos)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente universitário."),
    ("human", "{input}")
])

# Parser de saída padrão
parser = StrOutputParser()

# Define a cadeia base
chain_base = prompt | llm | parser

# Função para histórico de sessão
histories = {}
def get_session_history(session_id: str):
    if session_id not in histories:
        histories[session_id] = InMemoryChatMessageHistory()
    return histories[session_id]

# Cadeia com memória
# Encapsulando com memória: Esse objeto final já está pronto para ser usado com
# múltiplos usuários e lembrar do que foi dito.
chain_tutor = RunnableWithMessageHistory(
    runnable=chain_base,
    get_session_history=get_session_history,
    input_messages_key="input"
)

# Execução com sessão "aluno_ana"
# Na prática, o agente agora vai lembrar de Ana e suas dificuldades nas próximas interações.
session_id = "aluno_ana"
config = RunnableConfig(configurable={"session_id": session_id})
resposta1 = chain_tutor.invoke({"input": "Meu nome é Ana. O que posso esperar de um curso de Sistemas de Informação?"}, config=config)
print(resposta1)
resposta2 = chain_tutor.invoke({"input": "O que já conversamos até agora?"}, config=config)
print(resposta2)