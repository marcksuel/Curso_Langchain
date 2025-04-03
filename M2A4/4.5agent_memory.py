from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.runnables import RunnableWithMessageHistory
from langchain import hub
from langchain.memory import ChatMessageHistory
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from M2A3.utils import llm_loader

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia = WikipediaQueryRun(description="A tool to explain things in text format. Use this tool if you think the user’s asked concept is best explained through text.", api_wrapper=wiki_api_wrapper)

tools = [
    wikipedia
]

# Configurando a memória
memory = ChatMessageHistory(session_id="test-session")

# Criando a LLM
llm = llm_loader.llm_mistral()

# Carregando o prompt via Hub, lib com vários modelos de prompts
prompt = hub.pull("hwchase17/react")

# Criação dos agentes
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Criando o executável com memória
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# primeira pergunta
resposta = agent_with_chat_history.invoke(
    {"input": "O que é um banco NoSQL?"},
    config={"configurable": {"session_id": "test-session"}}
)
print("Resposta do agente:\n", resposta)


# segunda pergunta
resposta = agent_with_chat_history.invoke(
    {"input": "Cite alguns exemplos desse tipo de banco-."},
    config={"configurable": {"session_id": "test-session"}}
)
print("Resposta do agente:\n", resposta)