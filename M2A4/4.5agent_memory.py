from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.runnables import RunnableWithMessageHistory
from langchain import hub
from langchain.memory import ChatMessageHistory
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from M2A3.utils import llm_loader


def explicar_conceito(conceito: str) -> str:
    return f"'{conceito}' é um conceito essencial. Recomenda-se começar pelos fundamentos teóricos e exemplos práticos."

def resumir_texto(texto: str) -> str:
    return f"Resumo: {texto}..."

def sugerir_material(tema: str) -> str:
    return f"Sugestão: Consulte o livro 'Introdução a {tema}' da editora Universitária."

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia = WikipediaQueryRun(description="A tool to explain things in text format. Use this tool if you think the user’s asked concept is best explained through text.", api_wrapper=wiki_api_wrapper)

tools = [
    wikipedia
]

memory = ChatMessageHistory(session_id="test-session")

llm = llm_loader.llm_mistral()


prompt = hub.pull("hwchase17/react")


agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# primeira pergunta
resposta = agent_with_chat_history.invoke(
    {"input": "Quantas pessoas vivem no Brasil?"},
    config={"configurable": {"session_id": "test-session"}}
)
print("Resposta do agente:\n", resposta)


# segunda pergunta
resposta = agent_with_chat_history.invoke(
    {"input": "Como é chamado seu hino nacional?"},
    config={"configurable": {"session_id": "test-session"}}
)
print("Resposta do agente:\n", resposta)