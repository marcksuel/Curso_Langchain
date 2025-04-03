from langchain_community.tools import ArxivQueryRun
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent

# from langchain_community.agent_toolkits.load_tools import load_tools
# tools = load_tools(["arxiv", "google-search"])

# Tool do Arxiv
from langchain_core.messages import HumanMessage, SystemMessage

from M2A3.utils import llm_loader

arxiv = ArxivQueryRun(description="A tool to search for papers and articles in journals and conferences. Use this tool if you think the user’s asked concept can be best explained by read papers and articles.")
# print(arxiv.invoke('Data Quality')[:250])

# Tool da Wikipedia
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia = WikipediaQueryRun(description="A tool to explain things in text format. Use this tool if you think the user’s asked concept is best explained through text.", api_wrapper=wiki_api_wrapper)
# print(wikipedia.invoke("Corinthians"))

# Tool do Youtube
youtube = YouTubeSearchTool(
    description="A tool to search YouTube videos. Use this tool if you think the user’s asked concept can be best explained by watching a video."
)
# print(youtube.invoke("Oiling a bike's chain"))

tools = [wikipedia, arxiv, youtube]

# Outra forma de carregar ferramentas no chat
chat_model = llm_loader.llm_mistral()
model_with_tools = chat_model.bind_tools(tools)

# Sem agente
# response = model_with_tools.invoke("Can you find a video to watch about LangChain?")
# print(f"Text response: {response.content}")
# print(f"Tools used in the response: {response.tool_calls}")

# Com agente
system_prompt = SystemMessage("You are a helpful bot named Chandler.")
agent = create_react_agent(chat_model, tools, state_modifier=system_prompt)

response = agent.invoke({"messages": [
    HumanMessage('Can you find a video to watch about LangChain?')
]})

# Analisar os "logs" do usuário, agente e llm
for message in response['messages']:
    print(
        f"{message.__class__.__name__}: {message.content}"
    )
    print("-" * 20, end="\n")