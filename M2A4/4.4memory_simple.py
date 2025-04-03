from os import environ
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from M2A3.utils import llm_loader



template = """
You are a helpful assistant. Help the user with their requests.
"""


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}"),
    ]
)

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

llm_chain = LLMChain(
    llm=llm_loader.llm_mistral(),
    prompt=prompt_template,
    verbose=False,
    memory=memory,
)



response = llm_chain.run("Hi GPT, i am going to market in the evening and i need to buy milk, bread, almonds and bananas. Can you remember it for me?")
print(response)
response = llm_chain.run("Can you remind me all items i need to buy today?")
print(response)