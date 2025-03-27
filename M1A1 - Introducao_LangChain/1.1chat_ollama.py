from langchain_ollama import ChatOllama

llm = ChatOllama(
    # model="mistral",
    model="llama3.2",
    temperature=0.9
)

messages = [
    ("system", "You are a helpful assistant that translates English to Portuguese. Translate the user sentence."),
    ("human", "I love programming."),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)
