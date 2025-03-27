from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    # model="mistral",
    model="llama3.2",
    temperature=0.1
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

result = chain.invoke(
    {
        "input_language": "English",
        "output_language": "Portuguese",
        "input": "I love programming."
    }
)

print(result.content)
