import tiktoken

texto = "A inteligência artificial está mudando o mundo."

tokenizer = tiktoken.get_encoding("cl100k_base")  # compatível com GPT-3.5/4
tokens = tokenizer.encode(texto)

print(f"Texto original: {texto}")
print(f"Número de tokens: {len(tokens)}")
print(f"Tokens: {tokens}")
