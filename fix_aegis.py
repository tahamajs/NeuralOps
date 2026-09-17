import re
with open("Codes/framework/attentions/aegis_hybrid.py", "r") as f:
    text = f.read()

replacement = """\\1_decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
\\1if any(model_type.startswith(p) for p in _decoder_prefixes):"""

text = re.sub(r'^([ \t]+)if model_type in \["gpt2", "gpt", "qwen", "qwen2", "qwen2.5", "llama", "mistral"\]:', replacement, text, flags=re.MULTILINE)

with open("Codes/framework/attentions/aegis_hybrid.py", "w") as f:
    f.write(text)
