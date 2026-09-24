from huggingface_hub import HfApi
api = HfApi()
repos = list(api.list_models(author='Hooshaai'))
print("Remaining GPT2 repos:")
for r in repos:
    if 'gpt2' in r.modelId.lower():
        print(r.modelId)
