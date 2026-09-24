from huggingface_hub import HfApi

api = HfApi()
repos = list(api.list_models(author='Hooshaai'))

deleted = []
for r in repos:
    name = r.modelId.lower()
    # Delete anything that contains wikitext
    if 'wikitext' in name:
        try:
            api.delete_repo(r.modelId, repo_type='model')
            deleted.append(r.modelId)
        except Exception as e:
            print(e)
            
print(f"Deleted remaining wikitext repos: {deleted}")
