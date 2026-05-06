def upload(local_path, **kwargs):
    pass   # already on disk
def download(local_path, **kwargs):
    pass   # assumed to exist
def delete(local_path, **kwargs):
    import os, shutil
    if os.path.exists(local_path):
        shutil.rmtree(local_path, ignore_errors=True)