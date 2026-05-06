import os, hashlib, tarfile, shutil, logging
logger = logging.getLogger(__name__)

def compute_checksum(file_path: str, algo='sha256') -> str:
    h = hashlib.new(algo)
    with open(file_path,'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''): h.update(chunk)
    return h.hexdigest()

def compress_directory(dir_path: str, output_file: str = None) -> str:
    if output_file is None: output_file = dir_path + '.tar.gz'
    with tarfile.open(output_file, 'w:gz') as tar:
        tar.add(dir_path, arcname=os.path.basename(dir_path))
    return output_file

def extract_archive(archive_path: str, dest_dir: str) -> str:
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(dest_dir)
    # return the directory inside
    return dest_dir