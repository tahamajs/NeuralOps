import os
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_DATASETS_OFFLINE'] = '0'
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading GPT-Neo 125M...')
AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m')
AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125m')
print('Download complete!')
