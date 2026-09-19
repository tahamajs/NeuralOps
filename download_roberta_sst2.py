from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "textattack/roberta-base-SST-2"
print("Downloading / checking", model_name)
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
save_dir = r"D:\svd_project\models\roberta_sst2"
tok.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print("Saved successfully to", save_dir)
