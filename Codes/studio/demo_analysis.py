# demo_analysis.py
from studio.visualization.dashboard import AttentionDashboard
from studio.training.methods import get_trainer_for_method
from studio.evaluation.benchmark import run_full_benchmark
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load a model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 1. Interactive attention dashboard
dashboard = AttentionDashboard(model, tokenizer)
dashboard.analyze_text("The cat sat on the mat.", layer=5, head=3, output_html="my_attention.html")

# 2. Run a full benchmark (quick mode)
df = run_full_benchmark(epochs=1, batch_size=4)
df.to_csv("benchmark_results.csv", index=False)
print(df)

# 3. Train with a specific method
train_ds, val_ds = load_dataset_from_config({"name":"cola","type":"glue"}, tokenizer)
trainer = get_trainer_for_method("lora+rdrop", model, tokenizer, train_ds, val_ds, "./output")
trainer.train()