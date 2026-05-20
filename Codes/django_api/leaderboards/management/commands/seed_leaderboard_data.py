from django.core.management.base import BaseCommand
from leaderboards.models import BenchmarkTask, BenchmarkDataset, Metric

class Command(BaseCommand):
    help = 'Seed common NLP benchmark tasks, datasets, and metrics'

    def handle(self, *args, **options):
        # Tasks
        tasks = [
            ('sentiment-analysis', 'Sentiment Analysis', 'NLP'),
            ('text-classification', 'Text Classification', 'NLP'),
            ('named-entity-recognition', 'Named Entity Recognition', 'NLP'),
            ('question-answering', 'Question Answering', 'NLP'),
            ('summarization', 'Summarization', 'NLP'),
            ('machine-translation', 'Machine Translation', 'NLP'),
            ('natural-language-inference', 'Natural Language Inference', 'NLP'),
            ('paraphrase-detection', 'Paraphrase Detection', 'NLP'),
            ('semantic-similarity', 'Semantic Textual Similarity', 'NLP'),
        ]
        for slug, name, cat in tasks:
            BenchmarkTask.objects.get_or_create(slug=slug, defaults={'name': name, 'category': cat})

        # Datasets
        datasets = [
            ('sst2', 'SST-2', 'sentiment-analysis'),
            ('mnli', 'MNLI', 'natural-language-inference'),
            ('qnli', 'QNLI', 'question-answering'),
            ('mrpc', 'MRPC', 'paraphrase-detection'),
            ('stsb', 'STS-B', 'semantic-similarity'),
            ('cola', 'CoLA', 'text-classification'),
            ('rte', 'RTE', 'natural-language-inference'),
            ('wnli', 'WNLI', 'natural-language-inference'),
            ('squad', 'SQuAD v2', 'question-answering'),
            ('cnn-dailymail', 'CNN/DailyMail', 'summarization'),
        ]
        for slug, name, task_slug in datasets:
            task = BenchmarkTask.objects.get(slug=task_slug)
            BenchmarkDataset.objects.get_or_create(slug=slug, defaults={'name': name, 'task': task})

        # Metrics
        metrics = [
            ('accuracy', 'Accuracy', 'maximize'),
            ('f1', 'F1 Score', 'maximize'),
            ('precision', 'Precision', 'maximize'),
            ('recall', 'Recall', 'maximize'),
            ('mcc', 'Matthews Correlation', 'maximize'),
            ('pearson', 'Pearson Correlation', 'maximize'),
            ('spearman', 'Spearman Correlation', 'maximize'),
            ('loss', 'Loss', 'minimize'),
            ('perplexity', 'Perplexity', 'minimize'),
            ('bleu', 'BLEU', 'maximize'),
            ('rouge-l', 'ROUGE-L', 'maximize'),
        ]
        for slug, name, direction in metrics:
            Metric.objects.get_or_create(slug=slug, defaults={'name': name, 'direction': direction})

        self.stdout.write(self.style.SUCCESS(f'Seeded {len(tasks)} tasks, {len(datasets)} datasets, {len(metrics)} metrics'))