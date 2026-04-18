import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparison:
    def __init__(self, results_dirs):
        self.results_dirs = results_dirs
        self.dataframes = []

    def load_all_results(self):
        for d in self.results_dirs:
            csv_path = os.path.join(d, 'performance.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['run'] = os.path.basename(d)
                self.dataframes.append(df)
        if self.dataframes:
            self.df_all = pd.concat(self.dataframes, ignore_index=True)

    def plot_accuracy_vs_rank(self, task='cola'):
        if not hasattr(self, 'df_all'):
            self.load_all_results()
        df_task = self.df_all[self.df_all['task'] == task]
        plt.figure(figsize=(12,6))
        sns.lineplot(data=df_task, x='rank', y='accuracy', hue='model', marker='o')
        plt.title(f'Accuracy vs Rank for {task.upper()}')
        plt.savefig(f'comparison_{task}.png')
        plt.close()

    def generate_summary_table(self):
        if not hasattr(self, 'df_all'):
            self.load_all_results()
        pivot = self.df_all.pivot_table(index=['model', 'rank'], columns='task', values='accuracy')
        pivot.to_csv('summary_table.csv')
        return pivot