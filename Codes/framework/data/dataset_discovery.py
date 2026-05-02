"""Dataset discovery utilities. No external config dependency."""
import json, pandas as pd, torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Optional

class DatasetConfig:
    def __init__(self, name: str, path: str, task_type: str = 'single',
                 num_choices: int = 0, num_labels: int = 2):
        self.name = name
        self.path = path
        self.task_type = task_type
        self.num_choices = num_choices
        self.num_labels = num_labels

class ExperimentConfig:
    pass

def discover_datasets(config: Optional[ExperimentConfig] = None) -> List[DatasetConfig]:
    dataset_paths = {
        "data_custom": "./data/data_custom.csv",
        "arc_challenge": "./data/reasoning/arc_challenge/processed/arc_challenge_train.csv",
        "arc_agi": "./data/reasoning/arc_agi/processed/arc_agi_train.csv",
    }
    datasets = []
    for name, path in dataset_paths.items():
        if not Path(path).exists():
            print(f"❌ {name}: File not found at {path}")
            continue
        try:
            df_sample = pd.read_csv(path, nrows=5)
            is_multichoice = False
            num_choices = 0
            final_num_labels = 2
            final_task_type = 'single'
            has_choices_json = 'choices_json' in df_sample.columns
            has_gold_answer = 'gold_answer_json' in df_sample.columns or 'answer_json' in df_sample.columns
            if has_choices_json and has_gold_answer:
                first_val = df_sample['choices_json'].iloc[0]
                if isinstance(first_val, str):
                    try:
                        choices_list = json.loads(first_val)
                        if isinstance(choices_list, list) and len(choices_list) >= 2:
                            is_multichoice = True
                            num_choices = len(choices_list)
                            final_num_labels = num_choices
                            final_task_type = 'mc'
                            print(f"✅ Detected {name}: Multi-Choice ({num_choices} options from JSON)")
                    except json.JSONDecodeError:
                        continue
                elif isinstance(first_val, list) and len(first_val) >= 2:
                    is_multichoice = True
                    num_choices = len(first_val)
                    final_num_labels = num_choices
                    final_task_type = 'mc'
                    print(f"✅ Detected {name}: Multi-Choice ({num_choices} options directly)")
            elif 'choice_a' in df_sample.columns and 'answer_label' in df_sample.columns:
                choice_cols = [col for col in df_sample.columns if col.startswith('choice_')]
                if len(choice_cols) >= 2:
                    is_multichoice = True
                    num_choices = len(choice_cols)
                    final_num_labels = num_choices
                    final_task_type = 'mc'
                    print(f"✅ Detected {name}: Multi-Choice ({num_choices} options - ARC Style)")
            elif 'endings' in df_sample.columns:
                is_multichoice = True
                final_task_type = 'mc'
                print(f"✅ Detected {name}: Multi-Choice (Endings Column)")
            elif 'label' in df_sample.columns:
                final_task_type = 'single'
                print(f"✅ Detected {name}: Single Label")
            else:
                print(f"⚠️ {name}: No clear labels found.")
                continue
            datasets.append(DatasetConfig(
                name=name, path=path, task_type=final_task_type,
                num_choices=num_choices if is_multichoice else 0,
                num_labels=final_num_labels))
        except Exception as e:
            print(f"⚠️ Error analyzing {name}: {e}")
            continue
    return datasets

class MultiChoiceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        has_choices_json = 'choices_json' in df.columns
        has_endings = 'endings' in df.columns
        has_gold_answer = 'gold_answer_json' in df.columns or 'answer_json' in df.columns
        valid_count = 0
        invalid_count = 0
        for idx in range(len(df)):
            row = df.iloc[idx]
            question = ""
            choices = []
            label = 0
            if 'prompt' in row: question = str(row['prompt'])
            elif 'ctx' in row: question = str(row['ctx'])
            else:
                parts = [str(v) for v in row[['ctx', 'ctx_a', 'ctx_b']].dropna().values]
                question = " ".join(parts)
            if has_choices_json:
                try: choices_str = str(row['choices_json']); choices = json.loads(choices_str)
                except: pass
            elif has_endings:
                try: endings_str = str(row['endings']); choices = json.loads(endings_str)
                except: pass
            if has_gold_answer:
                if 'gold_answer_json' in row:
                    ans_json = json.loads(str(row['gold_answer_json']))
                    if 'choice_index' in ans_json: label = int(ans_json['choice_index'])
                elif 'answer_json' in row:
                    ans_json = json.loads(str(row['answer_json']))
                    if 'choice_index' in ans_json: label = int(ans_json['choice_index'])
            elif 'label' in row:
                label = int(row['label'])
            if len(choices) >= 2 and 0 <= label < len(choices):
                self.samples.append({'question': question, 'choices': choices, 'label': label})
                valid_count += 1
            else:
                invalid_count += 1
        print(f"📊 Dataset Filtered ({len(df)} total): {valid_count} valid kept, {invalid_count} filtered.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        choices = sample['choices']
        label = sample['label']
        encoded_choices = []
        for choice in choices:
            text = f"{sample['question']} {choice}"
            enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
            encoded_choices.append({
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)
            })
        input_ids = torch.stack([ec['input_ids'] for ec in encoded_choices])
        attention_mask = torch.stack([ec['attention_mask'] for ec in encoded_choices])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(label, dtype=torch.long)}

class SingleLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            text = str(row.get('question', row.get('prompt', '')))
            label = 0
            self.texts.append(text)
            self.labels.append(label)
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
