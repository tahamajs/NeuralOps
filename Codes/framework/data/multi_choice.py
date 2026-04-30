# framework/data/multi_choice.py
"""
Multi-choice dataset handling for tasks like ARC, HellaSwag, etc.
"""

import torch
import json
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset as HFDataset


class MultiChoiceDataset(Dataset):
    """
    Dataset for multi-choice questions (MCQ) like ARC, HellaSwag.
    
    Handles JSON-encoded choices and answers.
    
    Example:
        >>> dataset = MultiChoiceDataset(df, tokenizer)
        >>> batch = dataset[0]
        >>> print(batch['input_ids'].shape)  # (num_choices, seq_len)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 128,
        question_column: Optional[str] = None,
        choices_column: Optional[str] = None,
        answer_column: Optional[str] = None
    ):
        """
        Args:
            df: DataFrame with question, choices, and answer columns
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            question_column: Name of question column (auto-detected)
            choices_column: Name of choices column (auto-detected)
            answer_column: Name of answer column (auto-detected)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Auto-detect columns
        self.question_column = question_column or self._detect_question_column(df)
        self.choices_column = choices_column or self._detect_choices_column(df)
        self.answer_column = answer_column or self._detect_answer_column(df)
        
        valid_count = 0
        invalid_count = 0
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Extract question
            question = self._extract_question(row)
            
            # Extract choices
            choices = self._extract_choices(row)
            
            # Extract answer / label
            label = self._extract_label(row, len(choices))
            
            if len(choices) >= 2 and 0 <= label < len(choices):
                self.samples.append({
                    'question': question,
                    'choices': choices,
                    'label': label
                })
                valid_count += 1
            else:
                invalid_count += 1
        
        print(f"📊 Multi-choice dataset: {valid_count} valid, {invalid_count} filtered")
    
    def _detect_question_column(self, df: pd.DataFrame) -> str:
        """Auto-detect question column."""
        candidates = ['question', 'prompt', 'ctx', 'ctx_a', 'text', 'sentence']
        for col in candidates:
            if col in df.columns:
                return col
        return df.columns[0]
    
    def _detect_choices_column(self, df: pd.DataFrame) -> str:
        """Auto-detect choices column."""
        candidates = ['choices_json', 'choices', 'endings', 'options']
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _detect_answer_column(self, df: pd.DataFrame) -> str:
        """Auto-detect answer column."""
        candidates = ['gold_answer_json', 'answer_json', 'label', 'answer', 'gold']
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _extract_question(self, row) -> str:
        """Extract question text from row."""
        if self.question_column:
            return str(row[self.question_column])
        
        # Combine multiple context columns
        parts = []
        for col in ['ctx', 'ctx_a', 'ctx_b']:
            if col in row and pd.notna(row[col]):
                parts.append(str(row[col]))
        
        return " ".join(parts) if parts else ""
    
    def _extract_choices(self, row) -> List[str]:
        """Extract choice list from row."""
        if self.choices_column is None:
            # Look for choice columns like 'choice_a', 'choice_b', etc.
            choice_cols = [col for col in row.index if col.startswith('choice_')]
            if choice_cols:
                return [str(row[col]) for col in sorted(choice_cols)]
            return []
        
        choices_val = row[self.choices_column]
        
        if isinstance(choices_val, str):
            try:
                choices = json.loads(choices_val)
                if isinstance(choices, list):
                    return choices
            except json.JSONDecodeError:
                pass
        
        if isinstance(choices_val, list):
            return choices_val
        
        return []
    
    def _extract_label(self, row, num_choices: int) -> int:
        """Extract answer label from row."""
        if self.answer_column is None:
            if 'label' in row:
                return int(row['label'])
            return 0
        
        answer_val = row[self.answer_column]
        
        if isinstance(answer_val, str):
            try:
                answer_json = json.loads(answer_val)
                if 'choice_index' in answer_json:
                    return int(answer_json['choice_index'])
                if 'label' in answer_json:
                    return int(answer_json['label'])
            except json.JSONDecodeError:
                pass
        
        if isinstance(answer_val, (int, float)):
            return int(answer_val)
        
        return 0
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with all choices encoded."""
        sample = self.samples[idx]
        question = sample['question']
        choices = sample['choices']
        label = sample['label']
        
        encoded_choices = []
        for choice in choices:
            text = f"{question} {choice}" if question else choice
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            encoded_choices.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            })
        
        input_ids = torch.stack([ec['input_ids'] for ec in encoded_choices])
        attention_mask = torch.stack([ec['attention_mask'] for ec in encoded_choices])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def is_multichoice_dataset(df: pd.DataFrame) -> Tuple[bool, int]:
    """
    Detect if a dataset is multi-choice.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Tuple of (is_multichoice, num_choices)
    """
    # Check for choices_json column
    if 'choices_json' in df.columns:
        sample = df['choices_json'].iloc[0]
        if isinstance(sample, str):
            try:
                choices = json.loads(sample)
                if isinstance(choices, list) and len(choices) >= 2:
                    return True, len(choices)
            except json.JSONDecodeError:
                pass
    
    # Check for individual choice columns
    choice_cols = [col for col in df.columns if col.startswith('choice_')]
    if len(choice_cols) >= 2:
        return True, len(choice_cols)
    
    # Check for endings column (HellaSwag style)
    if 'endings' in df.columns:
        return True, 4  # HellaSwag typically has 4 endings
    
    return False, 0


def convert_to_multichoice_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a dataset to standard multi-choice format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with 'question', 'choices', 'label' columns
    """
    result_rows = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        # Extract question
        question = ""
        if 'prompt' in row:
            question = str(row['prompt'])
        elif 'ctx' in row:
            question = str(row['ctx'])
        
        # Extract choices
        choices = []
        if 'choices_json' in row:
            try:
                choices = json.loads(str(row['choices_json']))
            except:
                pass
        elif 'endings' in row:
            try:
                choices = json.loads(str(row['endings']))
            except:
                pass
        else:
            choice_cols = [col for col in df.columns if col.startswith('choice_')]
            choices = [str(row[col]) for col in sorted(choice_cols)]
        
        # Extract label
        label = 0
        if 'gold_answer_json' in row:
            try:
                ans = json.loads(str(row['gold_answer_json']))
                label = ans.get('choice_index', 0)
            except:
                pass
        elif 'answer_json' in row:
            try:
                ans = json.loads(str(row['answer_json']))
                label = ans.get('choice_index', 0)
            except:
                pass
        elif 'label' in row:
            label = int(row['label'])
        
        result_rows.append({
            'question': question,
            'choices': choices,
            'label': label
        })
    
    return pd.DataFrame(result_rows)