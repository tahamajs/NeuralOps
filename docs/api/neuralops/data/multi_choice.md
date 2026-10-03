Module neuralops.data.multi_choice
==================================
Multi‑choice dataset (ARC, HellaSwag, etc.) handling.

Functions
---------

`convert_to_multichoice_format(df: pandas.DataFrame) ‑> pandas.DataFrame`
:   Normalise a DataFrame into the standard multi‑choice format (question, choices, label).

`is_multichoice_dataset(df: pandas.DataFrame) ‑> tuple[bool, int]`
:   Check if a DataFrame contains a multi‑choice format and return the number of choices.

Classes
-------

`MultiChoiceDataset(df: pandas.DataFrame, tokenizer, max_length: int = 128, question_column: str | None = None, choices_column: str | None = None, answer_column: str | None = None)`
:   Dataset for multi‑choice questions (ARC, HellaSwag, etc.).

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic