"""Benchmark dataset metadata and registry.

Defines the BenchmarkDatasetInfo data structure and BenchmarkDatasetRegistry
maintaining metadata, aliases, formatting hooks, and synthetic fallbacks for 12+ standard benchmarks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from neuralops.data.benchmark_datasets.synthetic import (
    _generate_synthetic_arc,
    _generate_synthetic_c4,
    _generate_synthetic_glue,
    _generate_synthetic_gsm8k,
    _generate_synthetic_hellaswag,
    _generate_synthetic_humaneval,
    _generate_synthetic_lambada,
    _generate_synthetic_mmlu,
    _generate_synthetic_pg19,
    _generate_synthetic_pile,
    _generate_synthetic_ptb,
    _generate_synthetic_wikitext,
)


@dataclass
class BenchmarkDatasetInfo:
    """Metadata describing a benchmark dataset and its default loading strategy."""

    name: str
    aliases: list[str]
    category: str
    default_hf_path: str
    default_subset: str | None = None
    default_split: str = "validation"
    text_formatter: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    synthetic_generator: Callable[[int, int], list[dict[str, Any]]] | None = None
    description: str = ""


class BenchmarkDatasetRegistry:
    """Registry maintaining metadata and formatters for NLP/LLM benchmarks."""

    _registry: dict[str, BenchmarkDatasetInfo] = {}

    @classmethod
    def register(cls, info: BenchmarkDatasetInfo) -> None:
        """Register a new benchmark dataset info."""
        key = info.name.lower()
        cls._registry[key] = info
        for alias in info.aliases:
            cls._registry[alias.lower()] = info

    @classmethod
    def get(cls, name: str) -> BenchmarkDatasetInfo | None:
        """Retrieve benchmark dataset info by name or alias."""
        return cls._registry.get(
            name.lower().replace("-", "").replace("_", "")
        ) or cls._registry.get(name.lower())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a benchmark dataset name is registered."""
        return cls.get(name) is not None

    @classmethod
    def list_benchmarks(cls) -> list[str]:
        """List canonical names of all registered benchmarks."""
        canonical = set()
        for info in cls._registry.values():
            canonical.add(info.name)
        return sorted(list(canonical))

    @classmethod
    def list_datasets(cls) -> list[str]:
        """Alias for list_benchmarks."""
        return cls.list_benchmarks()


def _register_all_benchmarks() -> None:
    """Register default set of 12+ standard benchmarks."""
    # 1. WikiText-2 & WikiText-103
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="wikitext2",
            aliases=["wikitext-2", "wikitext_2", "wikitext-2-raw-v1", "wikitext2rawv1"],
            category="language_modeling",
            default_hf_path="Salesforce/wikitext",
            default_subset="wikitext-2-raw-v1",
            default_split="validation",
            text_formatter=lambda r: {"text": r.get("text", "")},
            synthetic_generator=_generate_synthetic_wikitext,
            description="WikiText-2 raw language modeling benchmark.",
        )
    )

    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="wikitext103",
            aliases=[
                "wikitext-103",
                "wikitext_103",
                "wikitext-103-raw-v1",
                "wikitext103rawv1",
            ],
            category="language_modeling",
            default_hf_path="Salesforce/wikitext",
            default_subset="wikitext-103-raw-v1",
            default_split="validation",
            text_formatter=lambda r: {"text": r.get("text", "")},
            synthetic_generator=_generate_synthetic_wikitext,
            description="WikiText-103 raw language modeling benchmark.",
        )
    )

    # 2. C4 (Colossal Clean Crawled Corpus)
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="c4",
            aliases=["c4_en", "c4-en", "allenai/c4", "allenaic4"],
            category="language_modeling",
            default_hf_path="allenai/c4",
            default_subset="en",
            default_split="validation",
            text_formatter=lambda r: {"text": r.get("text", "")},
            synthetic_generator=_generate_synthetic_c4,
            description="C4 (Colossal Clean Crawled Corpus) English split.",
        )
    )

    # 3. Penn Treebank (PTB)
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="ptb",
            aliases=["penn_treebank", "ptb_text_only", "ptbtextonly"],
            category="language_modeling",
            default_hf_path="ptb_text_only",
            default_subset=None,
            default_split="validation",
            text_formatter=lambda r: {"text": r.get("sentence", r.get("text", ""))},
            synthetic_generator=_generate_synthetic_ptb,
            description="Penn Treebank language modeling dataset.",
        )
    )

    # 4. PG-19 (Project Gutenberg 1919)
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="pg19",
            aliases=["pg-19", "deepmind/pg19", "deepmindpg19"],
            category="language_modeling",
            default_hf_path="pg19",
            default_subset=None,
            default_split="validation",
            text_formatter=lambda r: {"text": r.get("text", "")},
            synthetic_generator=_generate_synthetic_pg19,
            description="PG-19 long-context book language modeling benchmark.",
        )
    )

    # 5. GSM8K (Grade School Math 8K)
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="gsm8k",
            aliases=["gsm-8k", "gsm_8k", "openai/gsm8k"],
            category="reasoning",
            default_hf_path="gsm8k",
            default_subset="main",
            default_split="test",
            text_formatter=lambda r: {
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "text": f"Question: {r.get('question', '')}\nAnswer: {r.get('answer', '')}",
            },
            synthetic_generator=_generate_synthetic_gsm8k,
            description="GSM8K Grade School Math reasoning benchmark.",
        )
    )

    # 6. HumanEval
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="humaneval",
            aliases=["human_eval", "openai_humaneval", "openaihumaneval"],
            category="code",
            default_hf_path="openai_humaneval",
            default_subset=None,
            default_split="test",
            text_formatter=lambda r: {
                "task_id": r.get("task_id", ""),
                "prompt": r.get("prompt", ""),
                "canonical_solution": r.get("canonical_solution", ""),
                "test": r.get("test", ""),
                "entry_point": r.get("entry_point", ""),
                "text": f"{r.get('prompt', '')}{r.get('canonical_solution', '')}",
            },
            synthetic_generator=_generate_synthetic_humaneval,
            description="HumanEval code generation benchmark.",
        )
    )

    # 7. MMLU (Massive Multitask Language Understanding)
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="mmlu",
            aliases=["cais/mmlu", "caismmlu", "hendrycks_test"],
            category="multi_choice",
            default_hf_path="cais/mmlu",
            default_subset="all",
            default_split="test",
            text_formatter=lambda r: {
                "question": r.get("question", ""),
                "choices": r.get("choices", []),
                "answer": r.get("answer", 0),
                "subject": r.get("subject", ""),
                "text": f"Question: {r.get('question', '')}\nChoices: {r.get('choices', [])}\nAnswer: {r.get('answer', '')}",
            },
            synthetic_generator=_generate_synthetic_mmlu,
            description="MMLU multi-subject reasoning benchmark.",
        )
    )

    # 8. The Pile / Pile Val
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="pile",
            aliases=[
                "pile_val",
                "pileval",
                "the_pile",
                "thepile",
                "monology/pile-uncopyrighted",
            ],
            category="language_modeling",
            default_hf_path="monology/pile-uncopyrighted",
            default_subset=None,
            default_split="train",
            text_formatter=lambda r: {"text": r.get("text", "")},
            synthetic_generator=_generate_synthetic_pile,
            description="The Pile diverse pretraining & evaluation corpus.",
        )
    )

    # 9. ARC (AI2 Reasoning Challenge: Easy & Challenge)
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="arc",
            aliases=["arc_easy", "arceasy", "ai2_arc", "ai2arc"],
            category="multi_choice",
            default_hf_path="ai2_arc",
            default_subset="ARC-Easy",
            default_split="test",
            text_formatter=lambda r: {
                "id": r.get("id", ""),
                "question": r.get("question", ""),
                "choices": r.get("choices", {}),
                "answerKey": r.get("answerKey", ""),
                "text": f"Question: {r.get('question', '')}\nAnswer: {r.get('answerKey', '')}",
            },
            synthetic_generator=_generate_synthetic_arc,
            description="AI2 Reasoning Challenge (ARC-Easy).",
        )
    )

    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="arc_challenge",
            aliases=["arc-challenge", "arcchallenge"],
            category="multi_choice",
            default_hf_path="ai2_arc",
            default_subset="ARC-Challenge",
            default_split="test",
            text_formatter=lambda r: {
                "id": r.get("id", ""),
                "question": r.get("question", ""),
                "choices": r.get("choices", {}),
                "answerKey": r.get("answerKey", ""),
                "text": f"Question: {r.get('question', '')}\nAnswer: {r.get('answerKey', '')}",
            },
            synthetic_generator=_generate_synthetic_arc,
            description="AI2 Reasoning Challenge (ARC-Challenge).",
        )
    )

    # 10. HellaSwag
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="hellaswag",
            aliases=["hella_swag", "rowan/hellaswag"],
            category="multi_choice",
            default_hf_path="hellaswag",
            default_subset=None,
            default_split="validation",
            text_formatter=lambda r: {
                "ctx": r.get("ctx", ""),
                "endings": r.get("endings", []),
                "label": r.get("label", 0),
                "text": f"{r.get('ctx', '')} {r.get('endings', [''])[int(r.get('label', 0)) if str(r.get('label', '0')).isdigit() else 0]}",
            },
            synthetic_generator=_generate_synthetic_hellaswag,
            description="HellaSwag commonsense reasoning benchmark.",
        )
    )

    # 11. LAMBADA
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="lambada",
            aliases=["lambada_openai", "eleutherai/lambada_openai"],
            category="language_modeling",
            default_hf_path="lambada",
            default_subset=None,
            default_split="test",
            text_formatter=lambda r: {"text": r.get("text", "")},
            synthetic_generator=_generate_synthetic_lambada,
            description="LAMBADA word prediction and narrative understanding benchmark.",
        )
    )

    # 12. GLUE Benchmarks (all 9 tasks)
    glue_tasks = ["sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli", "cola", "stsb"]
    for task in glue_tasks:
        BenchmarkDatasetRegistry.register(
            BenchmarkDatasetInfo(
                name=f"glue_{task}",
                aliases=[f"glue/{task}", f"glue-{task}", task],
                category="glue",
                default_hf_path="glue",
                default_subset=task,
                default_split="validation" if task != "mnli" else "validation_matched",
                text_formatter=lambda r, t=task: {
                    "text": r.get("sentence", r.get("sentence1", r.get("text", ""))),
                    "label": r.get("label", 0),
                },
                synthetic_generator=lambda n, s=42, t=task: _generate_synthetic_glue(t, n, s),
                description=f"GLUE benchmark task: {task.upper()}.",
            )
        )


_register_all_benchmarks()

__all__ = [
    "BenchmarkDatasetInfo",
    "BenchmarkDatasetRegistry",
]
