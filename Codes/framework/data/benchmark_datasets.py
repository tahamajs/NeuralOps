"""
Unified Benchmark Dataset Loader & Registry for NLP / LLM Evaluation.
Supports 12+ standard NLP/LLM benchmarks with streaming, local caching,
tokenized PyTorch DataLoader creation, and offline synthetic fallbacks.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Models & Benchmark Information
# -----------------------------------------------------------------------------


@dataclass
class BenchmarkDatasetInfo:
    """Metadata describing a benchmark dataset and its default loading strategy."""

    name: str
    aliases: list[str]
    category: (
        str  # e.g., "language_modeling", "reasoning", "code", "multi_choice", "glue"
    )
    default_hf_path: str
    default_subset: str | None = None
    default_split: str = "validation"
    text_formatter: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    synthetic_generator: Callable[[int, int], list[dict[str, Any]]] | None = None
    description: str = ""


# -----------------------------------------------------------------------------
# Synthetic Data Generators (Offline Fallback)
# -----------------------------------------------------------------------------


def _generate_synthetic_wikitext(
    num_samples: int, seed: int = 42
) -> list[dict[str, Any]]:
    random.Random(seed)
    topics = [
        (
            "Valkyria Chronicles III",
            "is a tactical role-playing video game developed by Sega and Media.Vision for the PlayStation Portable.",
        ),
        (
            "Tower of London",
            "is a historic castle on the north bank of the River Thames in central London.",
        ),
        (
            "Supercell (thunderstorm)",
            "is a thunderstorm characterized by the presence of a mesocyclone: a deep, persistently rotating updraft.",
        ),
        (
            "Quantum chromodynamics",
            "is the theory of the strong interaction between quarks and gluons, the fundamental particles that make up hadrons.",
        ),
        (
            "Battle of Austerlitz",
            "also known as the Battle of the Three Emperors, was one of the most important and decisive engagements of the Napoleonic Wars.",
        ),
    ]
    samples = []
    for i in range(num_samples):
        title, lead = topics[i % len(topics)]
        text = f" = {title} = \n\n {lead} In year {1900 + (i * 7) % 120}, significant developments took place regarding this subject. Furthermore, numerous scholarly investigations and experimental analyses have demonstrated the enduring significance of {title.lower()} within modern scientific and cultural discourse."
        samples.append({"text": text, "id": f"wikitext_synth_{i}"})
    return samples


def _generate_synthetic_c4(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    random.Random(seed)
    domains = [
        "news-digest.org",
        "tech-insights.io",
        "culinary-arts.net",
        "travel-chronicles.com",
    ]
    samples = []
    for i in range(num_samples):
        dom = domains[i % len(domains)]
        text = (
            f"Published on {dom}: The rapid progression of contemporary computational methodologies has fundamentally transformed industry practices. "
            f"Engineers and researchers in sample #{i} note that optimizing memory efficiency and attention mechanisms allows scalable throughput across distributed clusters. "
            f"Read more at https://{dom}/article/{1000 + i} for in-depth evaluations and reproducible metrics."
        )
        samples.append(
            {
                "text": text,
                "url": f"https://{dom}/article/{1000 + i}",
                "timestamp": "2024-01-01T12:00:00Z",
            }
        )
    return samples


def _generate_synthetic_ptb(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    templates = [
        "no it was n't black monday <unk> but the stock market did take a sharp dive yesterday",
        "the company said sales rose N % to $ N million from $ N million in the prior year",
        "treasury bonds fell sharply as investors reacted to new economic indicators",
        "yields on money market mutual funds fell again in the latest week <unk>",
        "a spokesman for the department declined to comment on pending litigation",
    ]
    samples = []
    for i in range(num_samples):
        base = templates[i % len(templates)]
        samples.append({"text": f"{base} with index {i} <unk> .", "id": i})
    return samples


def _generate_synthetic_pg19(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    samples = []
    for i in range(num_samples):
        chapters = []
        for ch in range(1, 4):
            chapters.append(
                f"CHAPTER {ch}.\n\nThe morning light broke over the distant horizon, casting long shadows across the valley. "
                f"Protagonist {i} embarked on the long journey towards the old library, contemplating the secrets of natural philosophy. "
                f"Every page turned in the ancient manuscripts revealed forgotten theorems and profound observations on mathematics and metaphysics."
            )
        text = "\n\n".join(chapters)
        samples.append(
            {
                "text": text,
                "book_id": f"pg19_book_{i}",
                "short_book_title": f"The Chronicles of Explorer {i}",
            }
        )
    return samples


def _generate_synthetic_gsm8k(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    samples = []
    for i in range(num_samples):
        apples_per_basket = 5 + (i % 6)
        num_baskets = 3 + (i % 4)
        eaten = 2 + (i % 3)
        total_apples = apples_per_basket * num_baskets
        remaining = total_apples - eaten
        q = f"Janet has {num_baskets} baskets of apples. Each basket contains {apples_per_basket} apples. If she eats {eaten} apples, how many apples does she have left?"
        a = f"First, calculate the total number of apples: {num_baskets} * {apples_per_basket} = {total_apples}. Next, subtract the eaten apples: {total_apples} - {eaten} = {remaining}.\n#### {remaining}"
        samples.append(
            {"question": q, "answer": a, "text": f"Question: {q}\nAnswer: {a}"}
        )
    return samples


def _generate_synthetic_humaneval(
    num_samples: int, seed: int = 42
) -> list[dict[str, Any]]:
    samples = []
    functions = [
        (
            "has_close_elements",
            "numbers: List[float], threshold: float -> bool",
            "Check if in given list of numbers, are any two numbers closer to each other than threshold.",
        ),
        (
            "separate_paren_groups",
            "paren_string: str -> List[str]",
            "Input to this function is a string containing multiple groups of nested parentheses. Separate them into individual strings.",
        ),
        (
            "truncate_number",
            "number: float -> float",
            "Given a positive floating point number, it can be decomposed into integer part and decimals. Return the decimal part.",
        ),
        (
            "below_zero",
            "operations: List[int] -> bool",
            "You're given a list of deposit and withdrawal operations on a bank account. Detect if balance falls below zero at any point.",
        ),
        (
            "mean_absolute_deviation",
            "numbers: List[float] -> float",
            "For a given list of input numbers, calculate Mean Absolute Deviation around the mean of this dataset.",
        ),
    ]
    for i in range(num_samples):
        fn_name, sig, doc = functions[i % len(functions)]
        fn_name_i = f"{fn_name}_{i}"
        prompt = f'from typing import List\n\n\ndef {fn_name_i}({sig}):\n    """ {doc}\n    >>> {fn_name_i}([1.0, 2.0], 0.5)\n    """\n'
        canonical_solution = "    return True\n"
        test = (
            "def check(candidate):\n    assert candidate([1.0, 2.0], 0.5) is not None\n"
        )
        entry_point = fn_name_i
        samples.append(
            {
                "task_id": f"HumanEval/{i}",
                "prompt": prompt,
                "canonical_solution": canonical_solution,
                "test": test,
                "entry_point": entry_point,
                "text": f"{prompt}{canonical_solution}",
            }
        )
    return samples


def _generate_synthetic_mmlu(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    subjects = [
        "college_computer_science",
        "abstract_algebra",
        "high_school_physics",
        "econometrics",
        "philosophy",
    ]
    samples = []
    for i in range(num_samples):
        subject = subjects[i % len(subjects)]
        q = f"In the context of {subject.replace('_', ' ')}, which statement regarding fundamental principle #{i+1} is strictly valid?"
        choices = [
            f"Option A: Principle {i+1} guarantees monotonic convergence under convexity constraints.",
            f"Option B: Principle {i+1} always yields an asymptotic upper bound of O(N log N).",
            f"Option C: Principle {i+1} implies invariance under orthonormal coordinate rotations.",
            f"Option D: Principle {i+1} requires strict thermodynamic equilibrium.",
        ]
        label = i % 4
        formatted_text = f"Subject: {subject}\nQuestion: {q}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {chr(65 + label)}"
        samples.append(
            {
                "question": q,
                "choices": choices,
                "answer": label,
                "subject": subject,
                "text": formatted_text,
            }
        )
    return samples


def _generate_synthetic_pile(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    subsets = [
        "ArXiv",
        "GitHub",
        "FreeLaw",
        "PubMed Central",
        "StackExchange",
        "Wikipedia (en)",
    ]
    samples = []
    for i in range(num_samples):
        subset = subsets[i % len(subsets)]
        text = (
            f"[{subset} Document ID: {200000 + i}]\n"
            f"Abstract / Content: This document addresses standard analytical methodologies in {subset.lower()}. "
            f"Through rigorous verification of Lemma {i % 10 + 1}, we establish uniform bounds across all operational domains. "
            f"Reference code and benchmarks verify empirical speedup of 4.2x with zero precision degradation."
        )
        samples.append({"text": text, "meta": {"pile_set_name": subset}, "id": i})
    return samples


def _generate_synthetic_arc(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    samples = []
    for i in range(num_samples):
        q = f"Which process is primarily responsible for transferring thermal energy through fluid movement in system #{i+1}?"
        choices_text = ["Conduction", "Convection", "Radiation", "Sublimation"]
        choices_labels = ["A", "B", "C", "D"]
        answer_key = "B"
        label_idx = 1
        samples.append(
            {
                "id": f"ARC_Synth_{i}",
                "question": q,
                "choices": {"text": choices_text, "label": choices_labels},
                "answerKey": answer_key,
                "label": label_idx,
                "text": f"Question: {q}\nChoices: (A) Conduction (B) Convection (C) Radiation (D) Sublimation\nAnswer: {answer_key}",
            }
        )
    return samples


def _generate_synthetic_hellaswag(
    num_samples: int, seed: int = 42
) -> list[dict[str, Any]]:
    samples = []
    for i in range(num_samples):
        ctx = "A person is preparing ingredients in a bright kitchen. They slice fresh vegetables on a wooden cutting board"
        endings = [
            "and carefully place them into a preheated skillet with olive oil.",
            "and suddenly ride a bicycle across the highway without stopping.",
            "and throw the wooden board into the washing machine.",
            "and paint the vegetables blue using watercolor brushes.",
        ]
        label = 0
        samples.append(
            {
                "activity_label": "Cooking",
                "ctx": ctx,
                "endings": endings,
                "label": label,
                "text": f"{ctx} {endings[label]}",
            }
        )
    return samples


def _generate_synthetic_lambada(
    num_samples: int, seed: int = 42
) -> list[dict[str, Any]]:
    samples = []
    stories = [
        (
            "The young girl ran quickly across the open grass towards the tall oak tree, looking for her lost",
            "cat",
        ),
        (
            "He opened the dusty book and carefully read the handwritten note inscribed on the very first",
            "page",
        ),
        (
            "After hours of walking through the dense forest, the explorers finally arrived at the edge of the",
            "river",
        ),
        (
            "The chef tasted the simmering soup, nodded with satisfaction, and added a small pinch of",
            "salt",
        ),
        (
            "The astronomer adjusted the lens of the giant telescope and peered into the dark night",
            "sky",
        ),
    ]
    for i in range(num_samples):
        prefix, target_word = stories[i % len(stories)]
        full_text = f"{prefix} {target_word}."
        samples.append(
            {
                "text": full_text,
                "prefix": prefix,
                "target_word": target_word,
                "id": i,
            }
        )
    return samples


def _generate_synthetic_glue(
    task: str, num_samples: int, seed: int = 42
) -> list[dict[str, Any]]:
    task = task.lower()
    samples = []
    if task == "sst2":
        for i in range(num_samples):
            sentiment = i % 2
            text = f"This cinematic piece delivers {'an extraordinarily compelling and nuanced masterpiece' if sentiment == 1 else 'a completely tedious and utterly forgettable disaster'}."
            samples.append({"sentence": text, "label": sentiment, "text": text})
    elif task == "mrpc":
        for i in range(num_samples):
            equiv = i % 2
            s1 = f"The technology corporation announced a profit increase of {10 + i}% this quarter."
            s2 = (
                f"Profits for the tech firm grew by {10 + i}% during the current quarter."
                if equiv == 1
                else "The firm announced massive layoffs across international divisions."
            )
            samples.append(
                {
                    "sentence1": s1,
                    "sentence2": s2,
                    "label": equiv,
                    "text": f"{s1} [SEP] {s2}",
                }
            )
    elif task == "qqp":
        for i in range(num_samples):
            dup = i % 2
            q1 = f"How can one efficiently optimize memory usage in neural networks for task {i}?"
            q2 = (
                f"What are the best methods to reduce GPU memory footprint during neural net training {i}?"
                if dup == 1
                else f"What is the best recipe for baking sourdough bread at home {i}?"
            )
            samples.append(
                {
                    "question1": q1,
                    "question2": q2,
                    "label": dup,
                    "text": f"{q1} [SEP] {q2}",
                }
            )
    elif task == "stsb":
        for i in range(num_samples):
            score = round((i % 50) / 10.0, 1)
            s1 = f"A brown dog is running through the green grassy field {i}."
            s2 = (
                f"A canine is sprinting across the lawn {i}."
                if score > 3.0
                else f"A man is playing guitar on a concert stage {i}."
            )
            samples.append(
                {
                    "sentence1": s1,
                    "sentence2": s2,
                    "label": score,
                    "text": f"{s1} [SEP] {s2}",
                }
            )
    elif task in ("mnli", "qnli", "rte", "wnli"):
        for i in range(num_samples):
            label = i % 2
            premise = f"Scientific observation {i} confirmed that gravitational waves propagate at the speed of light."
            hypothesis = (
                f"Gravitational radiation travels at light speed in observation {i}."
                if label == 0
                else f"Light travels infinitely faster than gravitational waves in experiment {i}."
            )
            samples.append(
                {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "sentence1": premise,
                    "sentence2": hypothesis,
                    "label": label,
                    "text": f"{premise} [SEP] {hypothesis}",
                }
            )
    elif task == "cola":
        for i in range(num_samples):
            acceptable = i % 2
            sentence = (
                "The researcher submitted the journal paper yesterday."
                if acceptable == 1
                else "Submitted the researcher paper yesterday journal."
            )
            samples.append(
                {"sentence": sentence, "label": acceptable, "text": sentence}
            )
    else:
        for i in range(num_samples):
            samples.append(
                {
                    "sentence": f"Synthetic GLUE sample #{i} for task {task}.",
                    "label": i % 2,
                    "text": f"Synthetic GLUE sample #{i}.",
                }
            )
    return samples


# -----------------------------------------------------------------------------
# Benchmark Dataset Registry
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Default Benchmarks Registration
# -----------------------------------------------------------------------------


def _register_all_benchmarks():
    # 1. WikiText-2 & WikiText-103
    BenchmarkDatasetRegistry.register(
        BenchmarkDatasetInfo(
            name="wikitext2",
            aliases=["wikitext-2", "wikitext_2", "wikitext-2-raw-v1", "wikitext2rawv1"],
            category="language_modeling",
            default_hf_path="wikitext",
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
            default_hf_path="wikitext",
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
                synthetic_generator=lambda n, s=42, t=task: _generate_synthetic_glue(
                    t, n, s
                ),
                description=f"GLUE benchmark task: {task.upper()}.",
            )
        )


# Initialize registry on import
_register_all_benchmarks()


# -----------------------------------------------------------------------------
# PyTorch Dataset Implementations
# -----------------------------------------------------------------------------


class BenchmarkTorchDataset(Dataset):
    """PyTorch Dataset wrapper for processed benchmark records."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Any = None,
        seq_len: int = 2048,
        is_causal_lm: bool = True,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.is_causal_lm = is_causal_lm

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        if self.tokenizer is None:
            return item

        # Extract primary text string
        text = item.get("text", "")
        if not text:
            for k in ["prompt", "sentence", "question", "sentence1"]:
                if k in item and item[k]:
                    text = str(item[k])
                    break

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Handle causal LM labels or classification labels
        if "label" in item:
            data["labels"] = torch.tensor(
                item["label"],
                dtype=torch.long if isinstance(item["label"], int) else torch.float,
            )
        elif self.is_causal_lm:
            # Shifted causal LM targets (replace padding token ids with -100 in labels)
            labels = input_ids.clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            data["labels"] = labels

        return data


def _collate_benchmark_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates a list of sample dicts into batch tensors or lists."""
    if not batch:
        return {}

    first = batch[0]
    out: dict[str, Any] = {}

    for key in first.keys():
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], (int, float)):
            out[key] = torch.tensor(vals)
        else:
            out[key] = vals

    return out


# -----------------------------------------------------------------------------
# Main Unified Loader API
# -----------------------------------------------------------------------------


def load_benchmark_dataset(
    dataset_name: str,
    tokenizer: Any = None,
    seq_len: int = 2048,
    batch_size: int = 8,
    split: str | None = None,
    max_samples: int = 1000,
    streaming: bool = True,
    cache_dir: str | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
    use_synthetic: bool = False,
    seed: int = 42,
    return_dataloader: bool = True,
    **kwargs: Any,
) -> DataLoader | list[dict[str, Any]]:
    """
    Unified loader for 12+ NLP / LLM benchmark datasets.

    Supports:
      - wikitext2 / wikitext103
      - c4 / c4_en
      - ptb
      - pg19
      - gsm8k
      - humaneval
      - mmlu
      - pile / pile_val
      - arc (ARC-Easy & ARC-Challenge)
      - hellaswag
      - lambada
      - glue (sst2, mrpc, qqp, mnli, qnli, rte, wnli, cola, stsb)

    Args:
        dataset_name: Benchmark name or alias (e.g. 'wikitext2', 'c4', 'gsm8k', 'arc', 'glue/sst2').
        tokenizer: HuggingFace tokenizer instance (optional).
        seq_len: Maximum sequence length for tokenization.
        batch_size: DataLoader batch size.
        split: Dataset split ('train', 'validation', 'test'). Defaults to benchmark standard.
        max_samples: Maximum number of samples to load/stream.
        streaming: Whether to stream from HuggingFace Hub to avoid large downloads.
        cache_dir: Local directory to cache datasets.
        shuffle: Whether to shuffle samples.
        num_workers: Number of workers for DataLoader.
        use_synthetic: Force synthetic data generation for offline testing.
        seed: Random seed for reproducibility.
        return_dataloader: If True, returns PyTorch DataLoader; otherwise returns list of dicts.
        **kwargs: Additional arguments passed to `datasets.load_dataset`.

    Returns:
        torch.utils.data.DataLoader or List[Dict[str, Any]].
    """
    clean_name = dataset_name.lower().replace("-", "_")
    info = BenchmarkDatasetRegistry.get(clean_name)

    # Fallback to general lookup if not matched directly
    if info is None and "/" in dataset_name:
        prefix, subtask = dataset_name.split("/", 1)
        if prefix.lower() == "glue":
            info = BenchmarkDatasetRegistry.get(f"glue_{subtask}")

    if info is None:
        logger.warning(
            "Dataset '%s' not in official benchmark registry. Creating generic fallback.",
            dataset_name,
        )
        info = BenchmarkDatasetInfo(
            name=clean_name,
            aliases=[],
            category="custom",
            default_hf_path=dataset_name,
            default_split=split or "validation",
            text_formatter=lambda r: {"text": r.get("text", str(r))},
            synthetic_generator=lambda n, s: [
                {
                    "text": f"Synthetic custom text sample #{i} for {dataset_name}.",
                    "id": i,
                }
                for i in range(n)
            ],
        )

    target_split = split or info.default_split
    raw_samples: list[dict[str, Any]] = []

    # Attempt online loading if not forced synthetic
    if not use_synthetic:
        try:
            from datasets import load_dataset as hf_load_dataset

            hf_kwargs = dict(kwargs)
            if info.default_subset:
                args = (info.default_hf_path, info.default_subset)
            else:
                args = (info.default_hf_path,)

            if cache_dir:
                hf_kwargs["cache_dir"] = cache_dir

            logger.info(
                "Attempting to load benchmark '%s' from HF (%s)...", info.name, args
            )

            try:
                # First try streaming if requested
                if streaming:
                    hf_ds = hf_load_dataset(
                        *args, split=target_split, streaming=True, **hf_kwargs
                    )
                    count = 0
                    for row in hf_ds:
                        formatted = (
                            info.text_formatter(row)
                            if info.text_formatter
                            else dict(row)
                        )
                        # Filter out empty text rows (e.g. WikiText empty headers)
                        if (
                            formatted.get("text", "").strip()
                            or "prompt" in formatted
                            or "question" in formatted
                        ):
                            raw_samples.append(formatted)
                            count += 1
                            if count >= max_samples:
                                break
                else:
                    hf_ds = hf_load_dataset(*args, split=target_split, **hf_kwargs)
                    sub_ds = hf_ds.select(range(min(len(hf_ds), max_samples)))
                    for row in sub_ds:
                        formatted = (
                            info.text_formatter(row)
                            if info.text_formatter
                            else dict(row)
                        )
                        raw_samples.append(formatted)

            except Exception as inner_e:
                logger.warning(
                    "Streaming/split load failed for %s (%s). Retrying non-streaming or default split...",
                    info.name,
                    inner_e,
                )
                hf_ds = hf_load_dataset(*args, **hf_kwargs)
                actual_split = (
                    target_split if target_split in hf_ds else list(hf_ds.keys())[0]
                )
                ds_split = hf_ds[actual_split]
                sub_ds = ds_split.select(range(min(len(ds_split), max_samples)))
                for row in sub_ds:
                    formatted = (
                        info.text_formatter(row) if info.text_formatter else dict(row)
                    )
                    raw_samples.append(formatted)

        except Exception as e:
            logger.warning(
                "Failed to load '%s' from HuggingFace Hub: %s. Falling back to synthetic generator.",
                dataset_name,
                str(e),
            )
            raw_samples = []

    # If empty or failed, run synthetic generator
    if not raw_samples:
        logger.info(
            "Generating %d synthetic samples for benchmark '%s' (seed=%d)...",
            max_samples,
            info.name,
            seed,
        )
        if info.synthetic_generator:
            raw_samples = info.synthetic_generator(max_samples, seed)
        else:
            raw_samples = [
                {"text": f"Synthetic benchmark text #{i} for {info.name}", "id": i}
                for i in range(max_samples)
            ]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(raw_samples)

    if not return_dataloader:
        return raw_samples

    # Build PyTorch Dataset and DataLoader
    is_lm = info.category in ("language_modeling", "code")
    torch_ds = BenchmarkTorchDataset(
        samples=raw_samples,
        tokenizer=tokenizer,
        seq_len=seq_len,
        is_causal_lm=is_lm,
    )

    loader = DataLoader(
        torch_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader


def get_synthetic_benchmark_sample(
    dataset_name: str, num_samples: int = 4, seed: int = 42
) -> list[str]:
    """Retrieve raw text strings of synthetic benchmark samples."""
    raw = load_benchmark_dataset(
        dataset_name,
        max_samples=num_samples,
        use_synthetic=True,
        seed=seed,
        return_dataloader=False,
    )
    results = []
    for item in raw:
        if isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("prompt")
                or item.get("sentence")
                or str(item)
            )
            results.append(text)
        else:
            results.append(str(item))
    return results
