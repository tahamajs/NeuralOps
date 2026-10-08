"""Synthetic data generators for benchmark datasets.

Provides offline synthetic generation of language modeling, reasoning, code,
multiple-choice, and GLUE tasks for smoke tests and offline environments.
"""

from __future__ import annotations

import random
from typing import Any


def _generate_synthetic_wikitext(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate synthetic WikiText language modeling samples."""
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
    """Generate synthetic C4 web crawl samples."""
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
    """Generate synthetic Penn Treebank financial sentences."""
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
    """Generate synthetic PG-19 long book chapters."""
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
    """Generate synthetic grade school math problems."""
    samples = []
    for i in range(num_samples):
        apples_per_basket = 5 + (i % 6)
        num_baskets = 3 + (i % 4)
        eaten = 2 + (i % 3)
        total_apples = apples_per_basket * num_baskets
        remaining = total_apples - eaten
        q = f"Janet has {num_baskets} baskets of apples. Each basket contains {apples_per_basket} apples. If she eats {eaten} apples, how many apples does she have left?"
        a = f"First, calculate the total number of apples: {num_baskets} * {apples_per_basket} = {total_apples}. Next, subtract the eaten apples: {total_apples} - {eaten} = {remaining}.\n#### {remaining}"
        samples.append({"question": q, "answer": a, "text": f"Question: {q}\nAnswer: {a}"})
    return samples


def _generate_synthetic_humaneval(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate synthetic HumanEval Python function problems."""
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
        test = "def check(candidate):\n    assert candidate([1.0, 2.0], 0.5) is not None\n"
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
    """Generate synthetic MMLU multiple choice questions."""
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
        q = f"In the context of {subject.replace('_', ' ')}, which statement regarding fundamental principle #{i + 1} is strictly valid?"
        choices = [
            f"Option A: Principle {i + 1} guarantees monotonic convergence under convexity constraints.",
            f"Option B: Principle {i + 1} always yields an asymptotic upper bound of O(N log N).",
            f"Option C: Principle {i + 1} implies invariance under orthonormal coordinate rotations.",
            f"Option D: Principle {i + 1} requires strict thermodynamic equilibrium.",
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
    """Generate synthetic The Pile document fragments."""
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
    """Generate synthetic AI2 Reasoning Challenge questions."""
    samples = []
    for i in range(num_samples):
        q = f"Which process is primarily responsible for transferring thermal energy through fluid movement in system #{i + 1}?"
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


def _generate_synthetic_hellaswag(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate synthetic HellaSwag commonsense reasoning items."""
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


def _generate_synthetic_lambada(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate synthetic LAMBADA word prediction stories."""
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


def _generate_synthetic_glue(task: str, num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate synthetic GLUE task samples."""
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
            samples.append({"sentence": sentence, "label": acceptable, "text": sentence})
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


__all__ = [
    "_generate_synthetic_wikitext",
    "_generate_synthetic_c4",
    "_generate_synthetic_ptb",
    "_generate_synthetic_pg19",
    "_generate_synthetic_gsm8k",
    "_generate_synthetic_humaneval",
    "_generate_synthetic_mmlu",
    "_generate_synthetic_pile",
    "_generate_synthetic_arc",
    "_generate_synthetic_hellaswag",
    "_generate_synthetic_lambada",
    "_generate_synthetic_glue",
]
