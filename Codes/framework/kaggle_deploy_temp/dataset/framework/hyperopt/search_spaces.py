from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SearchSpace:
    name: str
    type: str  # "float", "int", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    default: Optional[Any] = None


def suggest_float(
    name: str,
    low: float,
    high: float,
    log: bool = False,
    default: Optional[float] = None,
) -> SearchSpace:
    """Suggest float."""
    return SearchSpace(
        name=name, type="float", low=low, high=high, log=log, default=default
    )


def suggest_int(
    name: str, low: int, high: int, log: bool = False, default: Optional[int] = None
) -> SearchSpace:
    """Suggest int."""
    return SearchSpace(
        name=name, type="int", low=low, high=high, log=log, default=default
    )


def suggest_categorical(
    name: str, choices: List[Any], default: Optional[Any] = None
) -> SearchSpace:
    """Suggest categorical."""
    return SearchSpace(name=name, type="categorical", choices=choices, default=default)


def build_search_space(spaces: List[SearchSpace]) -> Dict[str, Any]:
    """Convert a list of SearchSpace objects into a backend‑agnostic dictionary."""
    result = {}
    for s in spaces:
        if s.type == "float":
            result[s.name] = {
                "type": "float",
                "low": s.low,
                "high": s.high,
                "log": s.log,
            }
        elif s.type == "int":
            result[s.name] = {"type": "int", "low": s.low, "high": s.high, "log": s.log}
        elif s.type == "categorical":
            result[s.name] = {"type": "categorical", "choices": s.choices}
    return result
