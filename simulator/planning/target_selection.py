from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SelectedTarget:
    """Selected command target and its index in the people list."""

    person: Any
    index: int


def select_coffee_target(people: Sequence[Any]) -> SelectedTarget:
    """Return the first person whose command is exactly ``"coffee"``."""
    for index, person in enumerate(people):
        if getattr(person, "command", None) == "coffee":
            return SelectedTarget(person=person, index=index)
    raise ValueError("No person with command == 'coffee' was found")
