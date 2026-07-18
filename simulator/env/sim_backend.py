from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SimBackend(ABC):
    """Common simulator interface for training, testing, and evaluation backends.

    Implementations such as GenesisSimBackend should keep
    this contract stable so callers can swap simulation backends without changing
    training or evaluation code.
    """

    @abstractmethod
    def reset(self, start: Any, goal: Any, obstacles: Any) -> Any:
        """Reset the simulation scene to a start-goal-obstacle problem."""
        raise NotImplementedError

    @abstractmethod
    def rollout_trajectory(self, trajectory: Any) -> Any:
        """Roll out or evaluate a candidate trajectory in the backend."""
        raise NotImplementedError

    @abstractmethod
    def check_collisions(self, trajectory: Any) -> Any:
        """Return collision information for a candidate trajectory."""
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> Any:
        """Return the latest backend observation/state."""
        raise NotImplementedError

    @abstractmethod
    def render(self) -> Any:
        """Render the current scene if supported by the backend."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""
        raise NotImplementedError
