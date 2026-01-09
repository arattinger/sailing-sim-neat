"""Action schema for the sailing simulator.

Controllers should emit :class:`TurnAction` instances or raw dictionaries with a
``turn_rate`` key measured in radians per second. The environment clamps the
command to the boat model's ``max_turn_rate`` before integration.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TurnAction:
    """Simple rudder-like command that specifies a turn rate.

    Attributes:
        turn_rate: Desired angular velocity in radians per second. Positive
            values turn the bow counterclockwise.
    """

    turn_rate: float

    def to_dict(self) -> Dict[str, float]:
        return {"turn_rate": float(self.turn_rate)}
