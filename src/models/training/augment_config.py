"""
Configurable augmentation loader and flip augmentor.

Schema (JSON):
{
  "type": "flip",
  "swap": [["A","B"], ["C","D"]],
  "swap_negate": [["Ry_t1","Ry_t4"], ["Ry_t2","Ry_t3"]],
  "negate": ["X", "Y"]
}

- swap: exchange values of columns in each pair
- swap_negate: exchange and negate (a' = -b, b' = -a)
- negate: multiply by -1 in place

Profiles can be resolved from models/augments/{profile}.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import json
import pandas as pd


@dataclass
class FlipRules:
    swap: List[Tuple[str, str]]
    swap_negate: List[Tuple[str, str]]
    negate: List[str]


class FlipAugmentor:
    def __init__(self, rules: FlipRules) -> None:
        self.rules = rules

    def apply_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # swap
        for a, b in self.rules.swap:
            if a in out.columns and b in out.columns:
                tmp = out[a].copy()
                out[a] = out[b]
                out[b] = tmp
        # swap_negate
        for a, b in self.rules.swap_negate:
            if a in out.columns and b in out.columns:
                a_vals = out[a].to_numpy(copy=True)
                b_vals = out[b].to_numpy(copy=True)
                out[a] = -b_vals
                out[b] = -a_vals
        # negate
        for c in self.rules.negate:
            if c in out.columns:
                out[c] = -out[c]
        return out


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _rules_from_dict(d: dict) -> FlipRules:
    if str(d.get("type", "flip")) != "flip":
        raise ValueError("augment_config type must be 'flip'")
    swap = [(str(a), str(b)) for a, b in d.get("swap", [])]
    swap_negate = [(str(a), str(b)) for a, b in d.get("swap_negate", [])]
    negate = [str(c) for c in d.get("negate", [])]
    return FlipRules(swap=swap, swap_negate=swap_negate, negate=negate)


def load_augmentor(config_path: Optional[Path], profile: Optional[str]) -> Optional[FlipAugmentor]:
    """Load FlipAugmentor from explicit path or named profile.

    - If profile is 'none' or falsy, returns None (no-op)
    - If path provided, load from it
    - Else if profile provided, look under models/augments/{profile}.json
    """
    if profile and profile.lower() == "none":
        return None
    if config_path is not None:
        d = _load_json(Path(config_path))
        return FlipAugmentor(_rules_from_dict(d))
    if profile:
        base = Path(__file__).resolve().parents[1] / "augments" / f"{profile}.json"
        d = _load_json(base)
        return FlipAugmentor(_rules_from_dict(d))
    return None

