"""Column selection helpers shared across model entrypoints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    """Configuration for resolving columns from CSV headers."""

    names: Optional[Sequence[str]] = None
    patterns: Optional[Sequence[str]] = None

    def is_empty(self) -> bool:
        return not _normalize(self.names) and not _normalize(self.patterns)


def _normalize(items: Optional[Sequence[str]]) -> List[str]:
    if items is None:
        return []
    seen = set()
    out: List[str] = []
    for raw in items:
        val = str(raw).strip()
        if not val:
            continue
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def compile_patterns(patterns: Sequence[str]) -> List[re.Pattern[str]]:
    compiled: List[re.Pattern[str]] = []
    for pat in _normalize(patterns):
        try:
            compiled.append(re.compile(pat))
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern '{pat}': {exc}") from exc
    return compiled


def resolve_by_patterns(headers: Sequence[str], patterns: Sequence[str]) -> List[str]:
    regexes = compile_patterns(patterns)
    matched: List[str] = []
    seen = set()
    for name in headers:
        if name in seen:
            continue
        if any(rx.search(name) for rx in regexes):
            seen.add(name)
            matched.append(name)
    return matched


def resolve_columns(headers: Sequence[str], spec: ColumnSpec, *, allow_empty: bool = False) -> List[str]:
    if spec.is_empty():
        if allow_empty:
            return []
        raise ValueError("No columns specified. Provide explicit names or regex patterns.")
    resolved: List[str] = []
    seen = set()
    for name in _normalize(spec.names):
        if name not in headers:
            raise ValueError(f"Column '{name}' not found in available headers: {list(headers)}")
        if name in seen:
            continue
        seen.add(name)
        resolved.append(name)
    if spec.patterns:
        for name in resolve_by_patterns(headers, spec.patterns):
            if name in seen:
                continue
            seen.add(name)
            resolved.append(name)
    if not resolved and not allow_empty:
        raise ValueError("Resolved column list is empty after applying names and patterns.")
    return resolved


def read_headers(path: Path) -> List[str]:
    cols = list(pd.read_csv(path, nrows=0).columns)
    if not cols:
        raise ValueError(f"CSV file '{path}' has no columns.")
    return cols


def collect_header_map(paths: Sequence[Path]) -> Dict[Path, List[str]]:
    return {Path(p): read_headers(Path(p)) for p in paths}


def union_headers(header_map: Dict[Path, Sequence[str]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for cols in header_map.values():
        for name in cols:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
    return ordered


def ensure_consistent_headers(header_map: Dict[Path, Sequence[str]]) -> None:
    if len(header_map) <= 1:
        return
    iterator = iter(header_map.values())
    base = set(next(iterator))
    for idx, cols in enumerate(iterator, start=1):
        if set(cols) != base:
            raise ValueError(
                "Test files have inconsistent columns: "
                f"expected {sorted(base)}, got {sorted(set(cols))} for file index {idx}."
            )


def validate_required_columns(header_map: Dict[Path, Sequence[str]], required: Sequence[str], *, kind: str) -> None:
    missing_msgs: List[str] = []
    required_set = list(required)
    for path, headers in header_map.items():
        header_set = set(headers)
        missing = [name for name in required_set if name not in header_set]
        if missing:
            missing_msgs.append(f"{path}: {missing}")
    if missing_msgs:
        raise ValueError(f"Missing {kind} columns -> " + " | ".join(missing_msgs))


def resolve_from_csvs(
    train_paths: Sequence[Path],
    test_paths: Sequence[Path],
    input_spec: ColumnSpec,
    target_spec: ColumnSpec,
) -> Dict[str, List[str]]:
    train_map = collect_header_map(train_paths)
    test_map = collect_header_map(test_paths)
    all_train_headers = union_headers(train_map)
    input_cols = resolve_columns(all_train_headers, input_spec)
    target_cols = resolve_columns(all_train_headers, target_spec)
    ensure_consistent_headers(test_map)
    validate_required_columns(train_map, input_cols, kind="train input")
    validate_required_columns(train_map, target_cols, kind="train target")
    validate_required_columns(test_map, input_cols, kind="test input")
    validate_required_columns(test_map, target_cols, kind="test target")
    return {
        "input": input_cols,
        "target": target_cols,
    }
