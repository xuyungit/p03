# -*- coding: utf-8 -*-
"""自定义桥梁参数 CSV 生成器。

该工具用于构造 bridge_reaction_sampling.py 可消费的参数批量 CSV。

核心特性：
- 每个参数支持固定值、随机采样、均匀取样、离散集合选择、自定义或相对生成模式。
- 可选约束（如相邻温差 ≤ 5 ℃）协助清洗结果。
- CLI 指定输出文件、样本数量及随机种子。
- 自定义函数示例：模拟 24 小时温度梯度循环，可灵活扩展。
 - 支持时间相邻约束（相邻样本点温差 ≤ 3 ℃，可配置）与多轮迭代收敛以满足多重约束。
 - 支持列内时间平滑（移动平均等）以降低高频抖动。

配置文件格式使用 JSON（保留键顺序）。示例：
{
  "n_samples": 128,
  "seed": 123,
  "parameters": {
    "settlement_b_mm": {"mode": "fixed", "value": 0.6963472526859210, "dtype": "float"},
    "settlement_c_mm": {"mode": "fixed", "value": -6.539185045863360, "dtype": "float"},
    "settlement_d_mm": {"mode": "fixed", "value": -4.611447972265490, "dtype": "float"},
    "kv_factor_a": {"mode": "fixed", "value": 1.1938421055767500, "dtype": "float"},
    "kv_factor_b": {"mode": "fixed", "value": 1.2583246581216900, "dtype": "float"},
    "kv_factor_c": {"mode": "fixed", "value": 0.4296195021186910, "dtype": "float"},
    "kv_factor_d": {"mode": "fixed", "value": 0.3417910882384080, "dtype": "float"},
    "ei_factor_s1": {"mode": "fixed", "value": 0.49857114855577800, "dtype": "float"},
    "ei_factor_s2": {"mode": "fixed", "value": 0.6286975090938600, "dtype": "float"},
    "ei_factor_s3": {"mode": "fixed", "value": 0.4596396912496840, "dtype": "float"},
    "dT_s1_C": {
      "mode": "custom",
      "function": "diurnal_temperature_delta",
      "params": {"amplitude": 30.0, "base": 5.0, "phase_shift_hours": 14.0, "noise_std": 0.5},
      "round": 2
    },
    "dT_s2_C": {
      "mode": "relative",
      "base": "dT_s1_C",
      "delta_mode": "uniform",
      "max_delta": 5.0,
      "min_total": -10.0,
      "max_total": 20.0,
      "round": 2
    },
    "dT_s3_C": {
      "mode": "relative",
      "base": "dT_s2_C",
      "delta_mode": "normal",
      "delta_std": 1.2,
      "max_delta": 5.0,
      "min_total": -10.0,
      "max_total": 20.0,
      "round": 2,
      "clip": false
    }
  },
    "constraints": [
    {
      "type": "adjacent_delta_limit",
      "columns": ["dT_s1_C", "dT_s2_C", "dT_s3_C"],
      "max_delta": 5.0,
      "clip": true
    },
    {
      "type": "temporal_delta_limit",
      "columns": ["dT_s1_C", "dT_s2_C", "dT_s3_C"],
      "max_delta": 3.0,
      "clip": true
    }
    ]
}

使用：
uv run python data/generate_custom_bridge_params.py --config data/param_recipe.json --out data/custom_params.csv --n 200 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np
import pandas as pd


# ----------------- 自定义函数库 -----------------
def diurnal_temperature_delta(
    n: int,
    rng: np.random.Generator,
    *,
    amplitude: float = 12.0,
    base: float = 0.0,
    phase_shift_hours: float = 14.0,
    noise_std: float = 0.5,
) -> np.ndarray:
    """生成一天内的温度梯度变化（°C）。

    采用简化的正弦-余弦叠加，使上午升温、傍晚降温，可选噪声。
    返回长度为 n 的数组，适合映射到 dT_* 参数。
    """

    hours = np.linspace(0.0, 24.0, num=n, endpoint=False)
    omega = 2.0 * math.pi / 24.0
    phase = omega * phase_shift_hours
    trend = amplitude / 2.0 * np.sin(omega * hours - phase)
    baseline = float(base)
    values = baseline + trend
    if noise_std > 0.0:
        values += rng.normal(loc=0.0, scale=noise_std, size=n)
    return values


CUSTOM_FUNCTIONS: Dict[str, Callable[..., np.ndarray]] = {
    "diurnal_temperature_delta": diurnal_temperature_delta,
}


class ConfigError(RuntimeError):
    pass


# ----------------- 参数生成工具 -----------------
def load_config(path: Path) -> MutableMapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"找不到配置文件: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"配置文件解析失败（JSON）: {exc}") from exc


def _get_range(spec: Mapping[str, Any], name: str) -> tuple[float, float]:
    keys_low = ("min", "low", "start")
    keys_high = ("max", "high", "stop")
    low = next((float(spec[k]) for k in keys_low if k in spec), None)
    high = next((float(spec[k]) for k in keys_high if k in spec), None)
    if low is None or high is None:
        raise ConfigError(f"参数 '{name}' 需要提供 min/max（或 start/stop/low/high）")
    if high < low:
        raise ConfigError(f"参数 '{name}' 的上界需 ≥ 下界 (low={low}, high={high})")
    return low, high


def _cast_type(values: Iterable[Any], dtype: str | None) -> List[Any]:
    if dtype is None:
        return list(values)
    if dtype.lower() in {"int", "integer"}:
        return [int(round(float(v))) for v in values]
    if dtype.lower() in {"float", "double"}:
        return [float(v) for v in values]
    if dtype.lower() in {"str", "string"}:
        return [str(v) for v in values]
    raise ConfigError(f"不支持的 dtype: {dtype}")


def _apply_round(values: np.ndarray, digits: int | None) -> np.ndarray:
    if digits is None:
        return values
    return np.round(values, digits)


def _bounded_normal_sample(
    rng: np.random.Generator,
    low: np.ndarray,
    high: np.ndarray,
    std: float,
    max_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """从 N(0, std^2) 重采样，使结果落在 [low, high] 区间。

    返回 (deltas, remaining_mask)。若 remaining_mask 仍有 True，说明重采样耗尽。"""

    n = low.shape[0]
    deltas = np.empty(n, dtype=float)
    remaining = np.ones(n, dtype=bool)

    for _ in range(max_attempts):
        if not remaining.any():
            break
        idx = np.flatnonzero(remaining)
        draws = rng.normal(loc=0.0, scale=std, size=idx.size)
        ok = (draws >= low[idx]) & (draws <= high[idx])
        if ok.any():
            assign_idx = idx[ok]
            deltas[assign_idx] = draws[ok]
            remaining[assign_idx] = False

    return deltas, remaining


def generate_parameter_series(
    name: str,
    spec: Mapping[str, Any],
    n: int,
    rng: np.random.Generator,
    generated: Mapping[str, List[Any]],
) -> List[Any]:
    mode = str(spec.get("mode", "fixed")).lower()
    dtype = spec.get("dtype")

    if mode == "fixed":
        if "value" not in spec:
            raise ConfigError(f"参数 '{name}' 的 fixed 模式需要 'value'")
        values = [spec["value"]] * n
        return _cast_type(values, dtype)

    if mode in {"random", "uniform", "linspace", "sobol"}:
        low, high = _get_range(spec, name)

        if mode == "uniform" or mode == "linspace":
            endpoint = bool(spec.get("endpoint", True))
            values = np.linspace(low, high, num=n, endpoint=endpoint)
        elif mode == "sobol":
            values = _sample_sobol_1d(n, low, high, rng)
        else:  # random
            values = rng.uniform(low, high, size=n)

        values = _apply_round(values, spec.get("round"))
        return _cast_type(values.tolist(), dtype)

    if mode == "normal":
        mean = float(spec.get("mean", 0.0))
        std = float(spec.get("std", 1.0))
        if std <= 0.0:
            raise ConfigError(f"参数 '{name}' 的 std 需为正数")
        values = rng.normal(loc=mean, scale=std, size=n)
        if "min" in spec or "max" in spec:
            low, high = _get_range(spec, name)
            values = np.clip(values, low, high)
        values = _apply_round(values, spec.get("round"))
        return _cast_type(values.tolist(), dtype)

    if mode in {"choice", "choices"}:
        options = spec.get("values") or spec.get("options")
        if not options:
            raise ConfigError(f"参数 '{name}' 的 choice 模式需要 'values'")
        values = rng.choice(options, size=n)
        return _cast_type(values.tolist(), dtype)

    if mode == "custom":
        func_name = spec.get("function")
        if not func_name:
            raise ConfigError(f"参数 '{name}' 的 custom 模式需要 'function'")
        func = CUSTOM_FUNCTIONS.get(func_name)
        if func is None:
            raise ConfigError(f"未注册的自定义函数: {func_name}")
        params = spec.get("params", {})
        if not isinstance(params, Mapping):
            raise ConfigError(f"参数 '{name}' 的 params 需为对象 (dict)")
        values = func(n=n, rng=rng, **params)
        if len(values) != n:
            raise ConfigError(
                f"自定义函数 '{func_name}' 返回长度 {len(values)}，与样本数量 {n} 不符"
            )
        values = _apply_round(np.asarray(values), spec.get("round"))
        return _cast_type(values.tolist(), dtype)

    if mode == "relative":
        base_col = spec.get("base") or spec.get("reference")
        if not base_col:
            raise ConfigError(f"参数 '{name}' 的 relative 模式需要 'base'")
        if base_col not in generated:
            raise ConfigError(
                f"参数 '{name}' 的 base 列 '{base_col}' 尚未生成，请调整配置顺序"
            )
        base_values = np.asarray(generated[base_col], dtype=float)
        if len(base_values) != n:
            raise ConfigError(
                f"参数 '{name}' 的 base 列 '{base_col}' 长度 {len(base_values)} 与样本数 {n} 不符"
            )
        max_delta = float(spec.get("max_delta", spec.get("limit", 0.0)))
        if max_delta <= 0.0:
            raise ConfigError(f"参数 '{name}' 的 max_delta 需为正数")
        delta_mode = str(spec.get("delta_mode", "uniform")).lower()
        clip = bool(spec.get("clip", True))

        min_total = spec.get("min_total")
        max_total = spec.get("max_total")
        if min_total is not None:
            min_total = float(min_total)
        if max_total is not None:
            max_total = float(max_total)

        allowed_low = np.full(n, -max_delta, dtype=float)
        allowed_high = np.full(n, max_delta, dtype=float)
        if min_total is not None:
            allowed_low = np.maximum(allowed_low, min_total - base_values)
        if max_total is not None:
            allowed_high = np.minimum(allowed_high, max_total - base_values)

        if np.any(allowed_low > allowed_high):
            raise ConfigError(
                f"参数 '{name}' 的相对范围与总范围不兼容，请检查 max_delta/min_total/max_total"
            )

        if delta_mode == "uniform":
            deltas = rng.uniform(low=allowed_low, high=allowed_high, size=n)
        elif delta_mode == "normal":
            delta_std = float(spec.get("delta_std", max_delta / 2.0))
            if delta_std <= 0.0:
                raise ConfigError(f"参数 '{name}' 的 delta_std 需为正数")
            deltas, remaining = _bounded_normal_sample(
                rng,
                low=allowed_low,
                high=allowed_high,
                std=delta_std,
                max_attempts=int(spec.get("resample_attempts", 100)),
            )
            if remaining.any():
                idx = np.flatnonzero(remaining)
                if clip:
                    midpoint = 0.5 * (allowed_low + allowed_high)
                    deltas[idx] = midpoint[idx]
                else:
                    raise ConfigError(
                        f"参数 '{name}' 的 normal 重采样达上限，仍有 {idx.size} 条不满足范围，可提高 max_delta 或设置 clip=true"
                    )
        elif delta_mode == "fixed":
            delta_val = float(spec.get("delta", 0.0))
            deltas = np.full(n, delta_val, dtype=float)
        else:
            raise ConfigError(f"参数 '{name}' 的 delta_mode 不支持: {delta_mode}")

        value_low = base_values + allowed_low
        value_high = base_values + allowed_high
        values = base_values + deltas
        if clip:
            values = np.clip(values, value_low, value_high)
        else:
            if np.any(values < value_low) or np.any(values > value_high):
                raise ConfigError(
                    f"参数 '{name}' 的相对结果超出约束，可调大 max_delta 或设置 clip=true"
                )

        values = _apply_round(np.asarray(values), spec.get("round"))
        return _cast_type(values.tolist(), dtype)

    raise ConfigError(f"参数 '{name}' 不支持的 mode: {mode}")


def _sample_sobol_1d(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    try:
        from scipy.stats import qmc  # type: ignore
    except Exception as exc:  # pragma: no cover - SciPy 可选
        raise ConfigError("要求 Sobol 采样但未安装 SciPy") from exc

    # Sobol 需要样本数为 2^m，上取整后再裁剪
    m = int(math.ceil(math.log2(max(1, n))))
    sampler = qmc.Sobol(d=1, scramble=True, seed=rng.integers(0, 2**31 - 1))
    raw = sampler.random_base2(m=m)
    scaled = qmc.scale(raw[:n, 0], low=low, high=high)
    return scaled


def _apply_adjacent_delta_limit(
    df: pd.DataFrame,
    columns: List[str],
    max_delta: float,
    *,
    clip: bool,
    rounding: Mapping[str, int | None] | None = None,
) -> Tuple[int, float]:
    """行内相邻列差值约束（跨梁），返回(修改次数, 本次最大改变量)。

    说明：保持第一列为锚点，从左到右依次投影到允许区间；若 clip=False，则遇到违规即抛错。
    """
    if max_delta < 0:
        raise ConfigError("adjacent_delta_limit 的 max_delta 需为非负数")
    if any(col not in df.columns for col in columns):
        missing = [col for col in columns if col not in df.columns]
        raise ConfigError(f"adjacent_delta_limit 缺少列: {missing}")

    data = df[columns].to_numpy(dtype=float, copy=True)
    changes = 0
    max_change = 0.0
    for i in range(data.shape[0]):
        prev = data[i, 0]
        for j in range(1, data.shape[1]):
            old = data[i, j]
            low = prev - max_delta
            high = prev + max_delta
            if clip:
                new_val = float(np.clip(old, low, high))
                # 可选四舍五入
                if rounding and columns[j] in rounding and rounding[columns[j]] is not None:
                    digits = int(rounding[columns[j]])
                    new_val = float(np.round(new_val, digits))
                    # 再次确保在可行域内
                    new_val = float(np.clip(new_val, low, high))
                if new_val != old:
                    data[i, j] = new_val
                    changes += 1
                    max_change = max(max_change, abs(new_val - old))
                prev = data[i, j]
            else:
                if old < low or old > high:
                    raise ConfigError(
                        f"行 {i} 列 '{columns[j]}' 与上一列的差值超出 {max_delta}"
                    )
                prev = old

    df.loc[:, columns] = data
    return changes, max_change


def _apply_temporal_delta_limit(
    df: pd.DataFrame,
    columns: List[str],
    max_delta: float,
    *,
    clip: bool,
    rounding: Mapping[str, int | None] | None = None,
) -> Tuple[int, float]:
    """列内相邻行差值约束（跨时间），返回(修改次数, 本次最大改变量)。

    策略：对每列先前向一次（t-1 -> t），再反向一次（t+1 -> t），相当于投影到
    |x_t - x_{t-1}| <= D 和 |x_{t+1} - x_t| <= D 的交集，减小偏置并有助于收敛。
    若 clip=False，遇到违规即抛错。
    """
    if max_delta < 0:
        raise ConfigError("temporal_delta_limit 的 max_delta 需为非负数")
    if any(col not in df.columns for col in columns):
        missing = [col for col in columns if col not in df.columns]
        raise ConfigError(f"temporal_delta_limit 缺少列: {missing}")

    total_changes = 0
    max_change = 0.0
    n = len(df)
    if n <= 1:
        return 0, 0.0

    for col in columns:
        arr = df[col].to_numpy(dtype=float, copy=True)

        # 前向 pass
        for t in range(1, n):
            prev = arr[t - 1]
            old = arr[t]
            low = prev - max_delta
            high = prev + max_delta
            if clip:
                new_val = float(np.clip(old, low, high))
                if rounding and col in rounding and rounding[col] is not None:
                    digits = int(rounding[col])
                    new_val = float(np.round(new_val, digits))
                    new_val = float(np.clip(new_val, low, high))
                if new_val != old:
                    arr[t] = new_val
                    total_changes += 1
                    max_change = max(max_change, abs(new_val - old))
            else:
                if old < low or old > high:
                    raise ConfigError(
                        f"列 '{col}' 在行 {t} 与前一行差值超出 {max_delta}"
                    )

        # 反向 pass
        for t in range(n - 2, -1, -1):
            nxt = arr[t + 1]
            old = arr[t]
            low = nxt - max_delta
            high = nxt + max_delta
            if clip:
                new_val = float(np.clip(old, low, high))
                if rounding and col in rounding and rounding[col] is not None:
                    digits = int(rounding[col])
                    new_val = float(np.round(new_val, digits))
                    new_val = float(np.clip(new_val, low, high))
                if new_val != old:
                    arr[t] = new_val
                    total_changes += 1
                    max_change = max(max_change, abs(new_val - old))
            else:
                if old < low or old > high:
                    raise ConfigError(
                        f"列 '{col}' 在行 {t} 与后一行差值超出 {max_delta}"
                    )

        df.loc[:, col] = arr

    return total_changes, max_change


def _apply_temporal_smooth(
    df: pd.DataFrame,
    columns: List[str],
    *,
    method: str = "moving_average",
    window: int = 5,
    passes: int = 1,
    rounding: Mapping[str, int | None] | None = None,
) -> Tuple[int, float]:
    """列内时间平滑约束，返回(修改次数, 最大改变量)。

    当前实现：moving_average（中心对齐、min_periods=1）。重复多次可增强平滑。
    """
    if any(col not in df.columns for col in columns):
        missing = [col for col in columns if col not in df.columns]
        raise ConfigError(f"temporal_smooth 缺少列: {missing}")
    if window <= 1 or passes <= 0:
        return 0, 0.0

    method = str(method).lower()
    if method not in {"moving_average", "ma", "mean"}:
        raise ConfigError(f"temporal_smooth 不支持的方法: {method}")

    changes = 0
    max_change = 0.0
    for col in columns:
        arr = df[col].to_numpy(dtype=float, copy=True)
        orig = arr.copy()
        for _ in range(passes):
            s = pd.Series(arr).rolling(window=window, center=True, min_periods=1).mean()
            arr = s.to_numpy(dtype=float)
        if rounding and col in rounding and rounding[col] is not None:
            digits = int(rounding[col])
            arr = np.round(arr, digits)
        diff = np.abs(arr - orig)
        cnt = int(np.count_nonzero(diff > 0))
        if cnt:
            changes += cnt
            max_change = max(max_change, float(diff.max()))
        df.loc[:, col] = arr

    return changes, max_change


def apply_constraints(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """根据配置对 DataFrame 应用约束。

    支持多轮迭代以同时满足多重约束：
      - 行内跨梁差值（adjacent_delta_limit）
      - 列内跨时间差值（temporal_delta_limit）
      - 列范围裁剪（column_range_limit，可选）

    可通过顶层配置控制迭代：
      - constraint_passes / max_constraint_passes / max_constraint_iterations：最大迭代轮数（默认 20）
      - constraint_tol：若单轮最大改变量 ≤ 该阈值则提前收敛（默认 0.0）
    同时根据 parameters[*].round（若存在）在投影后再进行四舍五入，并确保仍在可行域内。
    """

    specs = config.get("constraints")
    if not specs:
        return df
    if not isinstance(specs, list):
        raise ConfigError("constraints 需为数组")

    # 收集各列的 round 位数（若配置了）
    rounding: Dict[str, int | None] = {}
    params_cfg = config.get("parameters", {})
    if isinstance(params_cfg, Mapping):
        for k, v in params_cfg.items():
            if isinstance(v, Mapping) and "round" in v:
                try:
                    rounding[str(k)] = int(v.get("round"))
                except Exception:
                    rounding[str(k)] = None

    max_passes = int(
        config.get("constraint_passes")
        or config.get("max_constraint_passes")
        or config.get("max_constraint_iterations")
        or 20
    )
    tol = float(config.get("constraint_tol", 0.0))

    def _apply_one(spec: Mapping[str, Any]) -> Tuple[int, float]:
        ctype = str(spec.get("type", "")).lower()
        if ctype == "adjacent_delta_limit":
            columns = spec.get("columns") or spec.get("fields")
            if not columns or not isinstance(columns, (list, tuple)):
                raise ConfigError("adjacent_delta_limit 需要 'columns' 列表")
            if len(columns) < 2:
                raise ConfigError("adjacent_delta_limit 至少需要两个列名")
            max_delta = spec.get("max_delta")
            if max_delta is None:
                max_delta = spec.get("limit")
            if max_delta is None:
                raise ConfigError("adjacent_delta_limit 需要 'max_delta' 或 'limit'")
            clip = bool(spec.get("clip", True))
            return _apply_adjacent_delta_limit(
                df, list(columns), float(max_delta), clip=clip, rounding=rounding
            )

        if ctype == "temporal_delta_limit":
            columns = spec.get("columns") or spec.get("fields")
            if not columns or not isinstance(columns, (list, tuple)):
                raise ConfigError("temporal_delta_limit 需要 'columns' 列表")
            if len(columns) < 1:
                raise ConfigError("temporal_delta_limit 需要至少一个列名")
            max_delta = spec.get("max_delta")
            if max_delta is None:
                max_delta = spec.get("limit")
            if max_delta is None:
                raise ConfigError("temporal_delta_limit 需要 'max_delta' 或 'limit'")
            clip = bool(spec.get("clip", True))
            return _apply_temporal_delta_limit(
                df, list(columns), float(max_delta), clip=clip, rounding=rounding
            )

        if ctype in {"temporal_smooth", "temporal_smoothing"}:
            columns = spec.get("columns") or spec.get("fields")
            if not columns or not isinstance(columns, (list, tuple)):
                raise ConfigError("temporal_smooth 需要 'columns' 列表")
            method = spec.get("method", "moving_average")
            window = int(spec.get("window", 5))
            passes = int(spec.get("passes", 1))
            return _apply_temporal_smooth(
                df, list(columns), method=str(method), window=window, passes=passes, rounding=rounding
            )

        if ctype == "column_range_limit":
            columns = spec.get("columns") or spec.get("fields")
            if not columns or not isinstance(columns, (list, tuple)):
                raise ConfigError("column_range_limit 需要 'columns' 列表")
            if "min" not in spec and "max" not in spec:
                raise ConfigError("column_range_limit 需要至少指定 'min' 或 'max'")
            min_v = spec.get("min")
            max_v = spec.get("max")
            if min_v is not None:
                min_v = float(min_v)
            if max_v is not None:
                max_v = float(max_v)
            clip = bool(spec.get("clip", True))
            if not clip:
                # 非 clip 模式下仅检测
                for c in columns:
                    if min_v is not None and (df[c] < min_v).any():
                        raise ConfigError(f"列 '{c}' 存在小于最小值 {min_v} 的条目")
                    if max_v is not None and (df[c] > max_v).any():
                        raise ConfigError(f"列 '{c}' 存在大于最大值 {max_v} 的条目")
                return 0, 0.0

            # clip 模式：裁剪并可选四舍五入
            changes = 0
            max_change = 0.0
            for c in columns:
                arr = df[c].to_numpy(dtype=float, copy=True)
                old_arr = arr.copy()
                if min_v is not None:
                    arr = np.maximum(arr, min_v)
                if max_v is not None:
                    arr = np.minimum(arr, max_v)
                if rounding and c in rounding and rounding[c] is not None:
                    digits = int(rounding[c])
                    arr = np.round(arr, digits)
                    if min_v is not None:
                        arr = np.maximum(arr, min_v)
                    if max_v is not None:
                        arr = np.minimum(arr, max_v)
                diff = np.abs(arr - old_arr)
                cnt = int(np.count_nonzero(diff > 0))
                if cnt:
                    changes += cnt
                    max_change = max(max_change, float(diff.max()))
                df.loc[:, c] = arr
            return changes, max_change

        raise ConfigError(f"未知的 constraint 类型: {ctype}")

    # 迭代执行直至收敛或达到上限
    for _ in range(max_passes):
        total_changes = 0
        max_abs_change = 0.0
        for spec in specs:
            if not isinstance(spec, Mapping):
                raise ConfigError("constraint 定义需为对象 (dict)")
            chg, md = _apply_one(spec)
            total_changes += int(chg)
            max_abs_change = max(max_abs_change, float(md))
        if total_changes == 0 or max_abs_change <= tol:
            break

    return df


# ----------------- CLI 主流程 -----------------
def build_dataframe(
    config: Mapping[str, Any],
    *,
    n: int,
    seed: int,
    id_field: str,
) -> pd.DataFrame:
    params_cfg = config.get("parameters")
    if not isinstance(params_cfg, Mapping) or not params_cfg:
        raise ConfigError("配置需要 'parameters' 键并包含至少一个参数定义")

    rng = np.random.default_rng(seed)

    columns: Dict[str, List[Any]] = {}
    for name, spec in params_cfg.items():
        if not isinstance(spec, Mapping):
            raise ConfigError(f"参数 '{name}' 的定义需为对象 (dict)")
        col_name = str(name)
        columns[col_name] = generate_parameter_series(
            col_name,
            spec,
            n=n,
            rng=rng,
            generated=columns,
        )

    df = pd.DataFrame(columns)
    if id_field:
        df.insert(0, id_field, np.arange(n))
    return apply_constraints(df, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成桥梁分析自定义参数 CSV")
    parser.add_argument("--config", type=str, required=True, help="参数配置 JSON 文件路径")
    parser.add_argument("--out", type=str, required=True, help="输出 CSV 路径")
    parser.add_argument("--n", type=int, default=None, help="样本数量（覆盖配置中的 n_samples）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（覆盖配置）")
    parser.add_argument(
        "--id-field",
        type=str,
        default="sample_id",
        help="输出 CSV 的序号列名，留空可禁用",
    )
    return parser.parse_args()


def resolve_runtime_options(config: MutableMapping[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    conf_n = int(config.get("n_samples", 0)) if config.get("n_samples") else 0
    conf_seed = int(config.get("seed", 42)) if config.get("seed") is not None else 42

    final_n = int(args.n) if args.n else (conf_n if conf_n > 0 else 1)
    if final_n <= 0:
        raise ConfigError("样本数量必须为正整数")

    final_seed = int(args.seed) if args.seed is not None else conf_seed
    return final_n, final_seed


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    out_path = Path(args.out)

    config = load_config(config_path)
    n_samples, seed = resolve_runtime_options(config, args)

    id_field = args.id_field.strip()
    id_field = id_field if id_field else ""

    df = build_dataframe(config, n=n_samples, seed=seed, id_field=id_field)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(
        f"已生成自定义参数 CSV: {out_path} (样本数={len(df)}) [seed={seed}]"
    )


if __name__ == "__main__":
    try:
        main()
    except ConfigError as exc:
        raise SystemExit(f"配置错误: {exc}")
