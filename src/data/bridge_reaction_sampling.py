# -*- coding: utf-8 -*-
"""
批量生成三跨连续梁支座反力（LHS采样）

可控变量与范围：
1) 支座沉降：以 A 为零点，B/C/D 为相对高度（mm），范围 [-10, 10]
2) 支座竖向刚度系数：A/B/C/D 的系数 ∈ [0.3, 1.5]（相对 Kv0）
3) 梁 EI 系数：每跨的系数 ∈ [0.3, 1.5]
4) 温度梯度：每跨 ΔT ∈ [-10, 20] ℃，相邻跨差值 |ΔT(i+1) - ΔT(i)| ≤ 5 ℃
5) 采样数量可配置，结果输出到 data/ 下 CSV

实现说明：
- 使用内置的拉丁超立方（LHS）实现，无 SciPy 依赖。
- 有限元模型沿用 Euler–Bernoulli 梁单元（每节点 w, θ 自由度），
  支座用竖向弹簧模拟，沉降作为预位移通过等效节点力/反力实现。
- 温差线性梯度对应恒定曲率 κ = α·ΔT/h，施加等效固定端弯矩 ±EIκ。

用法示例：
- 生成 200 条样本（默认 data/augmented/bridge_lhs_samples.csv）：
  uv run python data/bridge_reaction_sampling.py --n 200

- 指定随机种子和输出文件：
  uv run python data/bridge_reaction_sampling.py --n 1000 --seed 123 --out data/augmented/bridge_lhs_1000.csv

- 使用外部参数 CSV（禁用 LHS，逐行生成）：
  uv run python data/bridge_reaction_sampling.py --params data/custom_params.csv --out data/augmented/bridge_custom.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------- 常量与几何参数（可按需参数化） -----------------
E = 206_000.0                  # N/mm^2（钢）
Iy = 3.154279e11               # mm^4
EI_BASE = E * Iy               # N·mm^2

LMM = 40_000.0                 # 每跨长度 mm
SPAN_LEN = [LMM, LMM, LMM]
TOTAL_LEN = float(sum(SPAN_LEN))

# 均布恒载（向下为负，单位 N/mm）
Q_UNIFORM = -(12.0 + 20.0 + 29.64)

# 支座基准刚度（N/mm）
KV0 = 5_000.0 * 1_000.0

# 温差到曲率：κ = α·ΔT/h
ALPHA = 1.2e-5                 # 1/°C
H_MM = 2_000.0                 # mm


# ----------------- LHS 采样工具（无 SciPy 依赖） -----------------
def _lhs_builtin(n_samples: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """简易 LHS：[0,1] 超立方体，逐维分层并打乱。"""
    H = np.zeros((n_samples, dims), dtype=float)
    for j in range(dims):
        cut = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        rng.shuffle(cut)
        H[:, j] = cut
    return H


def _lhs_scipy(n_samples: int, dims: int, seed: int | None) -> np.ndarray:
    """若可用，使用 SciPy qmc.LatinHypercube 生成 LHS。"""
    try:
        from scipy.stats import qmc  # type: ignore
    except Exception:
        return None  # type: ignore
    # 优先使用带参数的新接口，失败则回退到基础构造
    try:
        sampler = qmc.LatinHypercube(d=dims, centered=True, optimization="random-cd", seed=seed)
    except TypeError:
        try:
            sampler = qmc.LatinHypercube(d=dims, seed=seed)
        except Exception:
            return None  # type: ignore
    return sampler.random(n=n_samples)


def lhs_unit_hypercube(n_samples: int, dims: int, rng: np.random.Generator, seed: int | None, backend: str = "auto") -> np.ndarray:
    """LHS 生成：优先 SciPy，失败则回退到内置实现。

    backend: 'auto' | 'scipy' | 'builtin'
    """
    if backend in ("auto", "scipy"):
        H = _lhs_scipy(n_samples, dims, seed)
        if H is not None:
            return H
        if backend == "scipy":
            raise RuntimeError("SciPy qmc.LatinHypercube not available.")
    return _lhs_builtin(n_samples, dims, rng)


def scale(u01: np.ndarray, low: float, high: float) -> np.ndarray:
    """把 [0,1] 数组线性缩放到 [low, high]。"""
    return low + (high - low) * u01


# ----------------- 梁单元 FE 工具 -----------------
def k_beam(ei: float, L: float) -> np.ndarray:
    fac = ei / (L ** 3)
    L2 = L * L
    return fac * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2],
        ],
        dtype=float,
    )


def f_udl(q: float, L: float) -> np.ndarray:
    return np.array([q * L / 2.0, q * L * L / 12.0, q * L / 2.0, -q * L * L / 12.0], dtype=float)


def f_thermal(ei: float, kappa: float) -> np.ndarray:
    # 固定端弯矩等效：EIκ * [0, -1, 0, +1]
    return ei * kappa * np.array([0.0, -1.0, 0.0, +1.0], dtype=float)


def build_mesh(ne_per_span: int) -> Tuple[np.ndarray, List[Tuple[int, int, float, int]]]:
    elements: List[Tuple[int, int, float, int]] = []
    x = [0.0]
    for si, L in enumerate(SPAN_LEN):
        dx = L / ne_per_span
        for _ in range(ne_per_span):
            n1 = len(x) - 1
            n2 = n1 + 1
            elements.append((n1, n2, dx, si))
            x.append(x[-1] + dx)
    return np.array(x, dtype=float), elements


@dataclass
class CaseParams:
    kv_factors: Tuple[float, float, float, float]
    settlements: Tuple[float, float, float, float]  # mm（A=0, B/C/D 相对 mm）
    ei_factors: Tuple[float, float, float]
    dT_spans: Tuple[float, float, float]  # °C
    q: float = Q_UNIFORM
    ne_per_span: int = 64


def solve_case(params: CaseParams) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """装配并求解，返回 (U, reactions_kN, reactions_list)。

    reactions_kN 顺序与支座 [A,B,C,D] 对应。
    """
    x_coords, elements = build_mesh(params.ne_per_span)
    ndof = 2 * len(x_coords)
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    # 热曲率
    kappa_span = tuple((ALPHA / H_MM) * np.array(params.dT_spans))

    # 组装
    for n1, n2, L, si in elements:
        ei = EI_BASE * params.ei_factors[si]
        ke = k_beam(ei, L)
        fe = f_udl(params.q, L)
        fth = f_thermal(ei, kappa_span[si])
        dof = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
        K[np.ix_(dof, dof)] += ke
        F[dof] += fe - fth

    # 支座：位于 x = 0, L, 2L, 3L
    support_x = [0.0, LMM, 2 * LMM, 3 * LMM]
    support_nodes: List[int] = []
    for xm in support_x:
        idx = int(np.argmin(np.abs(x_coords - xm)))
        support_nodes.append(idx)

    Kv = [KV0 * f for f in params.kv_factors]
    for i, idx in enumerate(support_nodes):
        # 竖向弹簧
        K[2 * idx, 2 * idx] += Kv[i]
        # 等效力：Kv * settlement
        F[2 * idx] += Kv[i] * params.settlements[i]

    # 求解
    U = np.linalg.solve(K, F)

    # 反力（kN），正向上
    reactions = []
    for i, idx in enumerate(support_nodes):
        ui = float(U[2 * idx])
        Ri = -Kv[i] * (ui - params.settlements[i]) / 1_000.0  # kN
        reactions.append(Ri)

    return U, np.array(reactions, dtype=float), [float(r) for r in reactions]


# ----------------- 采样与批处理 -----------------
def sample_cases_lhs(
    n: int,
    seed: int | None = None,
    lhs_backend: str = "auto",
) -> List[CaseParams]:
    rng = np.random.default_rng(seed)

    # 维度：B/C/D 沉降(3) + Kv系数(4) + EI系数(3) + 温度3维（t1, dt12, dt23） = 13
    U = lhs_unit_hypercube(n, dims=13, rng=rng, seed=seed, backend=lhs_backend)

    # 1) 沉降：A=0，B/C/D ∈ [-10, 10]
    sett_bcd = scale(U[:, 0:3], -10.0, 10.0)
    settlements = np.zeros((n, 4), dtype=float)
    settlements[:, 1:4] = sett_bcd

    # 2) Kv 系数：4 支座 ∈ [0.3, 1.5]
    kv_factors = scale(U[:, 3:7], 0.3, 1.5)

    # 3) EI 系数：3 跨 ∈ [0.3, 1.5]
    ei_factors = scale(U[:, 7:10], 0.3, 1.5)

    # 4) 温度：
    #    先采 t1 ∈ [-10, 20]，再基于当前值对相邻差值进行“逐步截断映射”，
    #    即 s1 ∈ [max(-5, -10-t1), min(5, 20-t1)]，t2 = t1 + s1；
    #    s2 ∈ [max(-5, -10-t2), min(5, 20-t2)]，t3 = t2 + s2。
    #    这样无需 clip，天然保证每跨在 [-10,20] 且相邻差值 ≤ 5。
    t1 = scale(U[:, 10], -10.0, 20.0)
    u12 = U[:, 11]
    u23 = U[:, 12]
    t2 = np.zeros_like(t1)
    t3 = np.zeros_like(t1)
    for i in range(n):
        # 针对第 i 个样本，计算允许的差值范围并映射
        low1 = max(-5.0, -10.0 - t1[i])
        high1 = min(5.0, 20.0 - t1[i])
        s1 = low1 + (high1 - low1) * u12[i]
        t2[i] = t1[i] + s1

        low2 = max(-5.0, -10.0 - t2[i])
        high2 = min(5.0, 20.0 - t2[i])
        s2 = low2 + (high2 - low2) * u23[i]
        t3[i] = t2[i] + s2

    dT_spans = np.stack([t1, t2, t3], axis=1)

    cases: List[CaseParams] = []
    for i in range(n):
        cases.append(
            CaseParams(
                kv_factors=tuple(map(float, kv_factors[i, :])),
                settlements=tuple(map(float, settlements[i, :])),
                ei_factors=tuple(map(float, ei_factors[i, :])),
                dT_spans=tuple(map(float, dT_spans[i, :])),
            )
        )
    return cases


PARAMETER_CSV_REQUIRED_COLS = [
    "settlement_b_mm",
    "settlement_c_mm",
    "settlement_d_mm",
    "kv_factor_a",
    "kv_factor_b",
    "kv_factor_c",
    "kv_factor_d",
    "ei_factor_s1",
    "ei_factor_s2",
    "ei_factor_s3",
    "dT_s1_C",
    "dT_s2_C",
    "dT_s3_C",
]

PARAMETER_CSV_OPTIONAL_SETTLEMENT_A = "settlement_a_mm"
PARAMETER_CSV_Q_CANDIDATES = ("q", "q_N_per_mm", "uniform_load_N_per_mm")
PARAMETER_CSV_NE_CANDIDATES = ("ne_per_span", "elements_per_span")


def _safe_float(value: Any, default: float) -> float:
    if value is None or pd.isna(value):
        return float(default)
    return float(value)


def _safe_int(value: Any, default: int) -> int:
    if value is None or pd.isna(value):
        return int(default)
    return int(value)


def load_cases_from_parameter_csv(param_csv: Path) -> Tuple[List[CaseParams], List[Any] | None]:
    """从 CSV 加载参数，构造 CaseParams 列表及可选的 sample_id 序列。"""
    df = pd.read_csv(param_csv)

    missing = [col for col in PARAMETER_CSV_REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            "参数 CSV 缺少必需列: " + ", ".join(missing)
        )

    has_sample_id = "sample_id" in df.columns
    sample_ids: List[Any] = []
    cases: List[CaseParams] = []

    for record in df.to_dict(orient="records"):
        settlements = (
            _safe_float(record.get(PARAMETER_CSV_OPTIONAL_SETTLEMENT_A, 0.0), 0.0),
            _safe_float(record["settlement_b_mm"], 0.0),
            _safe_float(record["settlement_c_mm"], 0.0),
            _safe_float(record["settlement_d_mm"], 0.0),
        )

        kv_factors = (
            _safe_float(record["kv_factor_a"], 1.0),
            _safe_float(record["kv_factor_b"], 1.0),
            _safe_float(record["kv_factor_c"], 1.0),
            _safe_float(record["kv_factor_d"], 1.0),
        )

        ei_factors = (
            _safe_float(record["ei_factor_s1"], 1.0),
            _safe_float(record["ei_factor_s2"], 1.0),
            _safe_float(record["ei_factor_s3"], 1.0),
        )

        dT_spans = (
            _safe_float(record["dT_s1_C"], 0.0),
            _safe_float(record["dT_s2_C"], 0.0),
            _safe_float(record["dT_s3_C"], 0.0),
        )

        q_value = None
        for col in PARAMETER_CSV_Q_CANDIDATES:
            if col in record and not pd.isna(record[col]):
                q_value = float(record[col])
                break
        if q_value is None:
            q_value = Q_UNIFORM

        ne_value = None
        for col in PARAMETER_CSV_NE_CANDIDATES:
            if col in record and not pd.isna(record[col]):
                ne_value = _safe_int(record[col], 64)
                break
        if ne_value is None:
            ne_value = 64

        case = CaseParams(
            kv_factors=tuple(float(v) for v in kv_factors),
            settlements=tuple(float(v) for v in settlements),
            ei_factors=tuple(float(v) for v in ei_factors),
            dT_spans=tuple(float(v) for v in dT_spans),
            q=float(q_value),
            ne_per_span=int(ne_value),
        )
        cases.append(case)

        if has_sample_id:
            sample_ids.append(record.get("sample_id"))

    return cases, (sample_ids if has_sample_id else None)


def generate_dataset(
    cases: Sequence[CaseParams],
    out_path: Path,
    ne_per_span: int | None = None,
    sample_ids: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """生成数据并写入 CSV，返回 DataFrame。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if sample_ids is not None and len(sample_ids) != len(cases):
        raise ValueError("sample_ids 长度与案例数量不一致")

    rows = []

    for idx, c in enumerate(cases):
        if ne_per_span is not None:
            c.ne_per_span = ne_per_span

        if sample_ids is not None:
            sid = sample_ids[idx]
            sample_id = idx if pd.isna(sid) else sid
        else:
            sample_id = idx

        load_kN = -c.q * TOTAL_LEN / 1_000.0  # 竖向总荷载（正数）

        U, R, Rlist = solve_case(c)

        sum_R = float(R.sum())
        eq_err = sum_R - load_kN  # 温度为纯弯矩，不引入合力

        # 按 19 点标签格式输出挠度与转角
        sensor_samples = sample_19pt_deflection_rotation(c, U)

        row = {
            "sample_id": sample_id,
            # 输入参数
            "settlement_a_mm": c.settlements[0],
            "settlement_b_mm": c.settlements[1],
            "settlement_c_mm": c.settlements[2],
            "settlement_d_mm": c.settlements[3],
            "kv_factor_a": c.kv_factors[0],
            "kv_factor_b": c.kv_factors[1],
            "kv_factor_c": c.kv_factors[2],
            "kv_factor_d": c.kv_factors[3],
            "ei_factor_s1": c.ei_factors[0],
            "ei_factor_s2": c.ei_factors[1],
            "ei_factor_s3": c.ei_factors[2],
            "dT_s1_C": c.dT_spans[0],
            "dT_s2_C": c.dT_spans[1],
            "dT_s3_C": c.dT_spans[2],
            "uniform_load_N_per_mm": c.q,
            # 输出（kN）
            "R_a_kN": Rlist[0],
            "R_b_kN": Rlist[1],
            "R_c_kN": Rlist[2],
            "R_d_kN": Rlist[3],
            "sum_R_kN": sum_R,
            "total_udl_kN": load_kN,
            "equilibrium_err_kN": eq_err,
            # 诊断
            "has_negative_reaction": bool(np.any(R < 0.0)),
        }

        # 追加：19点的挠度与转角（列名对齐 A, S1-1/6L, ... 的标签；将'/'替换为'_'）
        row.update(sensor_samples)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


# ----------------- 结果后处理：转角与挠度 -----------------
def sample_19pt_deflection_rotation(params: CaseParams, U: np.ndarray) -> dict:
    """对齐 19 点标签格式，输出每点的挠度与转角列。

    标签顺序：A, S1-1/6L..S1-5/6L, B, S2-1/6L..S2-5/6L, C, S3-1/6L..S3-5/6L, D
    列名：v_{label}_mm, theta_{label}_rad，其中 label 中的'/'替换为'_'
    """
    x_coords, elements = build_mesh(params.ne_per_span)

    def sensor_points_labels() -> Tuple[List[float], List[str]]:
        pts: List[float] = []
        labs: List[str] = []
        # A
        pts.append(0.0); labs.append("A")
        # S1
        for k in range(1, 6):
            pts.append(k / 6.0 * LMM)
            labs.append(f"S1-{k}/6L")
        # B
        pts.append(1.0 * LMM); labs.append("B")
        # S2
        for k in range(1, 6):
            pts.append(LMM + k / 6.0 * LMM)
            labs.append(f"S2-{k}/6L")
        # C
        pts.append(2.0 * LMM); labs.append("C")
        # S3
        for k in range(1, 6):
            pts.append(2 * LMM + k / 6.0 * LMM)
            labs.append(f"S3-{k}/6L")
        # D
        pts.append(3.0 * LMM); labs.append("D")
        return pts, labs

    def find_element(x: float) -> Tuple[int, float, float]:
        # 返回 (elem_left_node, L, x_local)
        if x >= x_coords[-1]:
            i = len(x_coords) - 2
        else:
            j = int(np.searchsorted(x_coords, x, side="right"))
            i = max(0, j - 1)
        xi = x_coords[i]
        L = x_coords[i + 1] - x_coords[i]
        return i, L, x - xi

    def eval_w_theta(x: float) -> Tuple[float, float]:
        i, L, xl = find_element(x)
        dof = [2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1]
        w1, th1, w2, th2 = U[dof[0]], U[dof[1]], U[dof[2]], U[dof[3]]
        xi = xl / L
        N1 = 1.0 - 3.0 * xi * xi + 2.0 * xi * xi * xi
        N2 = L * (xi - 2.0 * xi * xi + xi * xi * xi)
        N3 = 3.0 * xi * xi - 2.0 * xi * xi * xi
        N4 = L * (-xi * xi + xi * xi * xi)
        # 斜率函数（dN/dx）
        dN1 = (-6.0 * xi + 6.0 * xi * xi) / L
        dN2 = (1.0 - 4.0 * xi + 3.0 * xi * xi)
        dN3 = (6.0 * xi - 6.0 * xi * xi) / L
        dN4 = (-2.0 * xi + 3.0 * xi * xi)
        w = float(N1 * w1 + N2 * th1 + N3 * w2 + N4 * th2)
        th = float(dN1 * w1 + dN2 * th1 + dN3 * w2 + dN4 * th2)
        return w, th

    pts, labs = sensor_points_labels()
    out: dict = {}
    for x, lab in zip(pts, labs):
        w, th = eval_w_theta(x)
        safe_lab = lab.replace("/", "_")
        out[f"v_{safe_lab}_mm"] = w
        out[f"theta_{safe_lab}_rad"] = th
    return out


def main():
    parser = argparse.ArgumentParser(description="三跨连续梁支座反力 LHS 采样生成器")
    parser.add_argument("--n", type=int, default=200, help="样本数量（仅 LHS 模式）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（仅 LHS 模式）")
    parser.add_argument("--ne", type=int, default=None, help="每跨单元数量，默认 64")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data/augmented/bridge_lhs_samples.csv")),
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="参数 CSV 路径；提供后跳过 LHS，直接使用文件中的参数",
    )
    parser.add_argument(
        "--lhs",
        type=str,
        default="auto",
        choices=["auto", "scipy", "builtin"],
        help="LHS 后端：auto优先SciPy，否则回退；scipy强制SciPy；builtin内置实现",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    ne_override = args.ne

    if args.params:
        param_path = Path(args.params)
        cases, sample_ids = load_cases_from_parameter_csv(param_path)
        print(f"从参数 CSV 加载 {len(cases)} 条样本: {param_path}")
    else:
        cases = sample_cases_lhs(n=args.n, seed=args.seed, lhs_backend=args.lhs)
        sample_ids = None

    df = generate_dataset(
        cases=cases,
        out_path=out_path,
        ne_per_span=ne_override,
        sample_ids=sample_ids,
    )

    print(f"生成完成: {out_path} ({len(df)} 行)")
    # 简要统计
    neg = int(df["has_negative_reaction"].sum())
    print(f"含负反力样本: {neg}/{len(df)}")
    print(
        f"平衡误差 |mean| = {df['equilibrium_err_kN'].abs().mean():.3f} kN, max = {df['equilibrium_err_kN'].abs().max():.3f} kN"
    )


if __name__ == "__main__":
    main()
