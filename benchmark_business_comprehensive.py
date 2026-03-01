#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import statistics
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SEED = 42
REPEAT = 5

ONE_D_SIZES = [100, 10_000, 1_000_000]
TWO_D_SIZES = [100, 500, 1_000]
SORT_SIZES = [100, 10_000, 200_000]
MATMUL_SIZES = [100]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    operation: str
    business_use: str
    note: str
    sizes: list[int]
    profile: str
    builder: Callable[[int], tuple[Callable[[], object], Callable[[], object]]]
    rtol: float = 1e-9
    atol: float = 1e-9


def stable_seed(case_id: str, size: int) -> int:
    total = 0
    for idx, ch in enumerate(case_id, start=1):
        total += idx * ord(ch)
    return (SEED * 1_000_003 + total * 97 + size * 131) % (2**32)


def make_1d(case_id: str, size: int, low: float = -1.0, high: float = 1.0) -> tuple[list[float], np.ndarray]:
    if size <= 0:
        raise ValueError(f"size must be positive: {size}")
    rng = np.random.default_rng(stable_seed(case_id, size))
    arr = rng.uniform(low, high, size).astype(np.float64, copy=False)
    return arr.tolist(), arr


def make_2d(case_id: str, size: int, low: float = -1.0, high: float = 1.0) -> tuple[list[list[float]], np.ndarray]:
    if size <= 0:
        raise ValueError(f"size must be positive: {size}")
    rng = np.random.default_rng(stable_seed(case_id, size))
    arr = rng.uniform(low, high, (size, size)).astype(np.float64, copy=False)
    return arr.tolist(), arr


def benchmark_loops(profile: str, size: int) -> int:
    if size <= 0:
        raise ValueError(f"size must be positive: {size}")
    if profile == "1d":
        if size <= 100:
            return 500
        if size <= 10_000:
            return 50
        return 1
    if profile == "2d":
        if size <= 100:
            return 20
        if size <= 500:
            return 3
        return 1
    if profile == "sort":
        if size <= 100:
            return 200
        if size <= 10_000:
            return 10
        return 1
    if profile == "matmul":
        return 1
    raise ValueError(f"unknown profile: {profile}")


def time_call(func: Callable[[], object], loops: int) -> float:
    if loops <= 0:
        raise ValueError(f"loops must be positive: {loops}")
    func()
    timer = timeit.Timer(func)
    runs = timer.repeat(repeat=REPEAT, number=loops)
    return min(runs) / loops


def to_array(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if np.isscalar(value):
        return np.asarray([value])
    return np.asarray(value)


def assert_equivalent(
    py_out: object,
    np_out: object,
    case_id: str,
    size: int,
    rtol: float,
    atol: float,
) -> None:
    py_arr = to_array(py_out)
    np_arr = to_array(np_out)
    if py_arr.shape != np_arr.shape:
        raise ValueError(f"{case_id} size={size}: shape mismatch {py_arr.shape} != {np_arr.shape}")
    if np.issubdtype(py_arr.dtype, np.number) and np.issubdtype(np_arr.dtype, np.number):
        if not np.allclose(py_arr, np_arr, rtol=rtol, atol=atol, equal_nan=True):
            diff = np.max(np.abs(py_arr - np_arr))
            raise ValueError(f"{case_id} size={size}: value mismatch max_abs_diff={diff}")
        return
    if not np.array_equal(py_arr, np_arr):
        raise ValueError(f"{case_id} size={size}: non-numeric output mismatch")


def case_a1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("A1", size)

    def py_func() -> object:
        return [x + 1.5 for x in py_data]

    def np_func() -> object:
        return np_data + 1.5

    return py_func, np_func


def case_a2(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("A2", size)

    def py_func() -> object:
        return [x * 1.5 for x in py_data]

    def np_func() -> object:
        return np_data * 1.5

    return py_func, np_func


def case_a3(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("A3", size)

    def py_func() -> object:
        return [(x * 1.1) + 0.3 for x in py_data]

    def np_func() -> object:
        return (np_data * 1.1) + 0.3

    return py_func, np_func


def case_a4(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("A4", size, low=-3.0, high=3.0)

    def py_func() -> object:
        return [1.0 if x > 1.0 else (-1.0 if x < -1.0 else x) for x in py_data]

    def np_func() -> object:
        return np.clip(np_data, -1.0, 1.0)

    return py_func, np_func


def case_b1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("B1", size)

    def py_func() -> object:
        return [math.sin(x) for x in py_data]

    def np_func() -> object:
        return np.sin(np_data)

    return py_func, np_func


def case_b2(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("B2", size, low=-3.0, high=3.0)

    def py_func() -> object:
        return [math.exp(x) for x in py_data]

    def np_func() -> object:
        return np.exp(np_data)

    return py_func, np_func


def case_b3(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("B3", size)

    def py_func() -> object:
        return [math.log1p(abs(x)) for x in py_data]

    def np_func() -> object:
        return np.log1p(np.abs(np_data))

    return py_func, np_func


def case_b4(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("B4", size)

    def py_func() -> object:
        return [math.sqrt(abs(x) + 1e-12) for x in py_data]

    def np_func() -> object:
        return np.sqrt(np.abs(np_data) + 1e-12)

    return py_func, np_func


def case_c1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("C1", size)

    def py_func() -> object:
        return [1.0 if x > 0 else 0.0 for x in py_data]

    def np_func() -> object:
        return np.where(np_data > 0.0, 1.0, 0.0)

    return py_func, np_func


def case_c2(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("C2", size)

    def py_func() -> object:
        out = []
        for x in py_data:
            if x < -0.5:
                out.append(-1.0)
            elif x > 0.5:
                out.append(1.0)
            else:
                out.append(0.0)
        return out

    def np_func() -> object:
        return np.where(np_data < -0.5, -1.0, np.where(np_data > 0.5, 1.0, 0.0))

    return py_func, np_func


def case_c3(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("C3", size)

    def py_func() -> object:
        return [x for x in py_data if x > 0.5]

    def np_func() -> object:
        return np_data[np_data > 0.5]

    return py_func, np_func


def case_c4(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("C4", size)

    def py_func() -> object:
        return [x if x > 0.0 else 0.0 for x in py_data]

    def np_func() -> object:
        return np.where(np_data > 0.0, np_data, 0.0)

    return py_func, np_func


def case_d1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_2d("D1", size)

    def py_func() -> object:
        return [sum(row) for row in py_data]

    def np_func() -> object:
        return np.sum(np_data, axis=1)

    return py_func, np_func


def case_d2(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_2d("D2", size)

    def py_func() -> object:
        return [sum(row[col] for row in py_data) for col in range(size)]

    def np_func() -> object:
        return np.sum(np_data, axis=0)

    return py_func, np_func


def case_d3(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_2d("D3", size)

    def py_func() -> object:
        return [sum(row) / len(row) for row in py_data]

    def np_func() -> object:
        return np.mean(np_data, axis=1)

    return py_func, np_func


def case_d4(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_2d("D4", size)

    def py_func() -> object:
        return [max(row) for row in py_data]

    def np_func() -> object:
        return np.max(np_data, axis=1)

    return py_func, np_func


def case_d5(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("D5", size)

    def py_func() -> object:
        n = len(py_data)
        mean = sum(py_data) / n
        var = sum((x - mean) ** 2 for x in py_data) / n
        return math.sqrt(var)

    def np_func() -> object:
        return np.std(np_data)

    return py_func, np_func


def case_d6(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("D6", size)

    def py_func() -> object:
        sorted_data = sorted(py_data)
        idx = round(0.9 * (len(sorted_data) - 1))
        return sorted_data[idx]

    def np_func() -> object:
        return np.quantile(np_data, 0.9, method="nearest")

    return py_func, np_func


def case_e1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_a, np_a = make_1d("E1A", size)
    py_b, np_b = make_1d("E1B", size)

    def py_func() -> object:
        return sum(x * y for x, y in zip(py_a, py_b))

    def np_func() -> object:
        return np.dot(np_a, np_b)

    return py_func, np_func


def case_e2(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_mat, np_mat = make_2d("E2M", size)
    py_vec, np_vec = make_1d("E2V", size)

    def py_func() -> object:
        return [sum(x * y for x, y in zip(row, py_vec)) for row in py_mat]

    def np_func() -> object:
        return np_mat @ np_vec

    return py_func, np_func


def case_e3(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_mat, np_mat = make_2d("E3M", size)
    py_vec, np_vec = make_1d("E3V", size)

    def py_func() -> object:
        return [[x + y for x, y in zip(row, py_vec)] for row in py_mat]

    def np_func() -> object:
        return np_mat + np_vec

    return py_func, np_func


def case_f1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("F1", size)

    def py_func() -> object:
        n = len(py_data)
        mean = sum(py_data) / n
        var = sum((x - mean) ** 2 for x in py_data) / n
        std = math.sqrt(var)
        if std == 0.0:
            raise ValueError("std must not be zero")
        return [(x - mean) / std for x in py_data]

    def np_func() -> object:
        mean = np.mean(np_data)
        std = np.std(np_data)
        if std == 0.0:
            raise ValueError("std must not be zero")
        return (np_data - mean) / std

    return py_func, np_func


def case_f2(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("F2", size)

    def py_func() -> object:
        min_v = min(py_data)
        max_v = max(py_data)
        span = max_v - min_v
        if span == 0.0:
            raise ValueError("range must not be zero")
        return [(x - min_v) / span for x in py_data]

    def np_func() -> object:
        min_v = np.min(np_data)
        max_v = np.max(np_data)
        span = max_v - min_v
        if span == 0.0:
            raise ValueError("range must not be zero")
        return (np_data - min_v) / span

    return py_func, np_func


def case_f3(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("F3", size)
    window = 5
    kernel = np.ones(window, dtype=np.float64) / float(window)

    def py_func() -> object:
        out: list[float] = []
        limit = len(py_data) - window + 1
        for idx in range(limit):
            out.append(sum(py_data[idx : idx + window]) / window)
        return out

    def np_func() -> object:
        return np.convolve(np_data, kernel, mode="valid")

    return py_func, np_func


def case_f4(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("F4", size)

    def py_func() -> object:
        n = len(py_data)
        mean = sum(py_data) / n
        var = sum((x - mean) ** 2 for x in py_data) / n
        std = math.sqrt(var)
        if std == 0.0:
            raise ValueError("std must not be zero")
        out: list[float] = []
        for x in py_data:
            z = (x - mean) / std
            if z > 2.0:
                out.append(2.0)
            elif z < -2.0:
                out.append(-2.0)
            else:
                out.append(z)
        return out

    def np_func() -> object:
        mean = np.mean(np_data)
        std = np.std(np_data)
        if std == 0.0:
            raise ValueError("std must not be zero")
        return np.clip((np_data - mean) / std, -2.0, 2.0)

    return py_func, np_func


def case_g1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("G1", size)
    start = size // 4
    end = (3 * size) // 4

    def py_func() -> object:
        return py_data[start:end]

    def np_func() -> object:
        return np_data[start:end]

    return py_func, np_func


def case_g2(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("G2", size)

    def py_func() -> object:
        return py_data[::2]

    def np_func() -> object:
        return np_data[::2]

    return py_func, np_func


def case_g3(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("G3", size)
    py_work = py_data.copy()
    np_work = np_data.copy()

    def py_func() -> object:
        for idx in range(size):
            py_work[idx] += 1.0
        return py_work

    def np_func() -> object:
        np_work[:] += 1.0
        return np_work

    return py_func, np_func


def case_g4(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("G4", size)
    py_holder = py_data
    np_holder = np_data

    def py_func() -> object:
        nonlocal py_holder
        py_holder = [x + 1.0 for x in py_holder]
        return py_holder

    def np_func() -> object:
        nonlocal np_holder
        np_holder = np_holder + 1.0
        return np_holder

    return py_func, np_func


def case_g5(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    half = size // 2
    py_left, np_left = make_1d("G5L", half)
    py_right, np_right = make_1d("G5R", size - half)

    def py_func() -> object:
        return py_left + py_right

    def np_func() -> object:
        return np.concatenate((np_left, np_right))

    return py_func, np_func


def case_h1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("H1", size)

    def py_func() -> object:
        return sorted(py_data)

    def np_func() -> object:
        return np.sort(np_data)

    return py_func, np_func


def case_h2(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = make_1d("H2", size)
    k = 10

    def py_func() -> object:
        return sorted(py_data, reverse=True)[:k]

    def np_func() -> object:
        return np.sort(np_data)[-k:][::-1]

    return py_func, np_func


def case_i1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_a, np_a = make_2d("I1A", size)
    py_b, np_b = make_2d("I1B", size)

    def py_func() -> object:
        out = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            row_a = py_a[i]
            out_row = out[i]
            for k in range(size):
                a_ik = row_a[k]
                row_b = py_b[k]
                for j in range(size):
                    out_row[j] += a_ik * row_b[j]
        return out

    def np_func() -> object:
        return np_a @ np_b

    return py_func, np_func


CASES: list[BenchmarkCase] = [
    BenchmarkCase("A1", "Element-wise", "scalar_add", "Feature shift", "x + 1.5", ONE_D_SIZES, "1d", case_a1),
    BenchmarkCase("A2", "Element-wise", "scalar_mul", "Scaling", "x * 1.5", ONE_D_SIZES, "1d", case_a2),
    BenchmarkCase("A3", "Element-wise", "affine_transform", "Linear calibration", "x * 1.1 + 0.3", ONE_D_SIZES, "1d", case_a3),
    BenchmarkCase("A4", "Element-wise", "clip", "Outlier capping", "clip [-1,1]", ONE_D_SIZES, "1d", case_a4),
    BenchmarkCase("B1", "Math Functions", "sin", "Signal transform", "sin(x)", ONE_D_SIZES, "1d", case_b1),
    BenchmarkCase("B2", "Math Functions", "exp", "Score exponential", "exp(x)", ONE_D_SIZES, "1d", case_b2),
    BenchmarkCase("B3", "Math Functions", "log1p_abs", "Log scaling", "log1p(abs(x))", ONE_D_SIZES, "1d", case_b3),
    BenchmarkCase("B4", "Math Functions", "sqrt_abs", "Magnitude transform", "sqrt(abs(x))", ONE_D_SIZES, "1d", case_b4),
    BenchmarkCase("C1", "Conditional", "binary_threshold", "Rule-based flags", "x > 0", ONE_D_SIZES, "1d", case_c1),
    BenchmarkCase("C2", "Conditional", "ternary_bucket", "Risk bucketing", "[-1, 0, 1]", ONE_D_SIZES, "1d", case_c2),
    BenchmarkCase("C3", "Conditional", "filter_gt_0_5", "Eligibility filter", "x > 0.5", ONE_D_SIZES, "1d", case_c3),
    BenchmarkCase("C4", "Conditional", "relu_fill", "Negative suppression", "max(x, 0)", ONE_D_SIZES, "1d", case_c4),
    BenchmarkCase("D1", "Aggregation", "sum_axis_1", "Row totals", "2D row sum", TWO_D_SIZES, "2d", case_d1),
    BenchmarkCase("D2", "Aggregation", "sum_axis_0", "Column totals", "2D column sum", TWO_D_SIZES, "2d", case_d2),
    BenchmarkCase("D3", "Aggregation", "mean_axis_1", "Row averages", "2D row mean", TWO_D_SIZES, "2d", case_d3),
    BenchmarkCase("D4", "Aggregation", "max_axis_1", "Peak value per row", "2D row max", TWO_D_SIZES, "2d", case_d4),
    BenchmarkCase("D5", "Aggregation", "std_1d", "Volatility feature", "Population std", ONE_D_SIZES, "1d", case_d5, rtol=1e-7, atol=1e-9),
    BenchmarkCase("D6", "Aggregation", "quantile_90", "Percentile KPI", "Nearest 90th percentile", ONE_D_SIZES, "1d", case_d6),
    BenchmarkCase("E1", "Linear Algebra", "dot_1d", "Similarity score", "dot(a,b)", ONE_D_SIZES, "1d", case_e1, rtol=1e-7, atol=1e-8),
    BenchmarkCase("E2", "Linear Algebra", "matvec", "Weighted scoring", "matrix @ vector", TWO_D_SIZES, "2d", case_e2, rtol=1e-7, atol=1e-8),
    BenchmarkCase("E3", "Linear Algebra", "broadcast_add", "Bias addition per row", "matrix + vector", TWO_D_SIZES, "2d", case_e3),
    BenchmarkCase("F1", "Preprocessing", "zscore", "Standardization", "(x-mean)/std", ONE_D_SIZES, "1d", case_f1, rtol=1e-7, atol=1e-8),
    BenchmarkCase("F2", "Preprocessing", "minmax_scale", "Range normalization", "(x-min)/(max-min)", ONE_D_SIZES, "1d", case_f2, rtol=1e-7, atol=1e-8),
    BenchmarkCase("F3", "Preprocessing", "rolling_mean_w5", "Smoothing", "Window=5 moving average", ONE_D_SIZES, "1d", case_f3, rtol=1e-7, atol=1e-8),
    BenchmarkCase("F4", "Preprocessing", "zscore_clip", "Robust scaling", "zscore then clip [-2,2]", ONE_D_SIZES, "1d", case_f4, rtol=1e-7, atol=1e-8),
    BenchmarkCase("G1", "Memory/Indexing", "slice_contiguous", "Batch slicing", "copy(view) behavior", ONE_D_SIZES, "1d", case_g1),
    BenchmarkCase("G2", "Memory/Indexing", "slice_strided", "Down-sampling", "step=2 stride", ONE_D_SIZES, "1d", case_g2),
    BenchmarkCase("G3", "Memory/Indexing", "in_place_add", "Mutable update", "No new allocation", ONE_D_SIZES, "1d", case_g3),
    BenchmarkCase("G4", "Memory/Indexing", "copy_add", "Immutable-style update", "New allocation", ONE_D_SIZES, "1d", case_g4),
    BenchmarkCase("G5", "Memory/Indexing", "concat", "Data union", "Concatenate two chunks", ONE_D_SIZES, "1d", case_g5),
    BenchmarkCase("H1", "Sorting/Ranking", "sort_asc", "Ordering", "Ascending sort", SORT_SIZES, "sort", case_h1),
    BenchmarkCase("H2", "Sorting/Ranking", "top10_desc", "Top-k reporting", "Top 10 values", SORT_SIZES, "sort", case_h2),
    BenchmarkCase("I1", "Matrix Multiplication", "matmul", "Model inference core", "NxN @ NxN", MATMUL_SIZES, "matmul", case_i1, rtol=1e-7, atol=1e-8),
]


def run(output_csv: Path) -> None:
    rows: list[dict[str, str]] = []
    for case in CASES:
        for size in case.sizes:
            py_func, np_func = case.builder(size)
            py_out = py_func()
            np_out = np_func()
            assert_equivalent(py_out, np_out, case.case_id, size, case.rtol, case.atol)

            loops = benchmark_loops(case.profile, size)
            py_time = time_call(py_func, loops)
            np_time = time_call(np_func, loops)
            speedup = py_time / np_time if np_time > 0 else float("inf")

            print(
                f"{case.case_id} size={size} loops={loops} "
                f"python={py_time:.6e}s numpy={np_time:.6e}s speedup={speedup:.2f}x"
            )

            rows.append(
                {
                    "Category": case.category,
                    "Operation": case.operation,
                    "Business_Use": case.business_use,
                    "Size_N": str(size),
                    "Python_Time_Sec": f"{py_time:.10f}",
                    "NumPy_Time_Sec": f"{np_time:.10f}",
                    "Speedup_Factor": f"{speedup:.4f}",
                    "Note": f"{case.case_id}:{case.note}",
                }
            )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Category",
                "Operation",
                "Business_Use",
                "Size_N",
                "Python_Time_Sec",
                "NumPy_Time_Sec",
                "Speedup_Factor",
                "Note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(output_csv: Path, summary_md: Path) -> None:
    rows: list[dict[str, str]] = []
    with output_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    by_category: dict[str, list[float]] = {}
    by_operation: dict[str, list[float]] = {}
    for row in rows:
        speed = float(row["Speedup_Factor"])
        by_category.setdefault(row["Category"], []).append(speed)
        key = f"{row['Category']}::{row['Operation']}"
        by_operation.setdefault(key, []).append(speed)

    lines = ["# Comprehensive Benchmark Summary", ""]
    lines.append("## Mean Speedup by Category")
    for category in sorted(by_category):
        lines.append(f"- {category}: {statistics.mean(by_category[category]):.2f}x")

    lines.append("")
    lines.append("## Mean Speedup by Operation")
    for operation in sorted(by_operation):
        lines.append(f"- {operation}: {statistics.mean(by_operation[operation]):.2f}x")

    def pick(operation: str, size: int) -> dict[str, str]:
        for row in rows:
            if row["Operation"] == operation and int(row["Size_N"]) == size:
                return row
        raise ValueError(f"missing row operation={operation} size={size}")

    lines.extend(
        [
            "",
            "## Practical Checkpoints",
            (
                "- Threshold overhead line (scalar_add): "
                f"N=100 {pick('scalar_add', 100)['Speedup_Factor']}x, "
                f"N=10000 {pick('scalar_add', 10_000)['Speedup_Factor']}x, "
                f"N=1000000 {pick('scalar_add', 1_000_000)['Speedup_Factor']}x"
            ),
            (
                "- Cache locality impact (sum_axis_1 vs sum_axis_0 at N=1000): "
                f"axis1 {pick('sum_axis_1', 1_000)['Speedup_Factor']}x, "
                f"axis0 {pick('sum_axis_0', 1_000)['Speedup_Factor']}x"
            ),
            (
                "- Allocation impact (in_place_add vs copy_add at N=1000000): "
                f"in-place {pick('in_place_add', 1_000_000)['Speedup_Factor']}x, "
                f"copy {pick('copy_add', 1_000_000)['Speedup_Factor']}x"
            ),
            (
                "- Pipeline gain (rolling_mean_w5 at N=1000000): "
                f"{pick('rolling_mean_w5', 1_000_000)['Speedup_Factor']}x"
            ),
            (
                "- Ranking workload (top10_desc at N=200000): "
                f"{pick('top10_desc', 200_000)['Speedup_Factor']}x"
            ),
        ]
    )

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_csv = Path("results_comprehensive.csv")
    output_summary = Path("summary_comprehensive.md")
    run(output_csv)
    summarize(output_csv, output_summary)
    print(f"\nWrote: {output_csv}")
    print(f"Wrote: {output_summary}")


if __name__ == "__main__":
    main()
