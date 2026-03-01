#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import random
import statistics
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

PY_RANDOM_SEED = 42
RNG = random.Random(PY_RANDOM_SEED)
NP_RNG = np.random.default_rng(PY_RANDOM_SEED)

ONE_D_SIZES = [100, 10_000, 1_000_000]
MATRIX_SIZES = [100, 500, 1_000]
MATMUL_SIZE = 100
REPEAT = 5


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    operation: str
    note: str
    sizes: list[int]
    factory: Callable[[int], tuple[Callable[[], object], Callable[[], object]]]


def time_callable(func: Callable[[], object], number: int) -> float:
    func()
    timer = timeit.Timer(func)
    samples = timer.repeat(repeat=REPEAT, number=number)
    best = min(samples)
    return best / number


def loops_for_size(size: int, category: str) -> int:
    if category in {"Aggregation", "Broadcasting"}:
        if size <= 100:
            return 20
        if size <= 500:
            return 5
        return 2
    if category == "Matrix Multiplication":
        return 1
    if size <= 100:
        return 1_000
    if size <= 10_000:
        return 100
    return 3


def build_1d_data(size: int) -> tuple[list[float], np.ndarray]:
    data_np = NP_RNG.standard_normal(size, dtype=np.float64)
    return data_np.tolist(), data_np


def build_2d_data(size: int) -> tuple[list[list[float]], np.ndarray]:
    data_np = NP_RNG.standard_normal((size, size), dtype=np.float64)
    return data_np.tolist(), data_np


def case_a1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = build_1d_data(size)

    def py_func() -> object:
        return [x + 1.5 for x in py_data]

    def np_func() -> object:
        return np_data + 1.5

    return py_func, np_func


def case_a2(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = build_1d_data(size)

    def py_func() -> object:
        return [math.sin(x) for x in py_data]

    def np_func() -> object:
        return np.sin(np_data)

    return py_func, np_func


def case_a3(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = build_1d_data(size)

    def py_func() -> object:
        return [1.0 if x > 0 else 0.0 for x in py_data]

    def np_func() -> object:
        return np.where(np_data > 0.0, 1.0, 0.0)

    return py_func, np_func


def case_b1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = build_2d_data(size)

    def py_func() -> object:
        return sum(sum(row) for row in py_data)

    def np_func() -> object:
        return np.sum(np_data)

    return py_func, np_func


def case_b2(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = build_2d_data(size)

    def py_func() -> object:
        return [sum(row) for row in py_data]

    def np_func() -> object:
        return np.sum(np_data, axis=1)

    return py_func, np_func


def case_b3(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = build_2d_data(size)

    def py_func() -> object:
        return [sum(row[idx] for row in py_data) for idx in range(size)]

    def np_func() -> object:
        return np.sum(np_data, axis=0)

    return py_func, np_func


def case_c1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_matrix, np_matrix = build_2d_data(size)
    py_vec = [float(v) for v in NP_RNG.standard_normal(size, dtype=np.float64)]
    np_vec = np.array(py_vec, dtype=np.float64)

    def py_func() -> object:
        return [[x + y for x, y in zip(row, py_vec)] for row in py_matrix]

    def np_func() -> object:
        return np_matrix + np_vec

    return py_func, np_func


def case_d1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = build_1d_data(size)

    def py_func() -> object:
        return [x for x in py_data if x > 0.5]

    def np_func() -> object:
        return np_data[np_data > 0.5]

    return py_func, np_func


def case_e1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = build_1d_data(size)
    start = size // 4
    end = (3 * size) // 4

    def py_func() -> object:
        return py_data[start:end]

    def np_func() -> object:
        return np_data[start:end]

    return py_func, np_func


def case_e2(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = build_1d_data(size)
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


def case_e3(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_data, np_data = build_1d_data(size)
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


def case_f1(size: int) -> tuple[Callable[[], object], Callable[[], object]]:
    py_a, np_a = build_2d_data(size)
    py_b, np_b = build_2d_data(size)

    def py_func() -> object:
        out = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            row_i = py_a[i]
            out_row = out[i]
            for k in range(size):
                a_ik = row_i[k]
                row_k = py_b[k]
                for j in range(size):
                    out_row[j] += a_ik * row_k[j]
        return out

    def np_func() -> object:
        return np_a @ np_b

    return py_func, np_func


CASES = [
    BenchmarkCase("A1", "Basic & Math", "scalar_add", "element-wise", ONE_D_SIZES, case_a1),
    BenchmarkCase("A2", "Basic & Math", "sin", "heavy_math", ONE_D_SIZES, case_a2),
    BenchmarkCase("A3", "Basic & Math", "conditional_replace", "mask_like", ONE_D_SIZES, case_a3),
    BenchmarkCase("B1", "Aggregation", "sum_all", "2d_total_sum", MATRIX_SIZES, case_b1),
    BenchmarkCase("B2", "Aggregation", "sum_axis_1", "row_sum", MATRIX_SIZES, case_b2),
    BenchmarkCase("B3", "Aggregation", "sum_axis_0", "column_sum_stride", MATRIX_SIZES, case_b3),
    BenchmarkCase("C1", "Broadcasting", "matrix_plus_vector", "broadcasting", MATRIX_SIZES, case_c1),
    BenchmarkCase("D1", "Filtering", "threshold_filter", "boolean_index", ONE_D_SIZES, case_d1),
    BenchmarkCase("E1", "Memory", "slice", "python_copy_numpy_view", ONE_D_SIZES, case_e1),
    BenchmarkCase("E2", "Memory", "in_place_add", "destructive_update", ONE_D_SIZES, case_e2),
    BenchmarkCase("E3", "Memory", "copy_add", "new_allocation", ONE_D_SIZES, case_e3),
    BenchmarkCase("F1", "Matrix Multiplication", "matmul", "small_only", [MATMUL_SIZE], case_f1),
]


def run_benchmarks(output_csv: Path) -> None:
    rows: list[dict[str, str]] = []
    for case in CASES:
        for size in case.sizes:
            py_func, np_func = case.factory(size)
            loops = loops_for_size(size, case.category)

            py_time = time_callable(py_func, number=loops)
            np_time = time_callable(np_func, number=loops)
            speedup = py_time / np_time if np_time > 0 else float("inf")

            rows.append(
                {
                    "Category": case.category,
                    "Operation": case.operation,
                    "Size_N": str(size),
                    "Python_Time_Sec": f"{py_time:.10f}",
                    "NumPy_Time_Sec": f"{np_time:.10f}",
                    "Speedup_Factor": f"{speedup:.4f}",
                    "Note": f"{case.case_id}:{case.note}",
                }
            )
            print(
                f"{case.case_id} size={size} loops={loops} "
                f"python={py_time:.6e}s numpy={np_time:.6e}s speedup={speedup:.2f}x"
            )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Category",
                "Operation",
                "Size_N",
                "Python_Time_Sec",
                "NumPy_Time_Sec",
                "Speedup_Factor",
                "Note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(output_csv: Path, summary_path: Path) -> None:
    rows: list[dict[str, str]] = []
    with output_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    by_op: dict[str, list[float]] = {}
    for row in rows:
        key = f"{row['Category']}::{row['Operation']}"
        by_op.setdefault(key, []).append(float(row["Speedup_Factor"]))

    lines = [
        "# Benchmark Summary",
        "",
        "## Mean Speedup by Operation",
    ]
    for key in sorted(by_op):
        lines.append(f"- {key}: {statistics.mean(by_op[key]):.2f}x")

    def find_row(operation: str, size: int) -> dict[str, str]:
        for row in rows:
            if row["Operation"] == operation and int(row["Size_N"]) == size:
                return row
        raise ValueError(f"row not found: {operation} size={size}")

    lines.extend(
        [
            "",
            "## Checkpoints",
            (
                "- Reverse threshold (A1): "
                f"N=100 speedup={find_row('scalar_add', 100)['Speedup_Factor']}x, "
                f"N=10000 speedup={find_row('scalar_add', 10_000)['Speedup_Factor']}x, "
                f"N=1000000 speedup={find_row('scalar_add', 1_000_000)['Speedup_Factor']}x"
            ),
            (
                "- Cache wall (B2 vs B3, N=1000): "
                f"B2={find_row('sum_axis_1', 1_000)['Speedup_Factor']}x, "
                f"B3={find_row('sum_axis_0', 1_000)['Speedup_Factor']}x"
            ),
            (
                "- Memory allocation weight (E2 vs E3, N=1000000): "
                f"E2={find_row('in_place_add', 1_000_000)['Speedup_Factor']}x, "
                f"E3={find_row('copy_add', 1_000_000)['Speedup_Factor']}x"
            ),
            (
                "- Broadcasting effectiveness (C1, N=1000): "
                f"{find_row('matrix_plus_vector', 1_000)['Speedup_Factor']}x"
            ),
        ]
    )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_csv = Path("results.csv")
    summary_md = Path("summary.md")
    run_benchmarks(output_csv)
    summarize(output_csv, summary_md)
    print(f"\nWrote: {output_csv}")
    print(f"Wrote: {summary_md}")


if __name__ == "__main__":
    main()
