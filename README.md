# python_performance

This repository contains benchmarks that compare Python standard implementations (mainly list-based) against NumPy implementations across multiple processing categories.

## Goals

- Quantify how much faster NumPy is for different operation types
- Compare behavior across small and large input sizes
- Save benchmark results in CSV and summary markdown files

## Included Files

- `benchmark_business_comprehensive.py`
  - Runs benchmarks for 9 categories and 33 operation types
  - Validates output equivalence, then writes timing results to CSV and markdown
- `benchmark_plan.py`
  - Additional benchmark script
- `results_comprehensive.csv`
  - Raw results from the comprehensive benchmark
- `summary_comprehensive.md`
  - Aggregated summary for the comprehensive benchmark
- `results.csv`
  - Results from the secondary benchmark
- `summary.md`
  - Summary from the secondary benchmark

## How to Run

```bash
./.venv/bin/python benchmark_business_comprehensive.py
```

After running, the following files are generated or updated:

- `results_comprehensive.csv`
- `summary_comprehensive.md`

## Requirements

- Python 3.x
- NumPy
- If you use `./.venv`, install dependencies in that environment before running
