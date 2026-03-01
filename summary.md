# Benchmark Summary

## Mean Speedup by Operation
- Aggregation::sum_all: 13.92x
- Aggregation::sum_axis_0: 174.03x
- Aggregation::sum_axis_1: 12.83x
- Basic & Math::conditional_replace: 29.31x
- Basic & Math::scalar_add: 110.70x
- Basic & Math::sin: 7.80x
- Broadcasting::matrix_plus_vector: 100.17x
- Filtering::threshold_filter: 5.00x
- Matrix Multiplication::matmul: 4004.87x
- Memory::copy_add: 88.08x
- Memory::in_place_add: 194.86x
- Memory::slice: 2254.31x

## Checkpoints
- Reverse threshold (A1): N=100 speedup=6.4578x, N=10000 speedup=115.7562x, N=1000000 speedup=209.8820x
- Cache wall (B2 vs B3, N=1000): B2=15.0628x, B3=264.2910x
- Memory allocation weight (E2 vs E3, N=1000000): E2=297.5653x, E3=165.9961x
- Broadcasting effectiveness (C1, N=1000): 112.5232x
