# Comprehensive Benchmark Summary

## Mean Speedup by Category
- Aggregation: 52.37x
- Conditional: 18.98x
- Element-wise: 71.80x
- Linear Algebra: 199.56x
- Math Functions: 21.37x
- Matrix Multiplication: 4430.47x
- Memory/Indexing: 1044.69x
- Preprocessing: 76.38x
- Sorting/Ranking: 3.07x

## Mean Speedup by Operation
- Aggregation::max_axis_1: 64.40x
- Aggregation::mean_axis_1: 12.09x
- Aggregation::quantile_90: 9.01x
- Aggregation::std_1d: 48.23x
- Aggregation::sum_axis_0: 167.59x
- Aggregation::sum_axis_1: 12.89x
- Conditional::binary_threshold: 30.80x
- Conditional::filter_gt_0_5: 4.97x
- Conditional::relu_fill: 23.14x
- Conditional::ternary_bucket: 16.99x
- Element-wise::affine_transform: 61.47x
- Element-wise::clip: 35.65x
- Element-wise::scalar_add: 99.89x
- Element-wise::scalar_mul: 90.21x
- Linear Algebra::broadcast_add: 101.79x
- Linear Algebra::dot_1d: 99.35x
- Linear Algebra::matvec: 397.54x
- Math Functions::exp: 8.09x
- Math Functions::log1p_abs: 8.27x
- Math Functions::sin: 9.83x
- Math Functions::sqrt_abs: 59.29x
- Matrix Multiplication::matmul: 4430.47x
- Memory/Indexing::concat: 5.74x
- Memory/Indexing::copy_add: 94.72x
- Memory/Indexing::in_place_add: 208.34x
- Memory/Indexing::slice_contiguous: 1710.10x
- Memory/Indexing::slice_strided: 3204.54x
- Preprocessing::minmax_scale: 50.61x
- Preprocessing::rolling_mean_w5: 173.54x
- Preprocessing::zscore: 43.38x
- Preprocessing::zscore_clip: 37.98x
- Sorting/Ranking::sort_asc: 3.06x
- Sorting/Ranking::top10_desc: 3.07x

## Practical Checkpoints
- Threshold overhead line (scalar_add): N=100 5.8204x, N=10000 112.7246x, N=1000000 181.1353x
- Cache locality impact (sum_axis_1 vs sum_axis_0 at N=1000): axis1 14.9840x, axis0 248.8475x
- Allocation impact (in_place_add vs copy_add at N=1000000): in-place 289.6144x, copy 180.2145x
- Pipeline gain (rolling_mean_w5 at N=1000000): 277.1744x
- Ranking workload (top10_desc at N=200000): 4.0320x
