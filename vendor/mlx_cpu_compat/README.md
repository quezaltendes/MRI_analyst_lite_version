# MLX CPU build compatibility

Upstream: https://github.com/ml-explore/mlx, tag v0.32.4, commit
`36969c094354086e15d2a327a9c50feb20b9cf11`.

A CPU-only source build is used for the restricted macOS session where Metal is
unavailable. `appleclang14.patch` makes two source-level compatibility changes for
Apple Clang 14.0.3: replace captured structured bindings with ordinary tuple
extraction, and make a dependent type explicitly `typename`. Numerical operations,
model weights and quantization remain unchanged. Apply with `git apply` in the
upstream checkout before building with `MLX_BUILD_METAL=OFF` and `MLX_BUILD_CUDA=OFF`.

MLX's upstream MIT license is included in `LICENSE`. This is not required for the
standard Metal-enabled MLX wheel on a supported native macOS process.
