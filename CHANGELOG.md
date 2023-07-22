# Changelog for gpt-tfjs

## 0.0.5 - 2023-07-22

### Added

- `textstream` project: Train GPT model using text stream in browser in node (without loading the whole dataset into memory)
- Support both `maxIter` and `epochs` in the `train` method. Switched from `df.forEachAsync` to `while` loop + `ds.iterator()` (it was not clear how to terminate long-running `forEachAsync`).

### Fixed

- Memory leakage in the `generate` method
- Use `config.lr` in the Adam optimizer
