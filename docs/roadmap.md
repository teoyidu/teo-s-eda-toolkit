# Roadmap

The following features are aspirational and planned for future releases of Teo's EDA Toolkit.

## Data Quality Processors
- **HTMLCleanerProcessor Enhancements**
  - Advanced XSS protection and sanitization capabilities.
  - Whitelist-based tag filtering with configurable rules.
- **NoiseDetectionProcessor**
  - Register and integrate `NoiseDetectionProcessor` into the pipeline.

## Metrics & Monitoring
- **Advanced Resource Monitoring**
  - Thread pool stats (Active threads, Blocked threads).
  - Open file handles tracking (File descriptor limits, Handle leaks detection).
  - Network connection monitoring (Active connections, Connection pools, Network errors).
- **Garbage Collection Statistics**
  - Collection frequency.
  - Memory reclaimed.
  - Pause times.

## Performance & Scalability
- **Rust/Julia Acceleration**
  - Build Rust extension via PyO3 for TF-IDF cosine similarity to accelerate duplicate detection.
  - Build Rust extension for regex-based boilerplate cleaning (matching 60+ patterns in a single pass).
  - Build Rust extension for MinHash/LSH near-duplicate detection.
- **Profiling & Benchmarks**
  - Add comprehensive benchmark suite (`pytest-benchmark`).
  - Profile and optimize `.count()` usage in the Spark pipeline.

## Configuration & Architecture
- **Environment Support**
  - Environment variable support for configurations.
  - Config merging (overlay environment-specific configs on base configs).
- **Extensibility**
  - `entry_points`-based plugin system for community processors.

## Deployment & CI/CD
- **Publishing**
  - First PyPI release.
  - Layered dependency extras (spark, nlp, turkish, viz, dev).
