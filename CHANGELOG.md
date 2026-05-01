# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-01

### Added
- Created `pyproject.toml` for standard Python packaging, replacing `setup.py`.
- Added `CONTRIBUTING.md` and `CHANGELOG.md` files.
- Layered dependency groups (extras) for `spark`, `nlp`, `turkish`, `viz`, and `dev` environments.
- Defined `BaseProcessor` abstract base class to enforce standardized processor contracts.
- Configured MkDocs to auto-generate API documentation using `mkdocstrings`.
- GitHub Actions CI workflows for linting, testing, and type-checking.
- Pytest support with `conftest.py` providing session-scoped Spark fixtures.
- Pipeline builder pattern implementation for easier user interaction.

### Changed
- Major repository restructure following the `src` layout.
- Consolidated `data_quality_framework.py` into a robust `src/data_quality` package, effectively eliminating the dual-framework discrepancy.
- Standardized all processor inputs to PySpark DataFrames, converting any previously Pandas-based logic.
- Rewrote `README.md` to reflect current capabilities realistically.
- Transitioned configuration logic to Pydantic v2 `BaseModel` schemas to reduce redundancy and enforce strict validation.
- Improved DataFrame batch creation performance by utilizing accurate bounds slicing rather than arbitrary index mappings.
- Restructured `dependencies` allowing for lightweight core installation.

### Removed
- `data_quality_framework.py` root monolith script.
- Deleted obsolete exceptions duplicated across different files.
- Removed dummy and placeholder configuration copies (`xlsx_processing_config.json`).
- Development and heavy testing libraries removed from base `dependencies`.

### Fixed
- Addressed silent failure conditions by replacing broad `except:` catch blocks with specific exceptions.
- Properly implemented a shutdown event for `LoggingManager` background threads to prevent zombie processes.
- Fixed `LegalDomainFilter` UDF serialization limits avoiding passing oversized classes onto Spark executors.
- Removed duplicated .count() evaluation calls from processors that caused severe PySpark performance degradation.

