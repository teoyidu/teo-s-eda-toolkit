# Teo's EDA Toolkit

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Spark Version](https://img.shields.io/badge/spark-3.4%2B-orange)](https://spark.apache.org/downloads.html)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/teo/teo-s-eda-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/teo/teo-s-eda-toolkit/actions)

A comprehensive, production-ready data quality framework for processing Parquet files using Apache Spark, with specialized support for Turkish NLP and Legal domain filtering.

## Features

- **Core Data Quality Processors**: Clean and validate missing values, categorical mapping, text validation, uniqueness, numeric formats, HTML, and Hadoop boilerplate.
- **Turkish NLP Support**: Includes advanced text analysis processors using BGE-M3 and SemHash for Turkish language duplicates and similarity.
- **Legal Domain Filtering**: Out-of-the-box legal document classification with PySpark UDFs.
- **Batch Processing**: Configurable batch operations to efficiently process large Parquet datasets without running out of memory.
- **Clean Architecture**: Built on a solid foundation with an extensible `BaseProcessor` class and a pipeline pattern.

## Installation

```bash
# Clone the repository
git clone https://github.com/teo/teo-s-eda-toolkit.git
cd teo-s-eda-toolkit

# Install via pip
pip install -e .
```

## Quick Start

```bash
# Process single file with default configuration
data-quality /path/to/your/data.parquet

# Or using python directly:
python -m data_quality /path/to/your/data.parquet
```

For more details, check out the documentation.

## Documentation
- [Documentation Home](https://teo.github.io/teo-s-eda-toolkit/)
- [API Reference](api.md)
- [Roadmap](roadmap.md)
- [Contributing](contributing.md)

## License
MIT License - see the LICENSE file for details.
