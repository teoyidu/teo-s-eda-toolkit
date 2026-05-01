# Repository Analysis Report — Teo's EDA Toolkit

> **Objective:** Evaluate the codebase for production-readiness as an open-source data quality toolkit, and identify every area needing improvement.

---

## 1. Executive Summary

The repository contains a **PySpark-based data quality framework** with strong domain ambitions — 16+ processors, Turkish NLP integration, batch processing, a TUI (Textual) CLI, and legal domain classification via BERTurk. However, it is currently in a **prototype/alpha state** with significant structural, quality, and packaging issues that block production use and community adoption.

| Dimension | Current State | Target |
|---|---|---|
| **Architecture** | 🔴 Dual-framework, monolithic | Clean single-tree package |
| **Packaging** | 🔴 Broken/incomplete | PyPI-ready `pyproject.toml` |
| **Testing** | 🔴 ~4% coverage, scattered | ≥80% with CI gates |
| **Documentation** | 🟡 Aspirational README | Accurate + API docs |
| **Code Quality** | 🟡 Inconsistent, no linting CI | Enforced via pre-commit |
| **Dependencies** | 🔴 Bloated, dev ≡ prod | Layered extras |
| **Performance** | 🟡 Good ideas, untested | Benchmarked + Rust hot paths |
| **Security** | 🔴 No audit | Dependency scanning, no secrets |

---

## 2. Critical: Dual-Framework Architecture

This is the **single biggest problem**. Two completely independent implementations of `DataQualityFramework` exist:

| Location | Lines | Role |
|---|---|---|
| [data_quality_framework.py](file:///Users/teo/teo-s-eda-toolkit/teo-s-explotary-data-analysis-toolkit/data_quality_framework.py) | **1,747** | Root-level monolith with ALL processors inlined, plus embedded `unittest` test class |
| [src/data_quality/core/framework.py](file:///Users/teo/teo-s-eda-toolkit/teo-s-explotary-data-analysis-toolkit/src/data_quality/core/framework.py) | 419 | Properly structured package version using the processor registry |

Both define `DataQualityFramework`, `ConfigurationValidator`, `MetricsCollector`, and exception classes. Different modules import from different locations:
- `main.py` → imports from `data_quality_framework` (root monolith)
- `cli_processor.py` → imports from `src.data_quality.core.framework`
- `batch_processor.py` → imports from `data_quality_framework` (root monolith)

> [!CAUTION]
> **A user cannot reliably know which framework they're using.** This creates silent divergence where fixes applied to one are missing from the other. The root monolith also embeds ~150 lines of `unittest.TestCase` code that runs on import when used as `__main__`.

### Recommendation
- **Delete `data_quality_framework.py`** entirely
- Consolidate everything into the `src/data_quality/` package
- Update all imports in `main.py`, `batch_processor.py`, and any other consumers

---

## 3. Project Structure & Layout

### Current layout issues

```
Root Directory (36 files, 6 dirs)
├── 14 test_*.py files scattered in root    ← Not discoverable by pytest
├── example_*.py files in root              ← Should be in docs/examples/
├── turkish 3.txt                           ← Filename with space, fragile path lookup
├── data_quality.log                        ← Committed log file
├── test_hadoop_cleaner_input.parquet       ← Test fixtures committed to root
├── turkish_test_data.csv                   ← Test data in root
├── spark_truba.egg-info/                   ← Build artifact committed
├── __pycache__/                            ← Should be .gitignore'd (pattern issue)
├── .vscode/                                ← Should be .gitignore'd
├── setup.py + pyproject.toml               ← Dual packaging config
└── src/data_quality/config/
    └── xlsx_processing_config.json         ← 1 byte (empty file)
```

### Recommendations
1. **Adopt `src`-layout** consistently: `src/teo_eda/` (rename from `data_quality` to match the project identity)
2. Move all tests to `tests/` with proper `conftest.py` and `pytest` discovery
3. Move examples to `docs/examples/`
4. Move test fixtures to `tests/fixtures/`
5. Rename `turkish 3.txt` → `turkish_stopwords.txt` (no spaces)
6. Delete committed build artifacts (`spark_truba.egg-info/`, `__pycache__/`)
7. Remove committed log files (`data_quality.log`)

---

## 4. Packaging & Distribution

> [!WARNING]
> The package is currently **uninstallable from PyPI** and has conflicting metadata.

### Issues

| Issue | Detail |
|---|---|
| **Name mismatch** | `setup.py` says `spark-truba`, README says "Teo's EDA Emporium", directory is `teo-s-explotary-data-analysis-toolkit` (typo: "explotary") |
| **Dual config** | Both `setup.py` and `pyproject.toml` exist, but `pyproject.toml` only has tool configs, not project metadata |
| **Dev deps in install_requires** | `pytest`, `pytest-cov`, `black`, `flake8`, `mypy` are in `install_requires` — they should be `extras_require[dev]` |
| **Heavy deps forced** | `torch>=2.0.0` (~2GB) is a hard requirement even if user only needs CSV cleaning |
| **No `__main__.py` at package root** | The entry point `data-quality=data_quality.__main__:main` won't resolve because the package is under `src/` |
| **Version** | `setup.py` says `0.1.0`, `__init__.py` says `1.0.0` |
| **Missing LICENSE file** | README claims MIT but no `LICENSE` file exists |
| **Missing CHANGELOG** | No changelog |

### Recommendations
1. **Single source of truth**: Migrate everything to `pyproject.toml` with `[project]` table, delete `setup.py`
2. **Fix the name**: Choose one canonical name (e.g. `teo-eda-toolkit` or `spark-truba`)
3. **Layered extras**:
   ```toml
   [project.optional-dependencies]
   spark = ["pyspark>=3.4.0"]
   nlp = ["torch>=2.0.0", "transformers>=4.30.0", "sentencepiece>=0.1.99"]
   turkish = ["FlagEmbedding>=1.2.0", "semhash>=0.3.0", "nltk>=3.8.1"]
   viz = ["matplotlib>=3.5.3", "seaborn>=0.12.2", "plotly>=5.13.0"]
   dev = ["pytest>=7.0", "pytest-cov>=4.0", "black>=22.0", "ruff>=0.4", "mypy>=1.0"]
   all = ["teo-eda-toolkit[spark,nlp,turkish,viz]"]
   ```
4. **Add `LICENSE` file** (MIT)
5. **Add `CHANGELOG.md`** (Keep a Changelog format)
6. **Unify version** — use `importlib.metadata` or a single `__version__` source

---

## 5. Dependency Concerns

### Excessive/Problematic Dependencies

| Dependency | Issue |
|---|---|
| `torch>=2.0.0` | ~2GB install, only needed for legal domain filter |
| `pathlib2>=2.3.7` | Unnecessary — `pathlib` is stdlib since Python 3.4 |
| `typing-extensions>=4.0.0` | Only needed if supporting Python <3.10 |
| `py4j>=0.10.9` | Transitive dep of PySpark, should not be explicit |
| `textual>=0.40.0` | Only needed for TUI, not core functionality |
| `FlagEmbedding>=1.2.0` | Heavy ML dep, already optional via try/except but still in requirements.txt |
| `dataclasses-json>=0.5.7` | Not used anywhere in the codebase |

### No version pinning
- `requirements.txt` uses only `>=` — no upper bounds or lock file
- No `requirements-lock.txt` or `pip-compile` output

### Recommendations
1. Remove `pathlib2`, `py4j`, `dataclasses-json`
2. Make `torch`, `transformers`, `FlagEmbedding`, `semhash` optional extras
3. Add a `uv.lock` or `pip-compile` lock file for reproducible environments
4. Set Python minimum to 3.10+ (drop 3.8/3.9 — they're EOL)

---

## 6. Code Quality Issues

### 6.1 Monolithic Files

| File | Lines | Problem |
|---|---|---|
| `data_quality_framework.py` | 1,747 | Entire framework + tests in one file |
| `boilerplate_cleaner.py` | 726 | Single class with 8 methods, could be split |
| `main.py` | 514 | Pipeline runner mixed with Spark session management |
| `config_manager.py` | 595 | Config schema duplicated from `data_quality_framework.py` |

### 6.2 Code Duplication

- **Configuration schema**: Defined 3 times — in `data_quality_framework.py`, `config_manager.py`, and `src/data_quality/utils/config_validator.py`
- **Exception classes**: Defined in `data_quality_framework.py`, `src/data_quality/exceptions.py`, `src/data_quality/core/exceptions.py`, and `src/legal_domain_filter.py` — four separate hierarchies
- **`create_sample_config`**: Defined in both `config_manager.py` and `src/data_quality/__main__.py`
- **Spark session creation**: Done in `main.py`, `cli_processor.py`, `batch_processor.py`, and `src/data_quality/__main__.py`
- **`LegalDomainFilter`**: Two versions — `src/legal_domain_filter.py` and `src/data_quality/utils/legal_domain_filter.py`

### 6.3 Anti-patterns

1. **Bare `except:` clauses** in `boilerplate_cleaner.py` (lines 358, 364) — swallows all errors silently
2. **`import unittest` at module level** in `data_quality_framework.py` — test code mixed with production code
3. **Mutable default arguments** — avoided but could be enforced via linting
4. **Hardcoded paths**: `/tmp/dq_checkpoints`, `/tmp/dq_output` appear as defaults in multiple places
5. **No abstract base class** for processors — each processor has its own `process()` signature with no contract
6. **`time.sleep(0.1)` in progress bar loop** in `cli_processor.py` — blocks the UI thread
7. **Background threads without shutdown** in `LoggingManager` — `_monitor_system_resources` runs `while True` with no stop flag

### 6.4 Type Safety

- `pyproject.toml` configures strict `mypy` (`disallow_untyped_defs = true`) but it's never run in CI
- Many functions use `Dict` instead of `TypedDict` or dataclasses
- `config` is always `Dict[str, Any]` — no schema enforcement at the type level

### Recommendations
1. Define a `BaseProcessor` ABC with `process(df) -> Tuple[DataFrame, Dict]`
2. Single exception hierarchy in `src/data_quality/exceptions.py`
3. Single config schema in `src/data_quality/config/schema.py`
4. Run `ruff` + `mypy` in CI with pre-commit hooks
5. Fix all bare `except:` clauses

---

## 7. Testing

### Current State

| Category | Count | Notes |
|---|---|---|
| Files in `tests/` | 2 | `test_hadoop_cleaner.py`, `test_legal_domain_filter.py` |
| Files in root | 14 | `test_*.py` — not discoverable by standard pytest |
| Tests embedded in source | 1 | `data_quality_framework.py` contains `TestDataQualityFramework` |
| Estimated coverage | **<5%** | Most processors have zero tests |

### Issues
1. **No `conftest.py`** — no shared Spark session fixture (each test file creates its own)
2. **No CI** — tests never run automatically
3. **Tests use Pandas** but processors operate on either Spark or Pandas DataFrames inconsistently
4. **No integration tests** — no end-to-end pipeline test
5. **No property-based testing** for data quality rules
6. **Test data committed to root** — `.parquet` and `.csv` files should be in fixtures

### Recommendations
1. Move all tests to `tests/` with subdirectories: `tests/unit/`, `tests/integration/`
2. Create `conftest.py` with a session-scoped Spark fixture
3. Target **≥80% line coverage** as a CI gate
4. Add `hypothesis`-based property tests for numerical/text processors
5. Add integration test: generate synthetic data → run full pipeline → validate output

---

## 8. Processor Architecture

### Current processor inventory

| Processor | Lines | Has Tests | Interface |
|---|---|---|---|
| `MissingValuesProcessor` | 75 | ❌ | `process(df)` |
| `MandatoryFieldsProcessor` | 42 | ❌ | `process(df)` |
| `NumericalFormatsProcessor` | 75 | ❌ | `process(df)` |
| `OutdatedDataProcessor` | 51 | ❌ | `process(df)` |
| `ExternalValidationProcessor` | 96 | ❌ | `process(df)` |
| `UniquenessProcessor` | 68 | ❌ | `process(df)` |
| `CategoriesProcessor` | 59 | ❌ | `process(df)` |
| `TextValidationProcessor` | 98 | ❌ | `process(df)` |
| `RelationshipsProcessor` | 101 | ❌ | `process(df)` |
| `EntryRulesProcessor` | 114 | ❌ | `process(df)` |
| `HTMLCleanerProcessor` | 121 | ❌ | `process(df)` |
| `HadoopCleanerProcessor` | 139 | ✅ (Pandas) | `process(df)` |
| `BoilerplateCleanerProcessor` | 726 | ✅ (partial) | `process(df)` |
| `DuplicateDetector` | 314 | ❌ | `process(df)` |
| `SemHashTurkishDetector` | 368 | ❌ | `process(df)` |
| `BGEM3TurkishDetector` | 487 | ❌ | `process(df)` |
| `NoiseDetectionProcessor` | 121 | ❌ | Not registered |
| `XLSXProcessor` | 141 | ❌ | `process_file(path)` |

### Issues
1. **No `BaseProcessor` contract** — each has `process()` but return types differ
2. **Pandas vs Spark inconsistency**: `HadoopCleanerProcessor`, `BoilerplateCleanerProcessor`, `HTMLCleanerProcessor` take Pandas DataFrames. Core processors take Spark DataFrames. **Users have no way to know which type a processor expects.**
3. **`NoiseDetectionProcessor` exists** but is never registered in the processor factory
4. **No plugin system** — adding a custom processor requires editing `__init__.py`
5. **Pipeline is hardcoded** — processor order is fixed, users can't skip/reorder processors

### Recommendations
1. Define `BaseProcessor(ABC)` with:
   ```python
   @abstractmethod
   def process(self, df: DataFrame) -> Tuple[DataFrame, ProcessorStats]: ...

   @abstractmethod
   def validate_config(self, config: Dict) -> List[str]: ...
   ```
2. Standardize on **one DataFrame type** (Spark) and convert Pandas-based processors
3. Implement a **pipeline builder** pattern so users can compose and reorder processors
4. Add an `entry_points`-based plugin system for community processors

---

## 9. Configuration System

### Issues
1. **Schema duplication** (3 copies, see §6.2)
2. **No environment variable support** — README claims it but it's not implemented
3. **No config merging** — can't overlay environment-specific configs on base configs
4. **`DataQualityConfig` dataclass** in `config_manager.py` duplicates schema fields and has no validation
5. **Empty config file**: `src/data_quality/config/xlsx_processing_config.json` is 1 byte

### Recommendations
1. Single schema definition using Pydantic v2 `BaseModel` with validators
2. Support `TOML` config files (Python's native config format since 3.11)
3. Implement proper config layering: defaults → file → env vars → CLI args
4. Delete the empty config file or populate it

---

## 10. Documentation

### Issues

| Issue | Detail |
|---|---|
| **README inflation** | 1,030 lines listing features that aren't implemented (e.g., "Network connection monitoring", "Thread pool stats", "Handle leaks detection") |
| **No API docs** | No Sphinx/MkDocs, no hosted documentation |
| **README project structure** | Shows directories that don't exist (`src/data_quality/tests/`) |
| **Badges are fake** | Build status badge is hardcoded `passing`, docs badge links to non-existent `docs/` |
| **No CONTRIBUTING.md** | README has contributing section but it's generic |
| **Typo in repo name** | "explotary" should be "exploratory" |

### Recommendations
1. **Rewrite README** to accurately describe what exists today
2. Remove all aspirational features or move them to a `ROADMAP.md`
3. Set up **MkDocs** with `mkdocstrings` for auto-generated API docs
4. Add real CI badges (GitHub Actions)
5. Add `CONTRIBUTING.md` with development setup, code style, PR process
6. Fix the typo in the repository name

---

## 11. CI/CD & DevOps

### Current State: **None**

No `.github/workflows/`, no `Makefile`, no `tox.ini`, no `noxfile.py`.

### Recommendations

1. **GitHub Actions** with:
   - `lint.yml`: ruff + mypy on every PR
   - `test.yml`: pytest with coverage gate (≥80%)
   - `publish.yml`: PyPI publish on tag
   - `security.yml`: `pip-audit` or Dependabot
2. **Pre-commit hooks** (`.pre-commit-config.yaml`):
   - ruff (replaces flake8, isort, black in one tool)
   - mypy
   - trailing whitespace, EOF fixer
3. **Makefile** or **justfile** for common commands:
   ```makefile
   lint:    ruff check . && mypy src/
   test:    pytest tests/ --cov=src/
   docs:    mkdocs serve
   build:   python -m build
   ```

---

## 12. Performance & Scalability

### Current Strengths
- Good Spark configuration defaults (AQE, vectorized reader, Arrow)
- `BatchProcessor` with configurable batch sizes
- `BatchOptimizer` for dataset-aware recommendations
- Memory pressure handling in `MetricsCollector`

### Issues
1. **Multiple `df.count()` calls** in the pipeline — each triggers a full Spark action. The pipeline calls `.count()` up to 30+ times for a single file processing
2. **No lazy evaluation strategy** — processors materialize intermediate results
3. **Batch creation is wrong**: `_create_batches()` creates batch definitions with `start_idx`/`end_idx` but never uses them to actually slice the DataFrame. Batches reference non-existent `dataframe` key
4. **No benchmarks** — performance claims in README are untested
5. **Thread-safety issues** in `LoggingManager`: background threads access shared mutable lists without locks

### Rust/Julia Acceleration Opportunities

These are compute-intensive hot paths where Rust or Julia extensions would provide significant speedups:

| Hot Path | Current Impl | Recommended Acceleration |
|---|---|---|
| **TF-IDF cosine similarity** for duplicate detection | Python scikit-learn | **Rust** via PyO3 — sparse matrix cosine similarity is 5-10x faster |
| **Regex pattern matching** for boilerplate cleaning (60+ patterns) | Python `re` module | **Rust** via `regex` crate + PyO3 — compiled regex set, 3-20x faster |
| **Text normalization** (Unicode NFKC, Turkish char handling) | Python `unicodedata` | **Rust** via `unicode-normalization` crate — batch processing, zero-copy |
| **MinHash/LSH** for near-duplicate detection | Python `datasketch` | **Rust** — custom MinHash with SIMD, 10-50x faster for large corpora |
| **Levenshtein distance** for fuzzy matching | PySpark `levenshtein` | **Rust** via `strsim` crate for Pandas path, JVM stays for Spark path |

> [!TIP]
> Start with a Rust extension for regex-based boilerplate cleaning — it's self-contained, measurable, and the `regex` crate supports building a compiled set of 60+ patterns that matches in a single pass instead of 60 sequential `re.sub()` calls.

---

## 13. Security Considerations

| Issue | Severity | Detail |
|---|---|---|
| Hardcoded `/tmp` paths | Medium | Default checkpoint/output dirs are world-writable |
| No dependency scanning | Medium | No `pip-audit`, Dependabot, or Snyk |
| Log files contain data | Low | `data_quality.log` committed to repo |
| `author_email` placeholder | Low | `your.email@example.com` in `setup.py` |
| No input sanitization | Medium | File paths from CLI are not sanitized against path traversal |

---

## 14. Turkish NLP Domain — Specific Issues

The Turkish NLP features are the most unique part of this toolkit, but have issues:

1. **Stopwords file has a space in the name** (`turkish 3.txt`) — fragile on Windows and in CI
2. **BGE-M3 model download on first use** — no offline/bundled option, no progress indicator
3. **`lru_cache` on `_is_legal_domain`** in `LegalDomainFilter` — cache is unbounded in practice (maxsize=1000 but texts are hashed, so near-duplicate texts get separate cache entries)
4. **UDF serialization**: `LegalDomainFilter.process()` registers UDFs that capture `self` — this sends the entire model (~1GB) to each Spark executor. Should use `pandas_udf` with broadcast variables or process on driver
5. **No Turkish tokenizer benchmark** — NLTK's `punkt` is trained on English, may not tokenize Turkish correctly
6. **SemHash integration** is incomplete — `self.semhash_model = True` is a boolean flag, not a model instance

---

## 15. Prioritized Improvement Roadmap

### Phase 1: Foundation (1-2 weeks)
- [x] Fix repository name typo
- [x] Delete `data_quality_framework.py` — consolidate into `src/data_quality/`
- [x] Delete duplicate exception classes, config schemas, sample configs
- [x] Fix all imports to use the single package
- [x] Rename `turkish 3.txt` → `turkish_stopwords.txt`
- [x] Add `LICENSE` file
- [x] Fix `.gitignore` and clean committed artifacts
- [x] Migrate to `pyproject.toml` only (delete `setup.py`)
- [x] Remove dev dependencies from `install_requires`

### Phase 2: Quality Gates (1-2 weeks)
- [x] Set up GitHub Actions CI (lint, test, type-check)
- [x] Add pre-commit hooks (ruff, mypy)
- [x] Move all tests to `tests/` directory
- [x] Add `conftest.py` with shared Spark session fixture
- [x] Write tests for all 16 processors (target ≥80% coverage)
- [x] Fix bare `except:` clauses
- [x] Fix background thread shutdown in `LoggingManager`

### Phase 3: Architecture (2-3 weeks)
- [x] Define `BaseProcessor` ABC
- [x] Standardize all processors on Spark DataFrames
- [x] Implement pipeline builder pattern
- [x] Implement proper config system (Pydantic + layering)
- [x] Fix `BatchProcessor._create_batches()` — actually slice DataFrames
- [x] Reduce `.count()` calls in pipeline (cache counts, use accumulators)
- [x] Fix UDF serialization in `LegalDomainFilter`

### Phase 4: Polish & Distribute (1-2 weeks)
- [x] Rewrite README to match reality
- [x] Set up MkDocs with API documentation
- [x] Add `CONTRIBUTING.md` and `CHANGELOG.md`
- [x] Implement dependency extras (spark, nlp, turkish, viz, dev)
- [ ] First PyPI release
- [ ] Add real CI badges

### Phase 5: Performance (2-4 weeks)
- [ ] Build Rust extension for regex-based boilerplate cleaning
- [ ] Build Rust extension for TF-IDF cosine similarity
- [ ] Add benchmark suite (`pytest-benchmark`)
- [ ] Profile and optimize `.count()` usage
- [ ] Add memory profiling tests

---

> [!IMPORTANT]
> The most impactful single change is **deleting `data_quality_framework.py`** and consolidating into the `src/data_quality/` package. This eliminates the dual-framework problem, removes ~1,700 lines of duplicated code, and makes every subsequent improvement cleaner.
