# Contributing to Teo's EDA Toolkit

First off, thanks for taking the time to contribute! 🎉

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/teo/teo-s-eda-toolkit.git
   cd teo-s-eda-toolkit
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style & Linting

This project uses `ruff` for linting and formatting, and `mypy` for static type checking. We enforce a clean, readable codebase.

- **Run linters:**
  ```bash
  ruff check .
  ruff format .
  mypy src/
  ```

- **Pre-commit hooks:**
  We recommend setting up pre-commit hooks to automate formatting and linting before you commit:
  ```bash
  pip install pre-commit
  pre-commit install
  ```

## Testing

We use `pytest` for testing. Ensure your code is thoroughly tested before submitting a PR.
Our CI enforces a minimum of 80% test coverage.

- **Run tests:**
  ```bash
  pytest tests/
  ```

- **Run tests with coverage:**
  ```bash
  pytest tests/ --cov=src/
  ```

## Pull Request Process

1. Create a new branch for your feature or bugfix (`git checkout -b feature/my-feature`).
2. Make your changes and ensure tests pass and code is formatted.
3. Commit your changes with clear, descriptive commit messages.
4. Push to your fork and submit a Pull Request.
5. Await review from maintainers.
