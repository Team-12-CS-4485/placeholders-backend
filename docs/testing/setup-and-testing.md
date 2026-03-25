# Local Development and Testing Guide

This document explains how to set up your local development environment, run tests, and understand the CI/CD pipeline for the Placeholders Backend.

## 1. Environment Setup

The project uses two dependency files to keep the production environment lean:

- **`requirements.txt`**: Core dependencies needed to run the application (FastAPI, AI libraries, Database clients).
- **`requirements-dev.txt`**: Tools for development and testing (Pytest, Black, Flake8).

### Installation
To install everything needed for development:
```bash
make install
```
*Alternatively: `pip install -r requirements.txt -r requirements-dev.txt`*

---

## 2. Local Development with `Makefile`

A `Makefile` is provided to simplify common tasks across different operating systems.

### Available Commands
- `make install`: Installs all production and development dependencies.
- `make test`: Executes the `pytest` suite.
- `make lint`: Runs `flake8` to check for syntax errors and style violations.
- `make format`: Automatically formats code using `black` and `isort`.
- `make run`: Starts the FastAPI development server with auto-reload.
- `make docker-build`: Validates the `Dockerfile` by building the image locally.

### Platform-Specific Notes
- **Mac/Linux:** `make` is pre-installed.
- **Windows:** You can install `make` via [Chocolatey](https://chocolatey.org/) (`choco install make`) or [Winget](https://github.com/microsoft/winget-cli) (`winget install ezwinmake`). Alternatively, these commands can be run manually as defined in the `Makefile`.

---

## 3. Automated Testing

### Test Structure
Tests are located in the `tests/` directory:
- `tests/api/`: Functional tests for API endpoints (e.g., `test_health.py`).
- `tests/integration/`: Tests for multi-service interactions.
- `tests/services/`: Unit tests for internal logic (Chunking, Embeddings).

### Running Tests
To run all tests:
```bash
make test
```

To run a specific test file:
```bash
pytest tests/api/test_health.py
```

---

## 4. CI/CD Pipeline (GitHub Actions)

Every time you push code or open a Pull Request to `main` or `master`, the GitHub Actions workflow (`.github/workflows/ci.yml`) automatically triggers:

1.  **Environment Setup:** Installs Python 3.11 and all dependencies.
2.  **Linting:** Runs `flake8` and `black --check`. The build will fail if there are critical linting errors.
3.  **Testing:** Runs the full `pytest` suite.
4.  **Docker Build:** Verifies that the application can be successfully containerized.

**Note:** Ensure all tests pass locally (`make test`) and code is formatted (`make format`) before pushing to avoid CI failures.
