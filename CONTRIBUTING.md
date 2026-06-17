# Contributing to Medical Data Analyser

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions with other contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/medical-data-analyser.git
   cd medical-data-analyser
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Development Workflow

### Creating a Branch

Create a new branch for your feature or fix:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Use descriptive branch names:
- `feature/add-new-model` - for new features
- `fix/correct-data-processing` - for bug fixes
- `docs/improve-readme` - for documentation

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function arguments and returns
- Keep functions small and focused
- Write descriptive docstrings

### Running Tests

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src
```

Run specific test file:
```bash
pytest tests/test_data_processor.py
```

### Code Quality

Format code with Black:
```bash
black src/ tests/
```

Check for style issues:
```bash
flake8 src/ tests/
```

Type checking:
```bash
mypy src/
```

## Making Changes

1. **Write your code** following the style guidelines
2. **Add/update tests** for new functionality
3. **Update documentation** if needed
4. **Run tests** to ensure everything passes

## Committing Changes

Write clear, descriptive commit messages:
```bash
git add .
git commit -m "feat: add new disease prediction model"
```

Commit message format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for tests
- `refactor:` for code refactoring
- `style:` for formatting
- `perf:` for performance improvements

## Submitting a Pull Request

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub with:
   - Clear title describing the changes
   - Description of what was changed and why
   - Reference any related issues (e.g., "Fixes #123")
   - Screenshots or examples if applicable

3. **Wait for review** and address any feedback

## Pull Request Checklist

- [ ] Code follows PEP 8 style guidelines
- [ ] All tests pass (`pytest`)
- [ ] New tests added for new functionality
- [ ] Code coverage not decreased
- [ ] Documentation updated
- [ ] Commit messages are descriptive
- [ ] Branch is up to date with main

## Reporting Issues

When reporting bugs, include:
- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/traceback

## Questions?

Feel free to open an issue for discussion or contact the maintainers.

---

**Thank you for contributing!**
