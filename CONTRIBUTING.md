# Contributing to oscfit

Thank you for your interest in contributing to `oscfit`! We welcome contributions from astrophysicists, solar physicists, scientific software engineers, and students.

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/oscfit.git
   cd oscfit
   ```

2. **Create a virtual environment and install dependencies in editable mode:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Verify the test suite:**
   ```bash
   pytest
   ```

## Development Workflow

1. Create a descriptive branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write clean, self-documenting Python code adhering to PEP 8 standards.
3. Include unit tests in `tests/` for any new functionality or bug fixes.
4. Ensure all tests pass with high coverage:
   ```bash
   pytest --cov=oscfit --cov-report=term-missing
   ```
5. Submit a Pull Request following the PR template.

## Scientific Validation

If adding new oscillation models or seismological derivations:
* Cite the relevant peer-reviewed literature in the docstring (Author Year, Journal).
* Include a validation test comparing analytical calculations against published benchmark values (similar to `tests/test_validation.py`).

## Questions or Support

Feel free to open an issue or reach out via [Discussions](https://github.com/Vibhav-Aggarwal/oscfit/discussions).
