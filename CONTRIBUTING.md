# Contributing to 2A1B Solver

Thank you for your interest in contributing! This document describes how to set up your development environment and submit changes.

## Getting Started

### Prerequisites

- Python 3.10 or later
- - Git
  - - (Optional) Playwright for web bridge tests
   
    - ### Development Setup
   
    - ```bash
      # 1. Fork and clone the repository
      git clone https://github.com/RogueTex/2A1B_Solver.git
      cd 2A1B_Solver

      # 2. Create a virtual environment
      python -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

      # 3. Install in editable mode with test dependencies
      pip install -e .[test]

      # 4. (Optional) Install web bridge dependencies
      pip install -e .[web]
      playwright install chromium
      ```

      ## Running Tests

      ```bash
      # Run all tests
      pytest

      # Run with verbose output
      pytest -v

      # Run a specific test file
      pytest tests/test_env.py -v

      # Run with coverage
      pytest --cov=rl2048 --cov-report=term-missing
      ```

      ## Code Style

      This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting.

      ```bash
      # Lint
      ruff check rl2048/ tests/

      # Format
      ruff format rl2048/ tests/

      # Auto-fix linting issues
      ruff check --fix rl2048/ tests/
      ```

      ## Submitting Changes

      1. **Create a feature branch** from `main`:
      2.    ```bash
               git checkout -b feat/your-feature-name
               ```

            2. **Write your changes** and add tests if applicable.
        
            3. 3. **Ensure tests pass** and code is formatted:
               4.    ```bash
                        ruff check rl2048/ tests/
                        pytest
                        ```

                     4. **Commit** with a descriptive message following [Conventional Commits](https://www.conventionalcommits.org/):
                     5.    ```
                              feat: add new curriculum schedule type
                              fix: correct invalid-action penalty calculation
                              docs: update training examples in README
                              test: add edge cases for reward shaper
                              ```

                           5. **Open a Pull Request** against `main` with:
                           6.    - A clear description of what the change does
                                 -    - Reference to any related issues (e.g., `Closes #42`)
                                      -    - Screenshots or benchmark results if relevant
                                       
                                           - ## Project Structure
                                       
                                           - ```
                                             2A1B_Solver/
                                             ├── rl2048/                  # Core library
                                             │   ├── env_alphabet2048.py  # Gymnasium environment
                                             │   ├── dqn_agent.py         # DQN agent implementation
                                             │   ├── curriculum.py        # Curriculum learning
                                             │   ├── reward_shaper.py     # Reward shaping
                                             │   ├── diagnostics.py       # Decision diagnostics
                                             │   └── scripts/             # Module entrypoints
                                             ├── scripts/                 # Utility scripts
                                             ├── tests/                   # Test suite
                                             └── pyproject.toml           # Build & dependency config
                                             ```

                                             ## Reporting Issues

                                             Please open a GitHub issue with:
                                             - A clear description of the bug or feature request
                                             - - Steps to reproduce (for bugs)
                                               - - Expected vs. actual behavior
                                                 - - Python version and OS
                                                  
                                                   - ## License
                                                  
                                                   - By contributing, you agree that your contributions will be licensed under the same license as the project.
