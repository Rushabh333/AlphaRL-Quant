# Contributing to AlphaRL-Quant

Thank you for considering contributing to AlphaRL-Quant! This document provides guidelines and best practices.

##  Ways to Contribute

- **Bug Reports**: Found an issue? Open a GitHub issue with reproduction steps
- **Feature Requests**: Have an idea? Propose it in Discussions
- **Code Contributions**: Submit pull requests following our guidelines
- **Documentation**: Improve README, docstrings, or create tutorials
- **Testing**: Add unit tests, integration tests, or performance benchmarks

---

##  Development Setup

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/AlphaRL-Quant.git
cd AlphaRL-Quant
```

###2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists, otherwise use requirements.txt
```

### 4. Set Up Pre-Commit Hooks

```bash
pre-commit install  # Auto-formats code before commits
```

---

##  Code Standards

### Style Guide

- **Formatting**: Use `black` (line length: 100)
- **Imports**: Use `isort` for automatic sorting
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all classes/functions

### Example

```python
from pathlib import Path
from typing import List, Optional

def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Trading periods per year (default: 252)
    
    Returns:
        Annualized Sharpe ratio
    
    Raises:
        ValueError: If returns list is empty
    
    Example:
        >>> returns = [0.01, -0.005, 0.02, 0.015]
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe: {sharpe:.2f}")
    """
    if not returns:
        raise ValueError("Returns list cannot be empty")
    
    # Implementation using numpy for efficiency
    import numpy as np
    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / periods_per_year)
    
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
```

---

##  Testing Requirements

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src tests/

# Specific module
pytest tests/test_trading_env.py

# Watch mode (re-run on file changes)
pytest-watch
```

### Writing Tests

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test component interactions
- **Property-based tests**: Use `hypothesis` for edge cases

Example:

```python
import pytest
from src.models.trading_env import TradingEnvironment

def test_environment_reset():
    """Test that environment resets to initial state."""
    import pandas as pd
    
    # Arrange
    data = pd.read_csv('tests/fixtures/sample_data.csv')
    env = TradingEnvironment(data, initial_capital=10000)
    
    # Act
    obs, info = env.reset()
    
    # Assert
    assert env.cash == 10000, "Cash should reset to initial capital"
    assert env.shares == 0, "Shares should be zero after reset"
    assert obs.shape == (40,), f"Observation shape should be (40,), got {obs.shape}"
```

---

##  Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write code following style guidelines
- Add tests for new functionality
- Update documentation (README, docstrings)
- Run tests locally

### 3. Commit with Conventional Commits

```bash
git commit -m "feat: add Sortino ratio reward function"
git commit -m "fix: correct MultiIndex column handling in data collector"
git commit -m "docs: update README with new feature examples"
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### 4. Push & Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- **Clear title**: Summarize the change
- **Description**: Explain what, why, and how
- **Tests**: Confirm all tests pass
- **Screenshots**: If UI/visualization changes

---

##  Performance Considerations

### Profiling

Before submitting performance improvements, profile your code:

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = expensive_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 slowest functions
```

### Benchmarking

Use `pytest-benchmark` for quantitative comparisons:

```python
def test_feature_engineering_performance(benchmark):
    """Benchmark feature engineering speed."""
    data = create_sample_data(n_rows=10000)
    
    result = benchmark(engineer_features, data)
    
    assert len(result) > 0
    # Ensure it runs in < 1 second for 10K rows
    assert benchmark.stats['mean'] < 1.0
```

---

##  Visualization Standards

When adding plots/visualizations:

- Use `matplotlib` with `seaborn` styling
- Include axis labels, title, and legend
- Save to `reports/` directory
- Add figure to README with caption

Example:

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def plot_training_curve(rewards: List[float], save_path: str = "reports/training_curve.png"):
    """Plot episode rewards over training."""
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.6, label='Episode Reward')
    plt.plot(pd.Series(rewards).rolling(100).mean(), label='100-Episode MA', linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

---

##  Bug Report Template

When reporting bugs, include:

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run `python scripts/run_pipeline.py`
2. Observe error at line X

**Expected Behavior**
What you expected to happen

**Environment**
- OS: macOS 13.2
- Python: 3.10.8
- Dependencies: (output of `pip freeze`)

**Logs**
```
Paste relevant logs here
```

**Additional Context**
Any other relevant information
```

---

##  Code Review Checklist

Reviewers should verify:

- [ ] Code follows style guide (`black`, `isort`)
- [ ] All tests pass (`pytest`)
- [ ] New features have tests (>80% coverage)
- [ ] Documentation updated
- [ ] No hardcoded values (use `config.py`)
- [ ] Type hints present
- [ ] Performance impact assessed
- [ ] No security vulnerabilities introduced

---

##  Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Writing Good Tests](https://docs.pytest.org/en/stable/goodpractices.html)

---

##  Community

- **Discussions**: Ask questions, share ideas
- **Discord**: (Add link if exists)
- **Weekly Office Hours**: (Add schedule if applicable)

---

##  License

By contributing, you agree that your contributions will be licensed under the same MIT License.

---

**Thank you for making AlphaRL-Quant better!** 
