# Contributing to TrustLens

## 🚀 First Contribution (Start Here)

We love welcoming new contributors! Whether you're a seasoned engineer or just starting out, there's a place for you here. We have many small "quick-win" tasks that take only 10–20 minutes to complete.

Look for issues labeled:
- `beginner`
- `good first issue`
- `hacktoberfest`

### Quick Start in 5 Steps:

1. **Fork** the repository to your own GitHub account.
2. **Clone** your fork locally: `git clone https://github.com/<YOUR_USERNAME>/trustlens.git`.
3. **Create a branch** for your changes: `git checkout -b my-new-feature`.
4. **Make your changes** (even small ones like fixing a typo are great!).
5. **Open a Pull Request (PR)**
Don’t worry if something feels unclear - just open a PR and we’ll help you refine it 👍

---

## 🎯 Choose Your Contribution Level

Pick a path that matches your interest and experience:

🟢 **Beginner**:
- Fix typos or improve documentation.
- Add logging to existing functions.
- Improve error messages for better clarity.

🟡 **Intermediate**:
- Enhance CLI arguments or outputs.
- Integrate with new ML frameworks or logging tools.
- Implement a standard metric from our roadmap.

🔵 **Advanced**:
- Implement new XAI algorithms (e.g., specific Grad-CAM variants).
- Add complex representation metrics.
- Optimize performance for large-scale datasets.

---

## 🤝 New Contributors Welcome

👉 Want to get started right now? Check out open issues labeled `good first issue`.
Browse issues here: https://github.com/Khanz9664/TrustLens/issues

We are committed to making TrustLens a friendly and supportive community:
- **We actively support first-time contributors**: Don't worry if you're new to Git or Open Source.
- **You won’t break anything**: All changes are reviewed, and our automated tests will catch any issues.
- **Maintainers will guide you**: If you're stuck, just ask in your PR or an issue.
- **Fast PR reviews**: We aim to review all contributions within 48 hours.

---

Thank you for your interest in TrustLens!
We welcome contributions from researchers, engineers, and data scientists of all skill levels.

> **First time contributing to open source?**
> Check out [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/) — we're beginner-friendly!

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Adding a New Metric](#adding-a-new-metric)
4. [Adding a New Visualization](#adding-a-new-visualization)
5. [Writing a Plugin](#writing-a-plugin)
6. [Coding Standards](#coding-standards)
7. [Writing Tests](#writing-tests)
8. [Pull Request Guidelines](#pull-request-guidelines)
9. [Reporting Issues](#reporting-issues)

---

## 1. Development Setup

This is only needed if you're making code changes. For small doc fixes, you can edit directly on GitHub.

### Prerequisites

- Python 3.9 or higher
- Git

### Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/Khanz9664/trustlens.git
cd trustlens
```

### Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### Install in Editable Mode with Dev Dependencies

```bash
pip install -e ".[dev]"
```

### Install Pre-commit Hooks

```bash
pre-commit install
```

Pre-commit will automatically run linting and formatting checks on every `git commit`.

### Verify Installation

```bash
pytest --tb=short
python -c "from trustlens import analyze; print(' TrustLens ready')"
```

---

## 2. Project Structure

```
trustlens/
 api.py          ← analyze() entry point
 report.py         ← TrustReport result container
 utils.py         ← shared helpers
 metrics/
  calibration.py    ← Brier Score, ECE, reliability curve
  failure.py      ← misclassification, confidence gap
  bias.py        ← imbalance, subgroup performance
  representation.py  ← silhouette, CKA
 explainability/
  gradcam.py      ← Grad-CAM (PyTorch)
  faithfulness.py   ← pixel deletion/insertion tests
 visualization/
  calibration_plots.py
  failure_plots.py
  bias_plots.py
  representation_plots.py
 plugins/
   base.py        ← BasePlugin ABC
   registry.py     ← PluginRegistry singleton
```

---

## 3. Adding a New Metric

Here is the step-by-step workflow for adding a new metric.

**Example: Adding Maximum Calibration Error (MCE)**

### Step 1 — Write the metric function

Add your function to the appropriate module file (or create a new one):

```python
# trustlens/metrics/calibration.py

def maximum_calibration_error(
  y_true: np.ndarray,
  y_prob: np.ndarray,
  n_bins: int = 10,
) -> float:
  """
  Compute Maximum Calibration Error (MCE).

  MCE is the worst-case calibration gap across all bins.
  Lower is better; MCE=0.0 means perfect calibration.

  Parameters
  ----------
  y_true : np.ndarray
    Binary ground-truth labels (0 or 1).
  y_prob : np.ndarray
    Predicted probabilities for the positive class.
  n_bins : int
    Number of confidence bins.

  Returns
  -------
  float
    MCE in [0, 1].
  """
  y_true = np.asarray(y_true, dtype=float)
  y_prob = np.asarray(y_prob, dtype=float)

  bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
  max_gap = 0.0

  for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = (y_prob >= lo) & (y_prob < hi)
    if mask.sum() == 0:
      continue
    accuracy  = y_true[mask].mean()
    confidence = y_prob[mask].mean()
    max_gap = max(max_gap, abs(accuracy - confidence))

  return float(max_gap)
```

### Step 2 — Export from the metrics `__init__.py`

```python
# trustlens/metrics/__init__.py
from trustlens.metrics.calibration import maximum_calibration_error
```

### Step 3 — Integrate into `api.py` (optional)

If the metric should run automatically, add it to the calibration block in `api.py`:

```python
results["calibration"]["mce"] = maximum_calibration_error(y_true, y_prob_pos)
```

### Step 4 — Write tests

```python
# tests/test_calibration.py
def test_mce_perfect_is_zero():
  y_true = np.array([0, 1, 0, 1])
  y_prob = np.array([0.0, 1.0, 0.0, 1.0])
  assert maximum_calibration_error(y_true, y_prob) == pytest.approx(0.0)

def test_mce_geq_ece(binary_random):
  y_true, y_prob = binary_random
  ece = expected_calibration_error(y_true, y_prob)
  mce = maximum_calibration_error(y_true, y_prob)
  assert mce >= ece # MCE is always >= ECE
```

### Step 5 — Document

Add a docstring entry to `docs/api_reference.rst` and mention it in the changelog.

---

## 4. Adding a New Visualization

Visualization functions go in `trustlens/visualization/`.

**Interface contract:**
- Accept pre-computed data from TrustReport (never raw models/data).
- Return a `matplotlib.Figure` — do not call `plt.show()` internally.
- Accept an optional `save_path` parameter.
- Use `matplotlib.use("Agg")` at the top of each visualization file.

**Wire it up in `trustlens/visualization/__init__.py`:**
```python
from trustlens.visualization.my_plot_file import my_plot_function
```

**Wire it up in the dispatcher (`plot_module`):**
```python
def _plot_mycategory(data: dict):
  return my_plot_function(data)

dispatch["mycategory"] = _plot_mycategory
```

---

## 5. Writing a Plugin

Full plugin authoring guide: see `trustlens/plugins/__init__.py`.

Minimum required code:

```python
from trustlens.plugins.base import BasePlugin
from trustlens.plugins.registry import PluginRegistry

class MyCustomPlugin(BasePlugin):
  name = "my_metric"    # unique identifier
  description = "A description shown in the registry."

  def run(self, model, X, y_true, y_pred, y_prob, **kwargs):
    # Your logic here
    return {"result_key": 42.0} # must be JSON-serializable

PluginRegistry().register(MyCustomPlugin)
```

Then run it with:
```python
report = analyze(model, X, y, plugins=["my_metric"])
```

---

## 6. Coding Standards

- **Python 3.9+** — use type hints, `from __future__ import annotations`
- **Ruff** for linting (config in `pyproject.toml`)
- **Black** for formatting (line length = 100)
- **NumPy-style docstrings** — Parameters, Returns, Raises, Examples sections
- **No bare `except:`** — always catch specific exceptions
- **No global state** — functions must be pure and stateless where possible
- **Imports** — standard library → third-party → trustlens internal

Run checks locally:
```bash
ruff check trustlens/
black --check trustlens/
mypy trustlens/
```

---

## 7. Writing Tests

- Place tests in `tests/` mirroring the source structure
- Use `pytest` fixtures for shared setup
- Target **100% branch coverage** for new functions
- Tests must be deterministic — seed all RNGs with a fixed value
- Name tests descriptively: `test_<function>_<scenario>`

```bash
pytest             # run all tests
pytest -k "calibration"    # run matching tests
pytest --cov=trustlens     # run with coverage
```

---

## ⚡ Quick PR Checklist

Before submitting your PR, please ensure:
- **Code runs**: The logic works as expected.
- **Tests pass**: All existing and new tests pass locally.
- **Clear commit message**: Use meaningful titles.
- **Linked issue**: Mention the issue your PR addresses.

## 8. Pull Request Guidelines

1. **One PR per feature/fix** — keep changes focused
2. **Branch naming**: `feature/my-feature`, `fix/bug-description`, `docs/update-readme`
3. **Commit messages**: use [Conventional Commits](https://www.conventionalcommits.org/)
  - `feat: add MCE metric`
  - `fix: handle empty calibration bins`
  - `docs: expand plugin authoring guide`
4. **All tests must pass** before requesting review
5. **Write a clear PR description** — what changed, why, how to test it
6. **Link the related GitHub Issue**

---

## 9. Reporting Issues

Found a bug? Have a feature request?

Open an issue with:
- **Environment**: Python version, OS, TrustLens version
- **Minimal reproducible example** (MRE)
- **Expected vs. actual behaviour**

We aim to respond within 48 hours. ⏱

---

Thanks again for contributing to TrustLens.
You're helping make ML systems more honest, one metric at a time.
