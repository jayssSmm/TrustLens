# TrustLens

<!-- PyPI description start -->

TrustLens is a Python library for analyzing machine learning model reliability beyond accuracy, including calibration, bias, failure analysis, and explainability.

Key capabilities:
- Trust Score (0–100 reliability metric)
- Calibration analysis (Brier Score, ECE)
- Failure analysis (confidence gaps, misclassifications)
- Bias detection (class imbalance, subgroup performance)
- Representation analysis (embedding separability, CKA)

```bash
pip install trustlens
```
<!-- PyPI description end -->

## TrustLens

### Your model has 92% accuracy. But can you trust it?

**TrustLens is the open-source library that answers the questions accuracy never does.**

[![PyPI](https://badge.fury.io/py/trustlens.svg)](https://pypi.org/project/trustlens)
[![CI](https://github.com/Khanz9664/TrustLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Khanz9664/TrustLens/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Khanz9664/TrustLens?style=social)](https://github.com/Khanz9664/TrustLens/stargazers)


---

## The Problem Nobody Talks About

You trained a model. It hits **92% accuracy** on your validation set.

So you ship it.

Three months later:
- A minority-class user gets consistently wrong predictions
- The model is 90% confident on its worst mistakes
- A regulator asks "why did it make that decision?" and you have no answer

**Accuracy tells you how often your model is right. It tells you nothing about when it fails, why it fails, or who it fails.**

TrustLens fixes that. In one function call.

---

## Quick Analyze (Zero-Friction Start)

Try TrustLens instantly without bringing your own data or models. We provide a zero-friction entry point:

```python
from trustlens import quick_analyze

# Automatically loads the breast cancer dataset, trains a baseline logic model,
# and runs the full analysis, returning a TrustReport and rendering the dashboard.
report = quick_analyze(dataset="breast_cancer")
```

---

## Quick Usage with Custom Models

```bash
pip install trustlens
```

```python
from trustlens import analyze

report = analyze(
  model,     # any sklearn-compatible model
  X_val,     # validation features
  y_val,     # ground truth
  y_prob=proba,  # predicted probabilities
)

print(report.trust_score)
report.show()
```

**Output Insight:**

```text
==================================================================
  TrustLens Report
==================================================================
 Timestamp : 2026-04-16T15:43:02Z
 Model   : RandomForestClassifier
 Samples  : 2,500
 Classes  : 2

==================================================================
 TRUST SCORE: 61/100 [B]
 Assessment: Good Trust - minor issues to address
==================================================================

 Key Observations:
  * Calibration needs improvement (ECE > 0.1).
  * Model is overconfident on incorrect predictions (low confidence gap).

==================================================================
 Dimension breakdown:
  calibration    52.3/100 []
  failure      74.1/100 []
  bias        41.2/100 []
  representation   68.5/100 []
```

Your calibration is fine. Your bias score is not. **TrustLens just saved you a PR disaster.**

---

## The Summary Dashboard

One line. One picture. Everything you need.

```python
report.summary_plot()
```

The presentation-ready 6-panel dashboard shows:
- **Trust Score gauge**: Your model's overall trustworthiness at a glance
- **Reliability diagram**: Is your model overconfident or underconfident?
- **Confidence gap**: Does high confidence actually mean high accuracy?
- **Error rate by class**: Which classes are being failed?
- **Class distribution**: Is your training data biased?
- **Sub-score breakdown**: Which dimension needs the most work?

---

## The Trust Score

A single, actionable number: **0 to 100**.

Computed from four dimensions, each independently interpretable:

| Dimension | What it measures | Weight |
|-----------|-----------------|--------|
| **Calibration** | Do probabilities reflect reality? | 35% |
| **Failure** | Does confidence correlate with accuracy? | 30% |
| **Bias** | Are all groups treated equally? | 25% |
| **Representation** | Is the embedding space well-structured? | 10% |

| Score | Grade | Recommendation |
|-------|-------|----------------|
| 80-100 | A | Production-ready |
| 60-79 | B | Good - fix flagged issues first |
| 40-59 | C | Investigate before deployment |
| 0-39 | D | Do not deploy |

---

## The Failure Showcase

Find your model's most dangerous mistakes in 1 line:

```python
report.show_failures(top_k=5)
```

**Output:**

```text
==================================================================
 TOP 5 CRITICAL FAILURES
 GradientBoostingClassifier | 58 total errors / 700 samples (8.3%)
==================================================================
 #  Sample  True  Pred  Confidence  Danger
 ------------------------------------------------------
 1  412     1   0    97.4%  CRITICAL
 2  88     0   1    95.1%  CRITICAL
 3  301     1   0    91.8%  HIGH
 4  556     0   1    89.2%  HIGH

 Insights:
   Mean confidence on top failures: 93.4%
   These are high-confidence mistakes - the model is
   certain it is right, but it is wrong.
   Overconfidence detected - consider calibration.
```

---

## Real-World Use Cases

### Medical AI
A diagnostic model with 94% accuracy has an ECE of 0.18 - dangerously overconfident on edge cases. TrustLens surfaces it before deployment.

### Fraud Detection
Your model's confidence gap is 0.04 - it's almost as confident on fraud it misses as on fraud it catches. That's your false-negative problem, quantified.

### Hiring, Loan, and Insurance
Subgroup analysis reveals a 23% accuracy gap between applicant demographics. You have a fairness problem. Now you know before a regulator tells you.

### Research
Use CKA to compare representation quality across model architectures. Use faithfulness testing to benchmark explanation methods honestly.

## Repository Structure

```text
TrustLens/
├── assets/
│   ├── banner.png
│   └── logo.png
├── docs/
│   ├── DESIGN_PRINCIPLES.md
│   ├── FUTURE_EXTENSIONS.md
│   ├── GITHUB_ISSUES.md
│   ├── POSITIONING.md
│   └── REWRITTEN_ISSUES.md
├── examples/
│   ├── calibration_deep_dive.py
│   ├── cnn_vs_vit_trustlens.py
│   ├── custom_plugin_demo.py
│   ├── quickstart.py
│   └── trustlens_demo.ipynb
├── .github/workflows/
│   └── ci.yml
├── tests/
│   ├── test_api.py
│   ├── test_bias.py
│   ├── test_calibration.py
│   ├── test_failure.py
│   ├── test_output_formatting.py
│   ├── test_plugins.py
│   ├── test_representation.py
│   └── test_trust_score.py
├── trustlens/
│   ├── explainability/
│   │   ├── faithfulness.py
│   │   └── gradcam.py
│   ├── metrics/
│   │   ├── bias.py
│   │   ├── calibration.py
│   │   ├── failure.py
│   │   ├── faithfulness.py
│   │   └── representation.py
│   ├── plugins/
│   │   ├── base.py
│   │   └── registry.py
│   ├── visualization/
│   │   ├── bias_plots.py
│   │   ├── calibration_plots.py
│   │   ├── failure_plots.py
│   │   ├── representation_plots.py
│   │   └── summary_plot.py
│   ├── api.py
│   ├── report.py
│   ├── trust_score.py
│   └── utils.py
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
├── requirements.txt
└── ROADMAP.md
```

---

## Contributing

TrustLens is designed to grow with the community. Adding a new metric takes just four simple steps:

> **Testing Policy:** Current test coverage is 67% to ensure core stability. It will be incrementally improved toward 85%+ as advanced modules (e.g., explainability, visualization) receive additional tests. All new contributions must maintain or improve this baseline.

1. Write a pure function `my_metric(y_true, y_pred) -> float`
2. Add it to the appropriate module (`metrics/calibration.py`, etc.)
3. Export it from `metrics/__init__.py`
4. Write a test in `tests/test_<module>.py`

See `CONTRIBUTING.md` for the full guide including instructions on adding plugins and explainability methods.
Review `docs/GITHUB_ISSUES.md` for open tasks ready to be developed.

---

## Citation

```bibtex
@software{trustlens2026,
 author = {Shahid Ul Islam},
 title = {TrustLens: Debug your ML models beyond accuracy},
 year  = {2026},
 url  = {https://github.com/Khanz9664/TrustLens},
}
```

---

## Author & Maintainer

**Shahid Ul Islam**
- **GitHub**: [Khanz9664](https://github.com/Khanz9664)
- **Portfolio**: [Visit Portfolio](https://khanz9664.github.io/portfolio/)
- **LinkedIn**: [Connect on LinkedIn](https://www.linkedin.com/in/shahid-ul-islam-13650998/)
- **Instagram**: [Follow on Instagram](https://instagram.com/shaddy9664)

---

<div align="center">

**If TrustLens saved you from a bad deployment, star it.**
It helps other engineers find it before they make the same mistake.

[GitHub](https://github.com/Khanz9664/TrustLens) | [Portfolio](https://khanz9664.github.io/portfolio/) | [LinkedIn](https://www.linkedin.com/in/shahid-ul-islam-13650998/) | [Discussions](https://github.com/Khanz9664/TrustLens/discussions)

</div>
