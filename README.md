# SELF-ADAPT Research Prototype

**Research title:** Choosing the Right Model Update Strategy under Concept Drift  
**Application:** SELF-ADAPT — Indian-inspired research analytics edition

SELF-ADAPT is a local Streamlit research prototype for running reproducible synthetic-stream experiments and comparing five update policies:

1. No Action
2. Probability Recalibration
3. Incremental Learning
4. Full Retraining
5. SELF-ADAPT — an explainable, rule-based adaptive policy

It supports Abrupt, Gradual, Incremental and Recurring drift; delayed and partially available labels; batch-prequential evaluation; persistent experiment history; interactive comparison charts; confusion matrices; pre-drift/transition/post-drift accuracy; beginner explanations; CSV/JSON exports; and detailed individual or combined PDF reports. It can also repeat one configuration across five consecutive random seeds and calculate descriptive mean and standard deviation summaries.

## Research integrity

- Metrics are calculated from experiment outputs; no preset accuracy values are embedded.
- Confusion matrices use rows for actual classes and columns for predicted classes: `[[TN, FP], [FN, TP]]`.
- Phase-wise accuracy separates observations before drift, during the transition, and after the transition. For recurring drift, the post-transition segment combines later recurring phases and is labelled accordingly.
- A late drift can make whole-stream accuracy misleading because most evaluated observations may belong to the old concept. Always inspect phase-wise and rolling accuracy as well.
- Predictions for a batch are generated before labels from that batch are released.
- Only labels that arrive under the configured delay and availability settings are used for supervised evaluation and model updates.
- Update time is wall-clock time for model-fitting/update operations; it excludes prediction latency and total application runtime, and depends on the computer running the experiment.
- SELF-ADAPT is a prototype rule-based policy, not a proven optimal selector.
- One run is a pilot observation. Repeat multiple seeds and evaluate real datasets before making general claims or statistical significance claims.
- Synthetic streams are for controlled experimentation, not a substitute for domain-specific real data.

## Requirements

- Python 3.10+ (Python 3.12 recommended)
- Windows PowerShell, macOS Terminal or Linux shell

## Windows setup

Open PowerShell in the extracted project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

If activation is blocked, call the venv Python directly:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the Local URL printed by Streamlit, normally `http://localhost:8501`.

## Local demo login

- Username: `researcher`
- Password: `adapt2026`

This is only a local prototype access gate, not production-grade authentication. Change it before sharing:

```powershell
$env:SELF_ADAPT_USER = "your-user"
$env:SELF_ADAPT_PASSWORD = "your-long-private-password"
python -m streamlit run app.py
```

Do not commit real credentials.

## Run tests

```powershell
python -m pytest tests -v
```

## Suggested first experiment

- Drift: Abrupt
- Stream size: 2,400
- Batch size: 80
- Drift starts: 55%
- Transition duration: 12%
- Label availability: 100%
- Label delay: 0
- Random seed: 42

Change one setting at a time for controlled comparisons. In the Experiment Lab, enable **Repeat with 5 seeds** to run the same configuration with the selected seed and the next four seeds. Use the same drift/stream/batch/feedback configuration when comparing seed averages. Do not combine different configurations into one average.

### Demonstration where Incremental Learning is not the overall winner

Try this controlled late-drift configuration: Abrupt drift, stream size 5,000, batch size 500, drift start 90%, seed 42, 100% label availability and zero label delay. In one verified run, Full Retraining achieved 0.8436 overall accuracy while Incremental Learning achieved 0.8348. This is a single-seed observation, not a general ranking. Only the final 500 observations were post-drift, so whole-stream accuracy was dominated by the old concept and post-drift accuracy was poor for every strategy. Inspect phase-wise accuracy and repeat the same setup across five seeds before interpreting it.

## Pages

- **Overview:** latest saved experiment and workflow
- **New Experiment:** configure and run simulations; optionally repeat the same configuration with five consecutive seeds and view mean/standard-deviation summaries
- **Strategy Comparison:** predictive metrics, update time, update count, accuracy-cost scatter plot, plain-language winner analysis, confusion-matrix explorer and phase-wise accuracy
- **Drift Analysis:** rolling prequential accuracy with simulated drift marker
- **Experiment History:** SQLite-backed run history and previous-vs-current comparison
- **Research Reports:** detailed individual/combined PDF, CSV and JSON exports. The PDF explains the setup, why it matters, how evaluation works, which strategy won, the margin to the runner-up, all strategy metrics, confusion matrices, drift-phase performance, practical interpretation and limitations. When selected runs share the same configuration but use different seeds, it adds mean and standard deviation tables.
- **Beginner Academy:** metric glossary and strategy explanations

## Project structure

```text
SELF-ADAPT_Research_Prototype/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/config.toml
├── assets/styles.css
├── self_adapt/
│   ├── simulation.py
│   ├── engine.py
│   ├── storage.py
│   └── reports.py
├── results/                 # SQLite database created at runtime
└── tests/test_core.py
```

## Strategy notes

- **No Action:** frozen initial SGDClassifier.
- **Probability Recalibration:** logistic calibration layer fitted on labelled base-model probability/logit pairs; the underlying classifier remains frozen.
- **Incremental Learning:** updates the existing SGDClassifier using `partial_fit`.
- **Full Retraining:** fits a fresh SGDClassifier on the available labelled history.
- **SELF-ADAPT:** uses recent available prediction errors and feedback volume to select no update, recalibration, incremental update or full retraining. A calibration layer and its old score history are cleared when the underlying classifier is incrementally updated or retrained, preventing a stale calibrator from being applied to changed model scores. This is a transparent prototype rule and should be validated with repeated experiments and ablation studies.

## Existing local experiment results

This ZIP is a clean rebuilt application. It does not automatically contain experiment history stored only in another copy on your computer. Export earlier runs from the old app before replacing it if you need to preserve them. New runs are saved in `results/experiments.sqlite`.

## Limitations and future work

The current stream is synthetic and binary. The policy thresholds need systematic tuning and ablation. Drift signals use delayed, available prediction errors and do not identify the causal source of drift. The PDF is a detailed descriptive report, not a peer-reviewed statistical analysis. Wall-clock update times depend on the machine. Future work should include real-world datasets, detector benchmarking, additional model families, ablation studies, formal statistical tests over repeated seeds and resource profiling.
