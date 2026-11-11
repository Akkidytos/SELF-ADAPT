# Changelog

## Research analysis update

- Added a confusion-matrix explorer in the Strategy Comparison page.
- Added TN, FP, FN, TP and specificity to strategy metrics and CSV exports.
- Added pre-drift, transition and post-transition accuracy for every strategy.
- Corrected abrupt-drift metadata so an abrupt change has no transition interval.
- Added plain-language experiment interpretation: winner, runner-up margin, fastest active updater, SELF-ADAPT comparison and late-drift caveat.
- Rebuilt PDF reporting with experiment setup and rationale, evaluation protocol, complete strategy metrics, confusion matrices, phase-wise accuracy, strategy-by-strategy interpretation, practical implications and limitations.
- Added optional five-seed repeated experiments and descriptive mean/standard-deviation summaries.
- Cleared SELF-ADAPT's calibration layer and old calibration-score history after its base classifier changes.
- Added a late-abrupt-drift demonstration configuration where Incremental Learning is not the overall winner in a verified single-seed run; the app explicitly warns that this is not a universal ranking and that post-drift accuracy must also be checked.
- Expanded automated tests for confusion matrices, phase accounting, summary margins and PDF generation.

## Validation

- Python compilation checks passed.
- `pytest tests -q`: 12 passed.
- PDF generation checked for single-run, multi-experiment, repeated-seed and zero-available-label cases.
- SQLite save/load round-trip passed.
- App module import passed with a lightweight Streamlit stub. A live Streamlit browser session was not run in the packaging environment because Streamlit is not installed there; it is listed in `requirements.txt` for local installation.
