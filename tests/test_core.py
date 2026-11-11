import numpy as np
from self_adapt.simulation import generate_stream
from self_adapt.engine import run_experiment, STRATEGIES
from self_adapt.storage import save_experiment, load_experiments, get_experiment

def cfg(**extra):
    x={"drift_type":"Abrupt","dataset_size":400,"batch_size":40,"drift_start":.5,
       "transition_duration":.1,"seed":17,"label_availability":1.0,"label_delay":0}
    x.update(extra)
    return x

def test_stream_reproducible():
    a=generate_stream(400,"Gradual",.5,.1,123)
    b=generate_stream(400,"Gradual",.5,.1,123)
    assert np.array_equal(a["X"],b["X"])
    assert np.array_equal(a["y"],b["y"])

def test_all_drift_types():
    for drift in ["Abrupt","Gradual","Incremental","Recurring"]:
        d=generate_stream(400,drift,.5,.1,21)
        assert d["X"].shape==(400,6)
        assert set(np.unique(d["y"])).issubset({0,1})

def test_five_strategies_and_metrics():
    r=run_experiment(cfg())
    assert set(r["strategies"])==set(STRATEGIES)
    assert r["summary"]["evaluated_observations"]>0
    for m in r["strategies"].values():
        assert 0<=m["accuracy"]<=1
        assert m["n_evaluated"]>0

def test_zero_label_availability():
    r=run_experiment(cfg(label_availability=0.0))
    assert r["summary"]["evaluated_observations"]==0
    assert r["strategies"]["No Action"]["accuracy"] is None

def test_delay_reduces_available_evaluations():
    immediate=run_experiment(cfg(label_delay=0))
    delayed=run_experiment(cfg(label_delay=200))
    assert delayed["summary"]["evaluated_observations"]<immediate["summary"]["evaluated_observations"]

def test_sqlite_roundtrip(tmp_path):
    r=run_experiment(cfg())
    db=tmp_path/"experiments.sqlite"
    save_experiment(r,db)
    assert get_experiment(r["experiment_id"],db)["experiment_id"]==r["experiment_id"]
    assert len(load_experiments(db))==1


def test_pdf_report_is_generated():
    from self_adapt.reports import build_pdf_report
    result = run_experiment(cfg())
    pdf = build_pdf_report([result], "Test Report")
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_confusion_matrices_have_named_counts_and_sum_to_evaluations():
    r = run_experiment(cfg())
    for metric in r["strategies"].values():
        cm = metric["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2
        assert sum(sum(row) for row in cm) == metric["n_evaluated"]
        assert metric["tn"] == cm[0][0]
        assert metric["fp"] == cm[0][1]
        assert metric["fn"] == cm[1][0]
        assert metric["tp"] == cm[1][1]


def test_phase_counts_cover_all_evaluated_labels_for_abrupt_drift():
    r = run_experiment(cfg(drift_type="Abrupt", drift_start=.5))
    for strategy, phases in r["phase_metrics"].items():
        phase_total = sum(phases[k]["n_evaluated"] for k in ["Before drift", "Transition", "After transition"])
        assert phase_total == r["strategies"][strategy]["n_evaluated"]
        assert phases["Transition"]["n_evaluated"] == 0


def test_phase_counts_cover_all_evaluated_labels_for_gradual_drift():
    r = run_experiment(cfg(drift_type="Gradual", drift_start=.5, transition_duration=.2))
    for strategy, phases in r["phase_metrics"].items():
        phase_total = sum(phases[k]["n_evaluated"] for k in ["Before drift", "Transition", "After transition"])
        assert phase_total == r["strategies"][strategy]["n_evaluated"]


def test_summary_includes_runner_up_and_margin():
    r = run_experiment(cfg())
    assert "best_accuracy_strategy" in r["summary"]
    assert "runner_up_accuracy_strategy" in r["summary"]
    assert r["summary"]["accuracy_margin"] is not None


def test_pdf_contains_detailed_analysis_and_confusion_matrices():
    from self_adapt.reports import build_pdf_report, analyze_experiment
    result = run_experiment(cfg())
    analysis = analyze_experiment(result)
    assert "verdict" in analysis and "why" in analysis
    pdf = build_pdf_report([result], "Detailed Test Report")
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 5000
