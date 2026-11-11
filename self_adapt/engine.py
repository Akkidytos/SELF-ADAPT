"""Batch-prequential experiment engine for four baselines and SELF-ADAPT."""
from collections import Counter
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)

from .simulation import generate_initial_data, generate_stream

STRATEGIES = [
    "No Action", "Probability Recalibration", "Incremental Learning",
    "Full Retraining", "SELF-ADAPT",
]


def _model(seed):
    return SGDClassifier(
        loss="log_loss", alpha=0.0008, random_state=seed,
        learning_rate="optimal", max_iter=1200, tol=1e-3,
    )


def _logit(p):
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _metrics(yt, yp, probs):
    if not yt:
        return {
            "n_evaluated": 0, "accuracy": None, "precision": None,
            "recall": None, "f1": None, "roc_auc": None,
            "specificity": None, "confusion_matrix": [[0, 0], [0, 0]],
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        }
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    try:
        auc = float(roc_auc_score(yt, probs)) if len(set(yt)) > 1 else None
    except Exception:
        auc = None
    return {
        "n_evaluated": len(yt),
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "roc_auc": auc,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        # Row = actual class [0, 1], column = predicted class [0, 1].
        "confusion_matrix": cm.tolist(),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def _phase_metrics(indices, yt, yp, start, end, drift_type):
    """Accuracy by stream phase; indices refer to original stream positions."""
    idx = np.asarray(indices, dtype=int)
    truth = np.asarray(yt, dtype=int)
    pred = np.asarray(yp, dtype=int)
    if end > start:
        transition_mask = (idx >= start) & (idx <= end)
        after_mask = idx > end
    else:
        # An abrupt change has no transition interval; the drift-start
        # observation belongs to the post-drift phase.
        transition_mask = np.zeros(len(idx), dtype=bool)
        after_mask = idx >= start
    masks = {
        "Before drift": idx < start,
        "Transition": transition_mask,
        "After transition": after_mask,
    }
    out = {}
    for label, mask in masks.items():
        count = int(mask.sum())
        out[label] = {
            "n_evaluated": count,
            "accuracy": float(accuracy_score(truth[mask], pred[mask])) if count else None,
        }
    if drift_type == "Recurring":
        out["note"] = "Recurring streams revisit concepts; 'After transition' combines later recurring phases."
    else:
        out["note"] = "Phase metrics use the original stream position of each prediction whose label became available."
    return out


def _winner(values):
    valid = [(float(v), name) for name, v in values.items() if v is not None]
    if not valid:
        return [], None
    best = max(v for v, _ in valid)
    winners = [name for value, name in valid if abs(value - best) < 1e-12]
    return winners, best


def run_experiment(config):
    """Run one reproducible batch-prequential experiment.

    Each batch is predicted before any labels from that batch are released.
    Only labels that arrive under the configured delay/availability protocol
    are used for supervised evaluation and model updates.
    """
    n = int(config.get("dataset_size", 2400))
    batch_size = int(config.get("batch_size", 80))
    seed = int(config.get("seed", 42))
    delay = int(config.get("label_delay", 0))
    availability = float(config.get("label_availability", 1.0))
    drift_type = config.get("drift_type", "Abrupt")
    if n < 100:
        raise ValueError("Stream size must be at least 100 observations.")
    if not 10 <= batch_size <= n:
        raise ValueError("Batch size must be between 10 and stream size.")
    if delay < 0 or not 0 <= availability <= 1:
        raise ValueError("Label delay must be non-negative and label availability must be between 0 and 1.")

    stream = generate_stream(
        n, drift_type, float(config.get("drift_start", .55)),
        float(config.get("transition_duration", .12)), seed,
    )
    X, y = stream["X"], stream["y"]
    X0, y0 = generate_initial_data(500, seed + 1000)
    rng = np.random.default_rng(seed + 9000)

    states = {}
    for name in STRATEGIES:
        model = _model(seed)
        model.fit(X0, y0)
        states[name] = {
            "model": model, "calibrator": None,
            "X_train": [row.copy() for row in X0],
            "y_train": [int(v) for v in y0],
            "yt": [], "yp": [], "probs": [], "indices": [],
            "errors": [], "cal_pairs": [],
            "update_time": 0.0, "updates": 0, "actions": Counter(),
        }

    pending, timeline, drift_events = [], [], []
    batch_number = 0
    for batch_start in range(0, n, batch_size):
        batch_number += 1
        batch_end = min(n, batch_start + batch_size)

        # Predict the entire batch before revealing any labels from it.
        for idx in range(batch_start, batch_end):
            rec = {
                "idx": idx, "x": X[idx].copy(), "y": int(y[idx]),
                "selected": bool(rng.random() < availability), "preds": {},
            }
            for name, state in states.items():
                base_p = float(state["model"].predict_proba(X[idx:idx + 1])[0, 1])
                final_p = base_p
                if name in ("Probability Recalibration", "SELF-ADAPT") and state["calibrator"] is not None:
                    final_p = float(state["calibrator"].predict_proba([[_logit(base_p)]])[0, 1])
                rec["preds"][name] = {
                    "pred": int(final_p >= .5), "prob": final_p, "base": base_p,
                }
            pending.append(rec)

        cutoff = batch_end - 1 - delay
        released = [r for r in pending if r["idx"] <= cutoff and r["selected"]]
        # Labels that were not selected are permanently unavailable. Labels
        # beyond the delay cutoff remain pending for a later batch.
        pending = [r for r in pending if r["idx"] > cutoff]

        if released:
            X_new = np.asarray([r["x"] for r in released])
            y_new = np.asarray([r["y"] for r in released], dtype=int)
            for name, state in states.items():
                for rec in released:
                    prediction = rec["preds"][name]
                    state["yt"].append(rec["y"])
                    state["yp"].append(prediction["pred"])
                    state["probs"].append(prediction["prob"])
                    state["indices"].append(rec["idx"])
                    state["errors"].append(int(prediction["pred"] != rec["y"]))
                    state["cal_pairs"].append((_logit(prediction["base"]), rec["y"]))
                    state["X_train"].append(rec["x"].copy())
                    state["y_train"].append(rec["y"])

            for name, state in states.items():
                action = "No update"
                started = perf_counter()

                if name == "Probability Recalibration":
                    pairs = state["cal_pairs"][-800:]
                    if len(pairs) >= 30 and len({p[1] for p in pairs}) == 2:
                        calibrator = LogisticRegression(C=1.0, solver="lbfgs")
                        calibrator.fit(
                            np.asarray([[p[0]] for p in pairs]),
                            np.asarray([p[1] for p in pairs]),
                        )
                        state["calibrator"] = calibrator
                        action = "Probability recalibration"

                elif name == "Incremental Learning":
                    state["model"].partial_fit(X_new, y_new, classes=np.array([0, 1]))
                    action = "Incremental update"

                elif name == "Full Retraining":
                    fresh = _model(seed)
                    fresh.fit(np.asarray(state["X_train"]), np.asarray(state["y_train"]))
                    state["model"] = fresh
                    action = "Full retraining"

                elif name == "SELF-ADAPT":
                    errors = state["errors"]
                    recent = float(np.mean(errors[-60:])) if len(errors) >= 20 else 0.0
                    if len(errors) >= 120:
                        prior = float(np.mean(errors[-120:-60]))
                    elif len(errors) > 60:
                        prior = float(np.mean(errors[:max(1, len(errors) - 60)]))
                    else:
                        prior = recent
                    signal = len(errors) >= 40 and recent >= .25 and recent - prior >= .08
                    enough_feedback = len(released) >= 5

                    if enough_feedback and signal and recent >= .42 and len(state["y_train"]) >= 150:
                        fresh = _model(seed)
                        fresh.fit(np.asarray(state["X_train"]), np.asarray(state["y_train"]))
                        state["model"] = fresh
                        state["calibrator"] = None
                        state["cal_pairs"] = []
                        action = "Full retraining"
                    elif enough_feedback and signal:
                        state["model"].partial_fit(X_new, y_new, classes=np.array([0, 1]))
                        # A calibrator fitted for the previous model is stale
                        # after its underlying classifier changes.
                        state["calibrator"] = None
                        state["cal_pairs"] = []
                        action = "Incremental update"
                    elif enough_feedback and len(state["cal_pairs"]) >= 40 and recent >= .18:
                        pairs = state["cal_pairs"][-800:]
                        if len({p[1] for p in pairs}) == 2:
                            calibrator = LogisticRegression(C=1.0, solver="lbfgs")
                            calibrator.fit(
                                np.asarray([[p[0]] for p in pairs]),
                                np.asarray([p[1] for p in pairs]),
                            )
                            state["calibrator"] = calibrator
                            action = "Probability recalibration"

                    if signal and action != "No update":
                        drift_events.append({
                            "batch": batch_number, "stream_index": batch_end - 1,
                            "recent_error": recent, "prior_error": prior,
                            "selected_action": action,
                        })

                elapsed = perf_counter() - started
                if action != "No update":
                    state["update_time"] += elapsed
                    state["updates"] += 1
                    state["actions"][action] += 1

        for name, state in states.items():
            yt, yp = state["yt"], state["yp"]
            rolling = float(accuracy_score(yt[-100:], yp[-100:])) if yt else None
            cumulative = float(accuracy_score(yt, yp)) if yt else None
            timeline.append({
                "batch": batch_number, "stream_end": batch_end - 1,
                "strategy": name, "labelled_observations": len(yt),
                "rolling_accuracy": rolling, "cumulative_accuracy": cumulative,
            })

    drift_start = int(stream["drift_start_index"])
    drift_end = int(stream["transition_end_index"])
    metrics, phase_metrics = {}, {}
    for name, state in states.items():
        m = _metrics(state["yt"], state["yp"], state["probs"])
        m.update({
            "update_time_seconds": float(state["update_time"]),
            "update_count": int(state["updates"]),
            "action_counts": dict(state["actions"]),
        })
        metrics[name] = m
        phase_metrics[name] = _phase_metrics(
            state["indices"], state["yt"], state["yp"],
            drift_start, drift_end, drift_type,
        )

    accuracy_winners, best_accuracy = _winner({n: m["accuracy"] for n, m in metrics.items()})
    f1_winners, best_f1 = _winner({n: m["f1"] for n, m in metrics.items()})
    active_times = {
        name: m["update_time_seconds"] for name, m in metrics.items()
        if m["update_count"] > 0
    }
    fastest_names, fastest_time = _winner({name: -time for name, time in active_times.items()})
    fastest_strategy = fastest_names[0] if fastest_names else None
    ranked = sorted(
        [(m["accuracy"], name) for name, m in metrics.items() if m["accuracy"] is not None],
        key=lambda item: (-item[0], STRATEGIES.index(item[1])),
    )
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = (ranked[0][0] - runner_up[0]) if len(ranked) > 1 else None
    evaluated = metrics["SELF-ADAPT"]["n_evaluated"]
    summary = {
        "best_accuracy_strategy": ", ".join(accuracy_winners) if accuracy_winners else None,
        "best_accuracy_strategies": accuracy_winners,
        "best_accuracy": best_accuracy,
        "runner_up_accuracy_strategy": runner_up[1] if runner_up else None,
        "runner_up_accuracy": runner_up[0] if runner_up else None,
        "accuracy_margin": margin,
        "best_f1_strategy": ", ".join(f1_winners) if f1_winners else None,
        "best_f1_strategies": f1_winners,
        "best_f1": best_f1,
        "fastest_updating_strategy": fastest_strategy,
        "fastest_update_time_seconds": active_times.get(fastest_strategy) if fastest_strategy else None,
        "self_adapt_accuracy": metrics["SELF-ADAPT"]["accuracy"],
        "self_adapt_f1": metrics["SELF-ADAPT"]["f1"],
        "self_adapt_updates": metrics["SELF-ADAPT"]["update_count"],
        "evaluated_observations": evaluated,
        "interpretation": (
            f"{', '.join(accuracy_winners)} achieved the highest overall accuracy "
            f"({best_accuracy:.3f}) in this run. "
            + (f"The next strategy was {runner_up[1]} ({runner_up[0]:.3f}), "
               f"a gap of {margin * 100:.2f} percentage points. " if runner_up else "")
            + "This is a single-run result, not a universal ranking."
            if best_accuracy is not None else
            "No labels became available for evaluation. Increase label availability or reduce label delay."
        ),
    }
    return {
        "experiment_id": str(uuid4())[:8].upper(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            **config, "dataset_size": n, "batch_size": batch_size,
            "seed": seed, "label_delay": delay, "label_availability": availability,
        },
        "drift": {
            "type": stream["drift_type"], "start_index": drift_start,
            "transition_end_index": drift_end,
        },
        "strategies": metrics, "phase_metrics": phase_metrics,
        "timeline": timeline, "drift_events": drift_events, "summary": summary,
        "evaluation_protocol": "Batch-prequential: predict each batch before releasing eligible labels.",
    }
