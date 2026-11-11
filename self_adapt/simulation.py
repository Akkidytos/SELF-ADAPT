"""Synthetic binary-classification streams with controlled concept drift."""
import numpy as np

CONCEPT_A = np.array([1.15, -0.85, 0.65, 0.25, -0.35, 0.45], dtype=float)
CONCEPT_B = np.array([-0.75, 1.10, -0.55, 0.60, 0.30, -0.25], dtype=float)


def _labels(X, weights, rng):
    return ((X @ weights[:X.shape[1]] + rng.normal(0, 0.45, len(X))) >= 0).astype(int)


def generate_initial_data(n_samples=500, seed=42, n_features=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    return X, _labels(X, CONCEPT_A, rng)


def generate_stream(n_samples=2400, drift_type="Abrupt", drift_start=0.55,
                    transition_duration=0.12, seed=42, n_features=6):
    """Abrupt, Gradual, Incremental and Recurring concept-drift stream."""
    allowed = {"Abrupt", "Gradual", "Incremental", "Recurring"}
    if drift_type not in allowed:
        raise ValueError(f"drift_type must be one of {sorted(allowed)}")
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")
    if not 0 <= drift_start < 1:
        raise ValueError("drift_start must be in [0, 1)")
    if not 0 <= transition_duration <= 0.8:
        raise ValueError("transition_duration must be in [0, 0.8]")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    start = int(n_samples * drift_start)
    width = max(1, int(n_samples * max(transition_duration, 0.01)))
    # Abrupt drift has no transition interval; gradual/incremental drift do.
    end = start if drift_type == "Abrupt" else min(n_samples - 1, start + width)
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if drift_type == "Abrupt":
            w = CONCEPT_A if i < start else CONCEPT_B
        elif drift_type == "Gradual":
            p_new = 0.0 if i < start else min(1.0, (i-start)/max(1, end-start))
            w = CONCEPT_B if rng.random() < p_new else CONCEPT_A
        elif drift_type == "Incremental":
            alpha = 0.0 if i < start else min(1.0, (i-start)/max(1, end-start))
            w = (1-alpha)*CONCEPT_A + alpha*CONCEPT_B
        else:
            phase = 0 if i < start else ((i-start)//width) % 2
            w = CONCEPT_A if phase == 1 else CONCEPT_B
            if i < start:
                w = CONCEPT_A
        y[i] = _labels(X[i:i+1], w, rng)[0]
    return {"X": X, "y": y, "drift_start_index": start,
            "transition_end_index": end, "drift_type": drift_type}
