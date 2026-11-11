"""Beginner-friendly, evidence-based PDF reporting for SELF-ADAPT."""
from collections import defaultdict
from datetime import datetime
from html import escape
from io import BytesIO
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, KeepTogether,
)

NAVY = colors.HexColor("#172B46")
SAFFRON = colors.HexColor("#D98232")
GREEN = colors.HexColor("#27845B")
IVORY = colors.HexColor("#F7F5F0")
MUTED = colors.HexColor("#586779")

STRATEGY_MECHANICS = {
    "No Action": "Keeps the original model frozen. It is the control baseline and has almost no update cost, but it cannot learn a changed concept.",
    "Probability Recalibration": "Keeps the classifier frozen and learns a mapping from its scores to corrected probabilities. It can help when the ranking signal remains useful but confidence or the decision threshold has shifted; it cannot learn new feature relationships.",
    "Incremental Learning": "Updates the existing classifier with newly available labelled batches. It can react continuously without fitting a new model from scratch, although its result depends on feedback volume, batch size and the optimizer's learning behaviour.",
    "Full Retraining": "Fits a fresh classifier using the labelled history available so far. It can rebuild model parameters, but may cost more and older observations can dilute a recent concept.",
    "SELF-ADAPT": "Uses recent prediction errors and feedback volume to choose between no update, recalibration, incremental learning and full retraining. It is a transparent prototype policy, not a proven optimal selector.",
}


def _fmt(value, digits=3):
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _counts(metric):
    matrix = metric.get("confusion_matrix", [[0, 0], [0, 0]])
    tn, fp, fn, tp = int(matrix[0][0]), int(matrix[0][1]), int(matrix[1][0]), int(matrix[1][1])
    return tn, fp, fn, tp


def _pct(value):
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


def _phase_accuracy(result, strategy, phase):
    phases = result.get("phase_metrics", {}).get(strategy, {})
    return phases.get(phase, {}).get("accuracy")


def analyze_experiment(result):
    """Return a plain-language interpretation grounded in one saved run."""
    strategies = result.get("strategies", {})
    summary = result.get("summary", {})
    config = result.get("config", {})
    drift = result.get("drift", {})
    winners = summary.get("best_accuracy_strategies") or (
        [summary.get("best_accuracy_strategy")] if summary.get("best_accuracy_strategy") else []
    )
    best_name = winners[0] if winners else None
    best_acc = summary.get("best_accuracy")
    runner = summary.get("runner_up_accuracy_strategy")
    runner_acc = summary.get("runner_up_accuracy")
    margin = summary.get("accuracy_margin")
    fastest = summary.get("fastest_updating_strategy")
    self_adapt = strategies.get("SELF-ADAPT", {})
    no_action = strategies.get("No Action", {})
    n = int(config.get("dataset_size", 0) or 0)
    drift_start = int(drift.get("start_index", 0) or 0)
    late_drift = bool(n and drift_start / n >= .80)

    if best_name is None:
        verdict = "No strategy can be declared the winner because no ground-truth labels became available for evaluation."
    else:
        verdict = f"{', '.join(winners)} recorded the highest overall accuracy at {_pct(best_acc)}."
        if runner and runner_acc is not None and margin is not None:
            verdict += f" {runner} followed at {_pct(runner_acc)}, a difference of {margin * 100:.2f} percentage points."
        verdict += " This ranking describes this run only; it is not proof that the strategy will win on every stream."

    if late_drift:
        why = (
            "The simulated drift begins late in the stream. Overall accuracy therefore contains many more "
            "pre-drift observations than post-drift observations. A strategy can win the overall score while "
            "still performing poorly after drift. Use the phase-wise table and rolling-accuracy chart before "
            "claiming that it adapted well."
        )
    elif best_name == "Incremental Learning":
        why = (
            "Incremental Learning updates the existing model whenever eligible labelled feedback arrives. "
            "Its result is consistent with continuous adaptation being useful in this stream. The score alone "
            "does not prove that the update mechanism caused the advantage; repeat the run with other seeds."
        )
    elif best_name == "Probability Recalibration":
        why = (
            "Probability Recalibration changes the score-to-probability mapping while leaving the classifier "
            "itself frozen. Its result is consistent with a stream where probability calibration is useful "
            "without needing to relearn all feature relationships. Check recall, precision and the confusion "
            "matrix to understand which errors were reduced."
        )
    elif best_name == "Full Retraining":
        why = (
            "Full Retraining fits a fresh classifier using the labelled history available at each update. "
            "Its result is consistent with rebuilding model parameters being useful in this run, but it may "
            "consume more time and older data can dilute a newer concept."
        )
    elif best_name == "SELF-ADAPT":
        actions = self_adapt.get("action_counts", {})
        action_text = ", ".join(f"{k}: {v}" for k, v in actions.items()) or "no model updates"
        why = (
            f"SELF-ADAPT selected actions recorded in this run ({action_text}). Its score should be read "
            "together with those actions and the update cost. This is evidence about this prototype policy, "
            "not evidence of a universally optimal selector."
        )
    elif best_name == "No Action":
        why = (
            "The frozen baseline achieved the highest whole-stream score in this run. This can happen when "
            "the original concept remains dominant, drift is weak or late, or adaptation costs more than it "
            "helps on the evaluated observations. Check post-drift accuracy before concluding that no update is best."
        )
    else:
        why = "The winner's mechanism and the observed error profile should be considered together; one run cannot establish causation."

    if fastest:
        fastest_text = (
            f"{fastest} had the lowest measured update time among strategies that performed at least one update "
            f"({_fmt(strategies[fastest].get('update_time_seconds'), 5)} seconds)."
        )
    else:
        fastest_text = "No strategy performed a measured model update in this run."

    if best_name and best_name != "SELF-ADAPT" and self_adapt.get("accuracy") is not None:
        gap = best_acc - self_adapt["accuracy"]
        self_text = (
            f"SELF-ADAPT was {abs(gap) * 100:.2f} percentage points "
            f"{'behind' if gap >= 0 else 'ahead of'} the overall accuracy winner. "
            f"It made {self_adapt.get('update_count', 0)} update(s)."
        )
    elif self_adapt.get("accuracy") is not None:
        self_text = f"SELF-ADAPT accuracy was {_pct(self_adapt['accuracy'])} with {self_adapt.get('update_count', 0)} updates."
    else:
        self_text = "SELF-ADAPT could not be evaluated because no labels were available."

    return {
        "winner_names": winners,
        "winner": best_name,
        "verdict": verdict,
        "why": why,
        "fastest": fastest_text,
        "self_adapt_comparison": self_text,
        "late_drift": late_drift,
    }


def _accuracy_chart(result):
    names = list(result.get("strategies", {}))
    values = [result["strategies"][n].get("accuracy") for n in names]
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    palette = ["#7A8798", "#D98232", "#27845B", "#5576A8", "#6E63A8"]
    bars = ax.bar(names, [v if v is not None else 0 for v in values], color=palette[:len(names)])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, (value or 0) + .018,
                "N/A" if value is None else f"{value:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall predictive accuracy (higher is better)")
    ax.tick_params(axis="x", rotation=18, labelsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _update_chart(result):
    names = list(result.get("strategies", {}))
    values = [result["strategies"][n].get("update_time_seconds", 0) or 0 for n in names]
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    palette = ["#7A8798", "#D98232", "#27845B", "#5576A8", "#6E63A8"]
    ax.bar(names, values, color=palette[:len(names)])
    ax.set_ylabel("Seconds")
    ax.set_title("Measured model-update time (machine-dependent)")
    ax.tick_params(axis="x", rotation=18, labelsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _phase_chart(result):
    phases = ["Before drift", "Transition", "After transition"]
    names = list(result.get("strategies", {}))
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    if not result.get("phase_metrics"):
        ax.text(.5, .5, "Phase-wise metrics were not stored for this older experiment record.",
                ha="center", va="center", wrap=True)
        ax.set_axis_off()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=145, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf
    x = np.arange(len(phases))
    width = .15
    palette = ["#7A8798", "#D98232", "#27845B", "#5576A8", "#6E63A8"]
    for i, name in enumerate(names):
        vals = [result.get("phase_metrics", {}).get(name, {}).get(p, {}).get("accuracy") for p in phases]
        ax.bar(x + (i - (len(names) - 1) / 2) * width,
               [v if v is not None else np.nan for v in vals], width,
               label=name, color=palette[i % len(palette)])
    ax.set_xticks(x, phases)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by stream phase")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _confusion_chart(result):
    names = list(result.get("strategies", {}))
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.1))
    axes = axes.flatten()
    for ax, name in zip(axes, names):
        matrix = result["strategies"][name].get("confusion_matrix", [[0, 0], [0, 0]])
        matrix = np.asarray(matrix, dtype=int)
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"], fontsize=7)
        ax.set_yticks([0, 1], ["Actual 0", "Actual 1"], fontsize=7)
        threshold = max(1, int(matrix.max())) / 2
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                        color="white" if matrix[row, col] > threshold else "black", fontsize=10)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices for all strategies", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .95])
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _table(rows, widths, font=7.2):
    table = Table(rows, repeatRows=1, colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#D6DCE3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, IVORY]),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _same_setup_groups(results):
    keys = ("drift_type", "dataset_size", "batch_size", "drift_start",
            "transition_duration", "label_availability", "label_delay")
    groups = defaultdict(list)
    for result in results:
        cfg = result.get("config", {})
        key = tuple(cfg.get(k) for k in keys)
        groups[key].append(result)
    return [group for group in groups.values() if len(group) >= 2]


def _append_repeated_seed_summary(story, results, styles):
    groups = _same_setup_groups(results)
    if not groups:
        return
    story += [Spacer(1, 5 * mm), Paragraph("Repeated-seed summary", styles["NavyHead"]),
              Paragraph(
                  "The following averages include only experiments with the same drift and configuration but different random seeds. "
                  "Mean and standard deviation describe run-to-run variation; they are not a substitute for a formal statistical test.",
                  styles["BodyText"]), Spacer(1, 3 * mm)]
    for group_no, group in enumerate(groups, 1):
        cfg = group[0].get("config", {})
        story.append(Paragraph(
            f"Study group {group_no}: {escape(str(cfg.get('drift_type')))} drift, "
            f"stream {cfg.get('dataset_size')}, batch {cfg.get('batch_size')}, "
            f"{len(group)} seeds", styles["Heading3"]))
        rows = [["Strategy", "Mean accuracy", "SD", "Mean F1", "Mean update time (s)"]]
        for name in group[0].get("strategies", {}):
            acc = [r["strategies"][name].get("accuracy") for r in group if r["strategies"][name].get("accuracy") is not None]
            f1 = [r["strategies"][name].get("f1") for r in group if r["strategies"][name].get("f1") is not None]
            times = [r["strategies"][name].get("update_time_seconds", 0) for r in group]
            rows.append([name, _fmt(mean(acc)) if acc else "N/A",
                         _fmt(stdev(acc), 4) if len(acc) > 1 else "0.0000",
                         _fmt(mean(f1)) if f1 else "N/A",
                         _fmt(mean(times), 5) if times else "N/A"])
        story += [_table(rows, [42 * mm, 27 * mm, 18 * mm, 22 * mm, 38 * mm], 6.8), Spacer(1, 4 * mm)]


def build_pdf_report(results, title="SELF-ADAPT Research Report"):
    if not results:
        raise ValueError("At least one experiment is required.")
    output = BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm, title=title,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Cover", parent=styles["Title"], fontSize=27,
                              textColor=NAVY, leading=32, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="NavyHead", parent=styles["Heading2"],
                              textColor=NAVY, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"],
                              fontSize=7.6, leading=10, textColor=MUTED))
    styles.add(ParagraphStyle(name="BodyTight", parent=styles["BodyText"],
                              fontSize=8.5, leading=12, spaceAfter=4))
    styles.add(ParagraphStyle(name="StrategyText", parent=styles["BodyText"],
                              fontSize=7.7, leading=10.2, spaceAfter=3))

    story = [
        Spacer(1, 24 * mm),
        Paragraph("SELF-ADAPT", styles["Cover"]),
        Paragraph("Choosing the Right Model Update Strategy under Concept Drift", styles["Heading2"]),
        Spacer(1, 8 * mm),
        Paragraph(escape(title), styles["Heading1"]),
        Spacer(1, 5 * mm),
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}<br/>"
            f"Experiments included: {len(results)}", styles["BodyText"]),
        Spacer(1, 14 * mm),
        Paragraph(
            "<b>How to use this report:</b> Start with the executive overview, then read each experiment's "
            "setup, comparison, confusion matrices and phase-wise performance. The report explains what was "
            "tested, why it matters, what happened and what can reasonably be concluded.", styles["BodyText"]),
        Spacer(1, 8 * mm),
        Paragraph(
            "<b>Research integrity:</b> All reported metrics are read from saved experiment records. "
            "No accuracy values are invented or adjusted to make a preferred strategy win. A single run is "
            "descriptive evidence and does not establish statistical significance.", styles["SmallNote"]),
        PageBreak(),
        Paragraph("Executive overview", styles["NavyHead"]),
        Paragraph(
            "<b>Research question:</b> When the relationship between incoming observations and their labels "
            "changes, which update strategy maintains predictive performance at a reasonable adaptation cost? "
            "Each strategy receives the same synthetic stream and the same eligible label feedback within an experiment.",
            styles["BodyTight"]),
    ]

    overview_rows = [["Experiment", "Drift", "Stream", "Batch", "Seed", "Labels", "Accuracy winner"]]
    for result in results:
        cfg = result.get("config", {})
        summary = result.get("summary", {})
        overview_rows.append([
            result.get("experiment_id", "—"), str(cfg.get("drift_type", "—")),
            str(cfg.get("dataset_size", "—")), str(cfg.get("batch_size", "—")),
            str(cfg.get("seed", "—")), str(summary.get("evaluated_observations", 0)),
            str(summary.get("best_accuracy_strategy") or "No evaluation"),
        ])
    story += [Spacer(1, 3 * mm), _table(overview_rows,
        [22 * mm, 23 * mm, 18 * mm, 15 * mm, 14 * mm, 17 * mm, 48 * mm], 6.5)]
    _append_repeated_seed_summary(story, results, styles)

    for idx, result in enumerate(results, 1):
        cfg = result.get("config", {})
        drift = result.get("drift", {})
        strategies = result.get("strategies", {})
        analysis = analyze_experiment(result)
        summary = result.get("summary", {})
        drift_kind = str(cfg.get("drift_type", "unknown"))
        if drift_kind == "Abrupt":
            transition_description = "Abrupt drift changes at one point; there is no transition interval."
        else:
            transition_description = f"The configured transition duration was {float(cfg.get('transition_duration', 0)) * 100:.1f}% of the stream."
        story += [
            PageBreak(),
            Paragraph(f"Experiment {idx}: {escape(str(result.get('experiment_id', '—')))}", styles["NavyHead"]),
            Paragraph("1. What was tested and why", styles["Heading3"]),
            Paragraph(
                f"This run used the <b>{escape(str(cfg.get('drift_type', 'unknown')))} drift</b> scenario with "
                f"<b>{cfg.get('dataset_size', 'N/A')} observations</b> and a batch size of "
                f"<b>{cfg.get('batch_size', 'N/A')}</b>. Drift began at observation "
                f"<b>{drift.get('start_index', 'N/A')}</b> (about "
                f"{float(cfg.get('drift_start', 0)) * 100:.1f}% of the stream). The configured transition "
                f"duration was {float(cfg.get('transition_duration', 0)) * 100:.1f}%. "
                f"Label availability was {float(cfg.get('label_availability', 1)) * 100:.1f}% and label delay "
                f"was {cfg.get('label_delay', 0)} observations. Random seed: {cfg.get('seed', 'N/A')}.",
                styles["BodyTight"]),
            Paragraph(
                "<b>Why these settings matter:</b> Stream size controls how much evidence is collected; batch size "
                "controls how often batch feedback can trigger an update; label availability and delay control how "
                "much feedback the strategies can actually use. The random seed makes this synthetic run repeatable.",
                styles["BodyTight"]),
            Paragraph("2. How the experiment works", styles["Heading3"]),
            Paragraph(
                "The simulator creates a labelled stream and an initial training set. At each batch, all five "
                "strategies predict first. Only after predictions are stored are eligible labels released. "
                "Available labels are used for evaluation and subsequent updates. This is batch-prequential "
                "evaluation: the model is not allowed to learn a batch before predicting that same batch.",
                styles["BodyTight"]),
            Paragraph("3. Who won and what does it mean?", styles["Heading3"]),
            Paragraph(f"<b>{escape(analysis['verdict'])}</b>", styles["BodyTight"]),
            Paragraph(escape(analysis["why"]), styles["BodyTight"]),
            Paragraph(escape(analysis["fastest"]), styles["BodyTight"]),
            Paragraph(escape(analysis["self_adapt_comparison"]), styles["BodyTight"]),
            RLImage(_accuracy_chart(result), width=174 * mm, height=68 * mm),
            Paragraph("4. Complete strategy comparison", styles["Heading3"]),
        ]

        metric_rows = [["Strategy", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Update s", "Updates"]]
        for name, metric in strategies.items():
            metric_rows.append([
                name, _fmt(metric.get("accuracy")), _fmt(metric.get("precision")),
                _fmt(metric.get("recall")), _fmt(metric.get("f1")),
                _fmt(metric.get("roc_auc")), _fmt(metric.get("update_time_seconds"), 4),
                str(metric.get("update_count", 0)),
            ])
        story.append(_table(metric_rows,
            [35 * mm, 18 * mm, 19 * mm, 17 * mm, 14 * mm, 18 * mm, 23 * mm, 14 * mm], 6.4))
        story += [Spacer(1, 3 * mm), Paragraph(
            "<b>Metric guide:</b> Accuracy = proportion correct. Precision = how reliable positive predictions were. "
            "Recall = how many actual positive cases were found. F1 balances precision and recall. ROC-AUC measures "
            "ranking quality across thresholds. Update time and update count describe adaptation effort; measured "
            "time depends on the computer and software environment and measures update/fitting operations rather than "
            "prediction latency or total application runtime.", styles["SmallNote"])]

        story += [PageBreak(), Paragraph(f"Experiment {idx}: error analysis and interpretation", styles["NavyHead"]),
                  Paragraph("5. Confusion matrices: what kinds of mistakes did each strategy make?", styles["Heading3"]),
                  Paragraph(
                      "Each matrix uses rows for the actual class and columns for the predicted class. "
                      "Class 0 is negative and class 1 is positive. Top-left = true negatives (TN); top-right = "
                      "false positives (FP); bottom-left = false negatives (FN); bottom-right = true positives (TP). "
                      "False positives are negative cases incorrectly flagged as positive. False negatives are "
                      "positive cases the model missed.", styles["BodyTight"]),
                  RLImage(_confusion_chart(result), width=172 * mm, height=108 * mm)]

        cm_rows = [["Strategy", "TN", "FP", "FN", "TP", "Specificity"]]
        for name, metric in strategies.items():
            tn, fp, fn, tp = _counts(metric)
            specificity = metric.get("specificity", tn / (tn + fp) if (tn + fp) else None)
            cm_rows.append([name, str(tn), str(fp), str(fn), str(tp), _fmt(specificity)])
        story += [_table(cm_rows, [42 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm, 30 * mm], 7),
                  Spacer(1, 4 * mm), Paragraph("6. Did the strategy handle the drift itself?", styles["Heading3"]),
                  Paragraph(
                      "Overall accuracy can hide what happened after the drift point. The following chart separates "
                      "the stream into observations before drift, the transition interval and observations after "
                      "the transition. For recurring drift, the final phase combines later returning concepts and "
                      "must be interpreted with that limitation.", styles["BodyTight"]),
                  RLImage(_phase_chart(result), width=172 * mm, height=68 * mm)]
        if drift_kind == "Recurring":
            story.append(Paragraph(
                "Recurring drift revisits earlier concepts. The 'After transition' segment combines multiple later phases, "
                "so use the rolling-accuracy timeline to inspect each return separately.", styles["SmallNote"]))

        phase_rows = [["Strategy", "Before drift", "Transition", "After transition"]]
        for name in strategies:
            pm = result.get("phase_metrics", {}).get(name, {})
            phase_rows.append([name,
                _fmt(pm.get("Before drift", {}).get("accuracy")),
                _fmt(pm.get("Transition", {}).get("accuracy")),
                _fmt(pm.get("After transition", {}).get("accuracy"))])
        story.append(_table(phase_rows, [48 * mm, 36 * mm, 36 * mm, 42 * mm], 6.8))
        if analysis["late_drift"]:
            story.append(Paragraph(
                "<b>Important interpretation:</b> Drift began in the final 20% of this stream. Whole-stream accuracy "
                "is therefore dominated by the earlier concept. A high overall score does not by itself mean the "
                "strategy adapted successfully; compare the after-transition column and rolling-accuracy curve.",
                styles["SmallNote"]))

        story += [PageBreak(), Paragraph(f"Experiment {idx}: practical meaning", styles["NavyHead"]),
                  Paragraph("7. What each approach contributed", styles["Heading3"])]
        for name, metric in strategies.items():
            accuracy = metric.get("accuracy")
            if analysis["winner"] and accuracy is not None and summary.get("best_accuracy") is not None:
                gap = summary["best_accuracy"] - accuracy
                score_line = "This strategy tied for the highest accuracy." if abs(gap) < 1e-12 else f"Its accuracy was {gap * 100:.2f} percentage points below the top score."
            else:
                score_line = "There were not enough available labels to compare its accuracy."
            action_counts = metric.get("action_counts", {})
            actions = ", ".join(f"{k}: {v}" for k, v in action_counts.items()) or "No update action recorded"
            story.append(KeepTogether([
                Paragraph(f"<b>{escape(name)}</b>", styles["StrategyText"]),
                Paragraph(escape(STRATEGY_MECHANICS.get(name, "")), styles["StrategyText"]),
                Paragraph(
                    f"Observed result: accuracy {_fmt(accuracy)}, F1 {_fmt(metric.get('f1'))}, "
                    f"update time {_fmt(metric.get('update_time_seconds'), 5)} seconds, "
                    f"{metric.get('update_count', 0)} update(s). {escape(score_line)} "
                    f"Recorded actions: {escape(actions)}.", styles["StrategyText"]),
            ]))

        story += [Spacer(1, 3 * mm), Paragraph("8. What practical decision can be made?", styles["Heading3"]),
                  Paragraph(
                      "If the immediate objective is highest predictive accuracy on this exact stream, the measured "
                      "winner is the strongest candidate for this run. If update cost is important, compare its "
                      "update time and count with the other strategies. If missed positive cases are costly, pay "
                      "particular attention to recall and false negatives; if false alarms are costly, inspect "
                      "precision and false positives. The best operational choice depends on the cost of each error.",
                      styles["BodyTight"]),
                  Paragraph("9. What this result does not prove", styles["Heading3"]),
                  Paragraph(
                      "This is a synthetic binary-classification experiment. A single random seed cannot establish "
                      "a general ranking, and measured update time is machine-dependent. SELF-ADAPT uses transparent "
                      "prototype thresholds rather than a learned or formally optimal policy. Results should be "
                      "repeated across multiple seeds under the same configuration, then validated on suitable real "
                      "datasets before making broad claims.", styles["BodyTight"])]
        if result.get("drift_events"):
            story.append(Paragraph(
                f"SELF-ADAPT recorded {len(result['drift_events'])} error-triggered policy event(s). "
                "These are policy actions based on available prediction errors; they are not independent proof "
                "of the true drift cause.", styles["SmallNote"]))

    story += [PageBreak(), Paragraph("Methodology, limitations and next research steps", styles["NavyHead"]),
              Paragraph("What is controlled", styles["Heading3"]),
              Paragraph(
                  "Within each experiment, all strategies receive the same generated observations, initial training "
                  "data, label-availability draws and feedback schedule. Each batch is predicted before eligible "
                  "labels are released. This makes the within-run comparison controlled at the stream and feedback level.",
                  styles["BodyTight"]),
              Paragraph("What should be repeated", styles["Heading3"]),
              Paragraph(
                  "Repeat the exact same configuration with several different random seeds. Compare mean and "
                  "standard deviation for accuracy, F1 and update cost. Do not combine experiments with different "
                  "drift types, stream sizes, batch sizes or feedback settings into one average. Use separate "
                  "groups for separate configurations.", styles["BodyTight"]),
              Paragraph("Limitations", styles["Heading3"]),
              Paragraph(
                  "The stream generator is synthetic and binary. The current SELF-ADAPT policy is heuristic and its "
                  "thresholds need ablation and systematic tuning. Delayed or unavailable labels restrict what any "
                  "strategy can know. Wall-clock update timings vary by machine. The report is an experimental "
                  "analysis aid, not a peer-reviewed statistical conclusion.", styles["BodyTight"]),
              Paragraph("Recommended next steps", styles["Heading3"]),
              Paragraph(
                  "(1) Run at least five seeds for every chosen configuration. (2) Compare abrupt, gradual, "
                  "incremental and recurring drift separately. (3) Vary stream size, batch size, label availability "
                  "and delay one factor at a time. (4) Inspect post-drift accuracy and confusion matrices, not only "
                  "whole-stream accuracy. (5) Add real datasets and statistical tests before final research claims.",
                  styles["BodyTight"])]

    doc.build(story)
    return output.getvalue()
