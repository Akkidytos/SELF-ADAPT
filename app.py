from __future__ import annotations
import json, os, hmac
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from self_adapt.engine import STRATEGIES, run_experiment
from self_adapt.storage import load_experiments, save_experiment, get_experiment
from self_adapt.reports import build_pdf_report, analyze_experiment

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="SELF-ADAPT | Research Lab", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
css = BASE / "assets" / "styles.css"
if css.exists():
    st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def hero(title, subtitle, chip="INDIAN AI RESEARCH PLATFORM"):
    st.markdown(f'<div class="sa-hero"><span class="sa-chip">{chip}</span><h1>{title}</h1><p>{subtitle}</p><div style="height:12px"></div><div class="sa-tricolor"></div></div>', unsafe_allow_html=True)


def login():
    left, right = st.columns([1.15,.85], gap="large")
    with left:
        st.markdown("""
        <div style="background:radial-gradient(circle at 85% 12%,rgba(217,130,50,.28),transparent 30%),linear-gradient(140deg,#172B46,#244A6B);padding:42px 38px;border-radius:24px;color:#F7F5F0;min-height:510px;border-bottom:5px solid #D98232">
          <div style="font-size:13px;letter-spacing:3px;color:#F3C38D;font-weight:700">INDIAN AI RESEARCH EDITION</div>
          <div style="font-size:48px;margin-top:34px">◉</div>
          <h1 style="color:white!important;font-size:42px;margin:8px 0">SELF-ADAPT</h1>
          <h3 style="color:#F5D8B7!important;font-size:22px">Intelligence that adapts.</h3>
          <p style="color:#E3EAF1;line-height:1.7">A research workspace for studying model update strategies under changing data concepts.</p>
          <div style="margin-top:60px;color:#D7E1EA;font-size:13px">RESEARCH · EVALUATE · ADAPT</div>
          <div style="height:5px;border-radius:10px;margin-top:16px;background:linear-gradient(90deg,#E88B36 0 33.3%,#F7F5F0 33.3% 66.6%,#27845B 66.6%)"></div>
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        st.markdown("## Welcome back")
        st.caption("Sign in to open your research workspace.")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        expected_user = os.getenv("SELF_ADAPT_USER", "researcher")
        expected_password = os.getenv("SELF_ADAPT_PASSWORD", "adapt2026")
        if st.button("Sign in to SELF-ADAPT →", type="primary", width="stretch"):
            if hmac.compare_digest(username, expected_user) and hmac.compare_digest(password, expected_password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Username or password is incorrect.")
        st.info("Local demo login: researcher / adapt2026. Change it with environment variables before sharing.")
        st.caption("This is a local prototype access gate, not production-grade authentication.")


def fmt(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def result_frame(result):
    rows=[]
    for name,m in result["strategies"].items():
        cm=m.get("confusion_matrix",[[0,0],[0,0]])
        tn,fp,fn,tp=int(cm[0][0]),int(cm[0][1]),int(cm[1][0]),int(cm[1][1])
        specificity=m.get("specificity",tn/(tn+fp) if (tn+fp) else None)
        rows.append({"Strategy":name,"Accuracy":m["accuracy"],"Precision":m["precision"],
                     "Recall":m["recall"],"F1-score":m["f1"],"ROC-AUC":m["roc_auc"],
                     "Specificity":specificity, "TN":tn, "FP":fp, "FN":fn, "TP":tp,
                     "Update time (s)":m["update_time_seconds"],"Model updates":m["update_count"],
                     "Evaluated labels":m["n_evaluated"]})
    return pd.DataFrame(rows)


def metric_chart(result, metric="accuracy"):
    df=result_frame(result)
    label={"accuracy":"Accuracy","precision":"Precision","recall":"Recall","f1":"F1-score","roc_auc":"ROC-AUC"}[metric]
    fig=px.bar(df,x="Strategy",y=label,color="Strategy",text_auto=".3f",
               color_discrete_sequence=["#7A8798","#D98232","#27845B","#5576A8","#6E63A8"])
    fig.update_layout(showlegend=False,yaxis_range=[0,1],height=360,margin=dict(l=10,r=10,t=35,b=10))
    return fig


def overview(experiments):
    hero("Research overview","Understand how update strategies behave when the underlying data concept changes.")
    if not experiments:
        st.info("No experiments saved yet. Start with **New Experiment** to generate your first reproducible result.")
        st.markdown("### Your research workflow")
        cols=st.columns(4)
        cards=[("01","Configure","Choose drift, stream size, label delay and seed."),
               ("02","Simulate","Generate a reproducible data stream."),
               ("03","Compare","Evaluate five update policies under the same conditions."),
               ("04","Explain","Read the summary and export a PDF report.")]
        for c,(num,title,desc) in zip(cols,cards):
            with c:
                st.markdown(f'<div class="sa-card"><span class="sa-chip" style="background:#172B46">{num}</span><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
        return
    latest=experiments[0]
    s=latest["summary"]
    cols=st.columns(4)
    cols[0].metric("Saved experiments",len(experiments))
    cols[1].metric("Latest drift",latest["config"].get("drift_type","—"))
    cols[2].metric("Best accuracy",fmt(s.get("best_accuracy")),s.get("best_accuracy_strategy") or "Not evaluated")
    cols[3].metric("Evaluated labels",s.get("evaluated_observations",0))
    st.subheader("Latest experiment")
    st.markdown(f"**{latest['experiment_id']}** · {latest['config'].get('drift_type')} drift · seed {latest['config'].get('seed')}")
    st.markdown(f'<div class="sa-note">{s.get("interpretation","No summary available.")}</div>',unsafe_allow_html=True)
    st.plotly_chart(metric_chart(latest),width="stretch")
    st.caption("A single run is descriptive. Repeat across multiple seeds before making general research claims.")


def experiment_lab():
    hero("Experiment lab","Configure a controlled stream and compare all strategies under identical conditions.")
    with st.form("experiment_form"):
        st.markdown("#### 1 · Stream configuration")
        c1,c2,c3=st.columns(3)
        with c1:
            drift=st.selectbox("Concept drift scenario",["Abrupt","Gradual","Incremental","Recurring"],
                help="Abrupt = sudden switch; Gradual = overlapping concepts; Incremental = moving boundary; Recurring = a previous concept returns.")
            n=st.number_input("Stream size",400,12000,2400,400)
            batch=st.number_input("Batch size",20,int(n),min(80,int(n)),20)
        with c2:
            start=st.slider("Drift starts at (%)",.20,.95,.55,.05)
            duration=st.slider("Transition duration (%)",.02,.40,.12,.02,help="Used by gradual/incremental and recurring drift. Abrupt drift changes at one point and ignores this duration.")
            seed=st.number_input("Random seed",0,999999,42,1)
            repeat_seeds=st.checkbox("Repeat with 5 seeds (recommended for research)",value=False,
                help="Runs the same configuration with five consecutive seeds, then lets the PDF report calculate mean and standard deviation.")
        with c3:
            availability=st.slider("Label availability (%)",0,100,100,5)/100
            delay=st.number_input("Label delay (observations)",0,2000,0,20)
            st.selectbox("Evaluation protocol",["Batch-prequential: predict before batch labels are released"],disabled=True)
        submitted=st.form_submit_button("Run controlled experiment",type="primary",width="stretch")
    st.markdown('<div class="sa-note"><b>Beginner tip:</b> Start with 2,400 observations, abrupt drift, 100% labels and zero delay. Change one setting at a time to keep comparisons interpretable.</div>',unsafe_allow_html=True)
    st.info("Want to test a case where Incremental Learning may not win overall? Try: Abrupt drift · stream size 5,000 · batch size 500 · drift starts at 90% · seed 42 · 100% labels · zero delay. In one checked run, Full Retraining scored 0.8436 vs Incremental Learning 0.8348. Important: only the final 500 observations were post-drift, so whole-stream accuracy was dominated by the old concept; post-drift accuracy was poor for every strategy. Repeat across seeds before drawing a conclusion.")
    if submitted:
        config={"drift_type":drift,"dataset_size":int(n),"batch_size":int(batch),
                "drift_start":float(start),"transition_duration":float(duration),"seed":int(seed),
                "label_availability":float(availability),"label_delay":int(delay)}
        with st.spinner("Generating stream(s), evaluating strategies and saving experiment records…"):
            try:
                seeds = [int(seed) + offset for offset in range(5)] if repeat_seeds else [int(seed)]
                completed_ids=[]
                for run_seed in seeds:
                    run_config={**config,"seed":run_seed}
                    result=run_experiment(run_config)
                    save_experiment(result)
                    completed_ids.append(result["experiment_id"])
                st.session_state["show_result_ids"]=completed_ids
                st.success(f"Saved {len(completed_ids)} experiment run(s): {', '.join(completed_ids)}")
            except Exception as exc:
                st.exception(exc)
    result_ids=st.session_state.get("show_result_ids",[])
    study_results=[get_experiment(exp_id) for exp_id in result_ids]
    study_results=[item for item in study_results if item]
    if study_results:
        st.divider()
        if len(study_results)>1:
            st.subheader("Repeated-seed study summary")
            seed_rows=[]
            for item in study_results:
                seed_rows.append({"Experiment":item["experiment_id"],"Seed":item["config"].get("seed"),
                    "Best strategy":item["summary"].get("best_accuracy_strategy"),
                    "Best accuracy":item["summary"].get("best_accuracy")})
            st.dataframe(pd.DataFrame(seed_rows),hide_index=True,width="stretch")
            aggregate_rows=[]
            for strategy in STRATEGIES:
                accuracies=[item["strategies"][strategy]["accuracy"] for item in study_results
                            if item["strategies"][strategy]["accuracy"] is not None]
                f1s=[item["strategies"][strategy]["f1"] for item in study_results
                     if item["strategies"][strategy]["f1"] is not None]
                times=[item["strategies"][strategy]["update_time_seconds"] for item in study_results]
                aggregate_rows.append({"Strategy":strategy,
                    "Mean accuracy":float(np.mean(accuracies)) if accuracies else None,
                    "Accuracy SD":float(np.std(accuracies,ddof=1)) if len(accuracies)>1 else 0.0,
                    "Mean F1":float(np.mean(f1s)) if f1s else None,
                    "Mean update time (s)":float(np.mean(times)) if times else None})
            st.markdown("#### Across-seed averages")
            st.dataframe(pd.DataFrame(aggregate_rows),hide_index=True,width="stretch")
            st.caption("These are descriptive mean and standard deviation values for the same configuration across different seeds. They are not a formal significance test.")
            detail_id=st.selectbox("Open detailed result for seed",[item["experiment_id"] for item in study_results],
                                   index=len(study_results)-1,key="detailed_seed_result")
            r=next(item for item in study_results if item["experiment_id"]==detail_id)
        else:
            r=study_results[0]
        st.subheader(f"Detailed result snapshot · {r['experiment_id']}")
        st.markdown(f'<div class="sa-note">{r["summary"]["interpretation"]}</div>',unsafe_allow_html=True)
        st.dataframe(result_frame(r),hide_index=True,width="stretch")
        analysis = analyze_experiment(r)
        st.markdown("#### What happened in this experiment?")
        st.write(analysis["verdict"])
        st.write(analysis["why"])
        st.write(analysis["fastest"])
        st.write(analysis["self_adapt_comparison"])
        if r.get("phase_metrics"):
            phase_rows=[]
            for strategy, phases in r["phase_metrics"].items():
                phase_rows.append({"Strategy":strategy,
                    "Before drift":phases.get("Before drift",{}).get("accuracy"),
                    "Transition":phases.get("Transition",{}).get("accuracy"),
                    "After transition":phases.get("After transition",{}).get("accuracy")})
            st.markdown("#### Did it handle the drift itself?")
            st.dataframe(pd.DataFrame(phase_rows),hide_index=True,width="stretch")


def comparison(experiments):
    hero("Strategy comparison","Compare predictive quality, update cost and update frequency without hiding trade-offs.")
    if not experiments:
        st.info("Run an experiment first.")
        return
    ids=[r["experiment_id"] for r in experiments]
    selected=st.selectbox("Select experiment",ids)
    r=next(x for x in experiments if x["experiment_id"]==selected)
    metric=st.selectbox("Predictive metric",["accuracy","precision","recall","f1","roc_auc"],
                        format_func=lambda x:{"accuracy":"Accuracy","precision":"Precision","recall":"Recall","f1":"F1-score","roc_auc":"ROC-AUC"}[x])
    st.plotly_chart(metric_chart(r,metric),width="stretch")
    df=result_frame(r)
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Measured update time")
        f=px.bar(df,x="Strategy",y="Update time (s)",color="Strategy",
                 color_discrete_sequence=["#7A8798","#D98232","#27845B","#5576A8","#6E63A8"])
        f.update_layout(showlegend=False,height=330)
        st.plotly_chart(f,width="stretch")
    with c2:
        st.subheader("Successful update count")
        f=px.bar(df,x="Strategy",y="Model updates",color="Strategy",
                 color_discrete_sequence=["#7A8798","#D98232","#27845B","#5576A8","#6E63A8"])
        f.update_layout(showlegend=False,height=330)
        st.plotly_chart(f,width="stretch")
    st.subheader("Accuracy vs computational cost")
    pareto=df.dropna(subset=["Accuracy"])
    f=px.scatter(pareto,x="Update time (s)",y="Accuracy",text="Strategy",color="Strategy",
                 size="Model updates",size_max=24,
                 color_discrete_sequence=["#7A8798","#D98232","#27845B","#5576A8","#6E63A8"])
    f.update_traces(textposition="top center")
    f.update_layout(height=390,yaxis_range=[0,1])
    st.plotly_chart(f,width="stretch")
    st.caption("Higher points have higher accuracy; points farther left use less measured update time. The preferred trade-off depends on the research objective.")
    st.subheader("Detailed metrics")
    st.dataframe(df,hide_index=True,width="stretch")

    analysis = analyze_experiment(r)
    st.subheader("Experiment interpretation")
    st.info(analysis["verdict"])
    st.write(analysis["why"])
    st.write(analysis["fastest"])
    st.write(analysis["self_adapt_comparison"])

    st.subheader("Confusion matrix explorer")
    selected_strategy = st.selectbox("Choose a strategy", list(r["strategies"]), key="cm_strategy")
    cm = r["strategies"][selected_strategy].get("confusion_matrix", [[0,0],[0,0]])
    cm_fig = px.imshow(cm, text_auto=True, x=["Predicted 0", "Predicted 1"],
                       y=["Actual 0", "Actual 1"], color_continuous_scale="Blues",
                       labels={"x":"Predicted class", "y":"Actual class", "color":"Count"},
                       aspect="auto")
    cm_fig.update_layout(height=330, title=f"{selected_strategy} · confusion matrix")
    st.plotly_chart(cm_fig,width="stretch")
    st.caption("Top-left = true negatives (TN); top-right = false positives (FP); bottom-left = false negatives (FN); bottom-right = true positives (TP). Class 0 is negative and class 1 is positive.")

    if r.get("phase_metrics"):
        st.subheader("Performance before and after drift")
        phase_rows=[]
        for strategy, phases in r["phase_metrics"].items():
            phase_rows.append({"Strategy":strategy,
                "Before drift":phases.get("Before drift",{}).get("accuracy"),
                "Transition":phases.get("Transition",{}).get("accuracy"),
                "After transition":phases.get("After transition",{}).get("accuracy")})
        st.dataframe(pd.DataFrame(phase_rows),hide_index=True,width="stretch")
        st.caption("Whole-stream accuracy can hide weak post-drift performance, especially when drift starts near the end of the stream.")
    st.markdown('<div class="sa-note"><b>How to read this:</b> Predictive metrics describe quality; confusion matrices show the types of mistakes; phase metrics show whether performance changed after drift; update time and count describe adaptation effort. The highest-accuracy strategy is not automatically the best operational choice.</div>',unsafe_allow_html=True)


def drift_analysis(experiments):
    hero("Drift impact analysis","See how prequential performance changes as the stream evolves.")
    if not experiments:
        st.info("Run an experiment to view performance timelines.")
        return
    ids=[r["experiment_id"] for r in experiments]
    r=next(x for x in experiments if x["experiment_id"]==st.selectbox("Experiment",ids,key="drift_exp"))
    df=pd.DataFrame(r["timeline"])
    if df.empty or df["rolling_accuracy"].notna().sum()==0:
        st.warning("No labels became available. Increase label availability or reduce label delay.")
        return
    f=px.line(df,x="stream_end",y="rolling_accuracy",color="strategy",markers=True,
              color_discrete_sequence=["#7A8798","#D98232","#27845B","#5576A8","#6E63A8"])
    f.add_vline(x=r["drift"]["start_index"],line_dash="dash",line_color="#D98232",annotation_text="Simulated drift onset")
    if r["drift"]["transition_end_index"]>r["drift"]["start_index"]:
        f.add_vrect(x0=r["drift"]["start_index"],x1=r["drift"]["transition_end_index"],fillcolor="#D98232",opacity=.08,line_width=0)
    f.update_layout(yaxis_range=[0,1],height=460,xaxis_title="Stream observation",yaxis_title="Rolling accuracy")
    st.plotly_chart(f,width="stretch")
    st.markdown('<div class="sa-note"><b>Beginner explanation:</b> Each line tracks recent accuracy using labels that have arrived. The vertical marker shows where the simulator introduced drift. Accuracy changes show how a strategy handled the changed stream; they do not identify the cause by themselves.</div>',unsafe_allow_html=True)
    if r.get("phase_metrics"):
        st.subheader("Accuracy by stream phase")
        phase_rows=[]
        for strategy, phases in r["phase_metrics"].items():
            phase_rows.append({"Strategy":strategy,
                "Before drift":phases.get("Before drift",{}).get("accuracy"),
                "Transition":phases.get("Transition",{}).get("accuracy"),
                "After transition":phases.get("After transition",{}).get("accuracy")})
        st.dataframe(pd.DataFrame(phase_rows),hide_index=True,width="stretch")
    if r["drift_events"]:
        st.subheader("SELF-ADAPT policy events")
        st.dataframe(pd.DataFrame(r["drift_events"]),hide_index=True,width="stretch")
    else:
        st.info("No SELF-ADAPT error-triggered update events were recorded in this run.")


def history(experiments):
    hero("Experiment history","Every run is saved independently in a local SQLite database.")
    if not experiments:
        st.info("No saved experiments yet.")
        return
    rows=[]
    for r in experiments:
        rows.append({"Experiment ID":r["experiment_id"],"Created":r["created_at"],
                     "Drift":r["config"].get("drift_type"),"Seed":r["config"].get("seed"),
                     "Labels evaluated":r["summary"].get("evaluated_observations",0),
                     "Best strategy":r["summary"].get("best_accuracy_strategy") or "N/A",
                     "Best accuracy":r["summary"].get("best_accuracy")})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    if len(experiments)>=2:
        st.subheader("Previous vs current experiment")
        ids=[r["experiment_id"] for r in experiments]
        a_col,b_col=st.columns(2)
        with a_col:
            aid=st.selectbox("Baseline experiment",ids,index=min(1,len(ids)-1))
        with b_col:
            bid=st.selectbox("Comparison experiment",ids,index=0)
        a=next(r for r in experiments if r["experiment_id"]==aid)
        b=next(r for r in experiments if r["experiment_id"]==bid)
        metric=st.selectbox("Metric",["accuracy","f1","update_time_seconds","update_count"],
            format_func=lambda x:{"accuracy":"Accuracy","f1":"F1-score","update_time_seconds":"Update time (s)","update_count":"Update count"}[x])
        comp=[]
        for name in STRATEGIES:
            av=a["strategies"][name].get(metric); bv=b["strategies"][name].get(metric)
            comp.append({"Strategy":name,"Baseline":av,"Comparison":bv,
                         "Difference":bv-av if av is not None and bv is not None else None})
        st.dataframe(pd.DataFrame(comp),hide_index=True,width="stretch")
        st.caption("Difference = comparison minus baseline. Lower time/count means less measured update effort; higher predictive metrics are generally better.")


def reports(experiments):
    hero("Research reports & export","Generate individual or combined PDF reports from saved experiment records.")
    if not experiments:
        st.info("Run and save an experiment before exporting a report.")
        return
    ids=[r["experiment_id"] for r in experiments]
    selected=st.multiselect("Experiments to include in PDF",ids,default=[ids[0]])
    if selected:
        chosen=[r for r in experiments if r["experiment_id"] in selected]
        try:
            pdf=build_pdf_report(chosen,"SELF-ADAPT Combined Research Report" if len(chosen)>1 else "SELF-ADAPT Experiment Report")
            st.download_button("Download PDF research report",data=pdf,file_name="SELF-ADAPT_Research_Report.pdf",
                               mime="application/pdf",type="primary",width="stretch")
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")
        rows=[]
        for r in chosen:
            for name,m in r["strategies"].items():
                cm=m.get("confusion_matrix",[[0,0],[0,0]])
                tn,fp,fn,tp=int(cm[0][0]),int(cm[0][1]),int(cm[1][0]),int(cm[1][1])
                rows.append({"experiment_id":r["experiment_id"],"drift":r["config"]["drift_type"],"strategy":name,
                             **{k:m.get(k) for k in ["accuracy","precision","recall","f1","roc_auc","update_time_seconds","update_count","n_evaluated"]},
                             "specificity":m.get("specificity",tn/(tn+fp) if (tn+fp) else None),
                             "tn":tn,"fp":fp,"fn":fn,"tp":tp})
        st.download_button("Download comparison CSV",data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                           file_name="SELF-ADAPT_Strategy_Comparison.csv",mime="text/csv",width="stretch")
    st.subheader("Reproducibility records")
    for r in experiments:
        with st.expander(f"{r['experiment_id']} · {r['config'].get('drift_type')} · seed {r['config'].get('seed')}"):
            st.json({"experiment_id":r["experiment_id"],"created_at":r["created_at"],"config":r["config"],"drift":r["drift"]})
            st.download_button(f"Download {r['experiment_id']} JSON",data=json.dumps(r,indent=2),
                file_name=f"{r['experiment_id']}_results.json",mime="application/json",key=f"json_{r['experiment_id']}")


def academy():
    hero("SELF-ADAPT Academy","A quick guide to the concepts behind the charts and experiment settings.")
    glossary={
        "Concept drift":"The relationship between input data and the target changes over time, so an older model may become less accurate.",
        "Prequential evaluation":"The model predicts first. Only after a label arrives is the observation used for evaluation and possible learning.",
        "Accuracy":"The fraction of evaluated predictions that are correct.",
        "Precision":"Among observations predicted positive, the fraction that were truly positive.",
        "Recall":"Among truly positive observations, the fraction successfully identified.",
        "F1-score":"The harmonic mean of precision and recall; it balances both.",
        "Label delay":"How long the learner waits before a ground-truth label becomes available.",
        "Label availability":"The fraction of stream observations whose labels eventually become available.",
        "Incremental learning":"Updating an existing model with new labelled examples instead of fitting from scratch.",
        "Full retraining":"Fitting a fresh model again using available labelled history.",
        "Probability recalibration":"Adjusting predicted probabilities with labelled calibration observations while keeping the underlying classifier frozen.",
        "Pareto trade-off":"A strategy is attractive when no alternative is both more accurate and less costly.",
        "Confusion matrix":"A 2x2 table that counts true negatives, false positives, false negatives and true positives. It shows which kinds of prediction errors a model makes.",
        "Phase-wise accuracy":"Accuracy calculated separately before drift, during a transition and after the transition. It helps prevent a long pre-drift period from hiding weak adaptation after drift."
    }
    for term,desc in glossary.items():
        with st.expander(term): st.write(desc)
    st.subheader("The five strategies")
    st.markdown("""
    - **No Action:** keeps the original model unchanged; it is a baseline.
    - **Probability Recalibration:** fits a calibration layer from labelled feedback while keeping the base classifier frozen.
    - **Incremental Learning:** updates the existing classifier using `partial_fit`.
    - **Full Retraining:** fits a fresh classifier on the available labelled history.
    - **SELF-ADAPT:** uses a transparent prototype rule to choose among no update, recalibration, incremental update or full retraining using recent available prediction errors and feedback volume.
    """)
    st.warning("SELF-ADAPT is an experimental rule-based policy, not a proven optimal strategy. Validate across repeated seeds and real datasets.")
    st.info("Research tip: never judge a strategy only by whole-stream accuracy. Read its confusion matrix and post-drift accuracy too.")


def main():
    if not st.session_state.get("authenticated"):
        login()
        return
    experiments=load_experiments()
    with st.sidebar:
        st.markdown("## ◉ SELF-ADAPT")
        st.caption("INDIAN AI RESEARCH EDITION")
        st.markdown('<div class="sa-tricolor"></div>',unsafe_allow_html=True)
        st.write(f"Researcher: **{st.session_state.get('username','researcher')}**")
        page=st.radio("WORKSPACE",["Overview","New Experiment","Strategy Comparison","Drift Analysis",
                                   "Experiment History","Research Reports","Beginner Academy"])
        st.divider()
        st.caption(f"{len(experiments)} saved experiment(s)")
        if st.button("Log out", width="stretch", key="logout_button"):
            st.session_state["authenticated"] = False
            st.session_state.pop("username", None)
            st.rerun()
    if page=="Overview": overview(experiments)
    elif page=="New Experiment": experiment_lab()
    elif page=="Strategy Comparison": comparison(experiments)
    elif page=="Drift Analysis": drift_analysis(experiments)
    elif page=="Experiment History": history(experiments)
    elif page=="Research Reports": reports(experiments)
    elif page=="Beginner Academy": academy()


if __name__=="__main__":
    main()
