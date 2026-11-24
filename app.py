
from pathlib import Path
import sys, io, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, confusion_matrix
import joblib

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.preprocess import load_data, prepare_splits, FEATURES
from src.train_iforest import train_isolation_forest, iforest_scores
from src.train_xgb import train_xgboost
from src.evaluate import choose_threshold, evaluate_scores
from src.reporting import build_markdown, save_pdf

st.set_page_config(page_title="FraudGuard | FinTech Intelligence", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")
css = (ROOT/"assets/style.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

TEAL, RED, AMBER, VIOLET = "#00D9B5", "#FF4D6D", "#FFB703", "#7C5CFF"
BG = "#121B2E"

@st.cache_data(show_spinner=False)
def get_data():
    return load_data()

@st.cache_resource(show_spinner=False)
def get_splits(data_json):
    df = pd.read_json(io.StringIO(data_json), orient="split")
    return prepare_splits(df)

def money(x):
    return f"${x:,.2f}"

df, data_source = get_data()
st.session_state.setdefault("trained", False)
st.session_state.setdefault("results", {})
st.session_state.setdefault("models", {})
st.session_state.setdefault("thresholds", {})

with st.sidebar:
    st.markdown("## 🛡️ FraudGuard")
    st.caption("FINTECH • RISK INTELLIGENCE")
    page = st.radio("NAVIGATION", ["Overview", "Data Explorer", "Model Training",
                                  "Evaluation Report", "Live Predictor"])
    st.divider()
    st.caption("DATA SOURCE")
    st.write(data_source)
    st.caption(f"{len(df):,} rows • {int(df.Class.sum()):,} fraud")
    if st.button("Clear cached data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

def plot_theme(fig, height=360):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=12,r=12,t=45,b=12),
                      font=dict(family="Inter", color="#E6EDF7"))
    return fig

def ensure_models():
    if not st.session_state.trained:
        st.warning("Train models in the Model Training page to unlock evaluation and live predictions.")
        return False
    return True

if page == "Overview":
    st.title("Fraud Intelligence Overview")
    st.caption("A high-signal view of transaction risk, model performance and portfolio exposure.")
    total, fraud = len(df), int(df.Class.sum())
    pct = 100*fraud/max(total,1)
    best_recall = max([v["recall"] for v in st.session_state.results.values()], default=0)
    best_f1 = max([v["f1"] for v in st.session_state.results.values()], default=0)
    cols = st.columns(5)
    for col, label, value in zip(cols,
        ["Total transactions","Fraud detected in data","Fraud prevalence","Best model recall","Best model F1"],
        [f"{total:,}",f"{fraud:,}",f"{pct:.3f}%",f"{best_recall:.3f}",f"{best_f1:.3f}"]):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1,1.3])
    with left:
        st.subheader("Class distribution")
        counts = df.Class.value_counts().reindex([0,1], fill_value=0)
        fig = go.Figure(go.Pie(labels=["Legitimate","Fraud"], values=counts.values,
            hole=.68, marker=dict(colors=[TEAL,RED]), textinfo="label+percent"))
        fig.update_layout(showlegend=False, annotations=[dict(text=f"{pct:.2f}%<br>fraud",x=.5,y=.5,showarrow=False,font=dict(size=18))])
        st.plotly_chart(plot_theme(fig, 340), use_container_width=True)
    with right:
        st.subheader("Transaction amount distribution")
        sample = df.sample(min(12000,len(df)), random_state=42)
        fig = px.histogram(sample, x="Amount", color=sample.Class.map({0:"Legitimate",1:"Fraud"}),
                           nbins=65, barmode="overlay", opacity=.72,
                           color_discrete_map={"Legitimate":TEAL,"Fraud":RED})
        st.plotly_chart(plot_theme(fig,340), use_container_width=True)
    st.info("Accuracy can look excellent on this dataset even when a model misses most fraud. Use recall, precision, F1 and PR-AUC to judge fraud detection.")

elif page == "Data Explorer":
    st.title("Data Explorer")
    st.caption("Inspect class balance, transaction patterns and anonymized feature relationships.")
    c1,c2,c3 = st.columns(3)
    with c1: cls = st.selectbox("Transaction class",["All","Legitimate","Fraud"])
    with c2: amount_max = st.slider("Maximum amount", float(df.Amount.min()), float(df.Amount.quantile(.99)), float(df.Amount.quantile(.95)))
    with c3: rows = st.slider("Rows to display", 100, 5000, 1000, step=100)
    view = df[df.Amount <= amount_max]
    if cls != "All": view = view[view.Class == (1 if cls=="Fraud" else 0)]
    st.dataframe(view.head(rows), use_container_width=True, height=300)
    a,b = st.columns(2)
    with a:
        st.subheader("Amount by class")
        fig=px.box(view.sample(min(10000,len(view)),random_state=42),x="Class",y="Amount",
                   color="Class",color_discrete_map={0:TEAL,1:RED},points=False)
        st.plotly_chart(plot_theme(fig),use_container_width=True)
    with b:
        st.subheader("Time pattern")
        hourly=(view.assign(Hour=(view.Time//3600)%48).groupby(["Hour","Class"]).size().reset_index(name="Count"))
        fig=px.line(hourly,x="Hour",y="Count",color="Class",
                    color_discrete_map={0:TEAL,1:RED})
        st.plotly_chart(plot_theme(fig),use_container_width=True)
    qc1, qc2 = st.columns(2)
    qc1.metric("Missing values", int(df.isna().sum().sum()))
    qc2.metric("Duplicate rows", int(df.duplicated().sum()))
    st.subheader("Correlation heatmap")
    corr=df.sample(min(12000,len(df)),random_state=42).corr(numeric_only=True)
    fig=px.imshow(corr,zmin=-1,zmax=1,color_continuous_scale="RdBu_r",aspect="auto")
    st.plotly_chart(plot_theme(fig,650),use_container_width=True)
    st.subheader("Largest absolute correlations with Class")
    top=corr["Class"].drop("Class").abs().sort_values(ascending=False).head(12).sort_values()
    fig=px.bar(x=top.values,y=top.index,orientation="h",labels={"x":"|Correlation|","y":"Feature"})
    fig.update_traces(marker_color=VIOLET)
    st.plotly_chart(plot_theme(fig,380),use_container_width=True)

elif page == "Model Training":
    st.title("Model Training Lab")
    st.caption("Stratified split • training-only resampling • validation-based threshold selection")
    left,right=st.columns([1,1])
    with left:
        contamination=st.number_input("Isolation Forest contamination",min_value=0.0001,max_value=0.05,value=max(0.0001,min(.05,float(df.Class.mean()))),format="%.4f")
        n_estimators=st.slider("Isolation Forest estimators",100,600,300,50)
        max_samples=st.selectbox("Isolation Forest max_samples", ["auto", 256, 1024, 4096], index=0)
        max_depth=st.slider("XGBoost max depth",2,10,4)
        learning_rate=st.select_slider("XGBoost learning rate",options=[.01,.03,.05,.08,.1,.15],value=.05)
    with right:
        sampling=st.slider("SMOTE minority / majority ratio",0.1,1.0,0.5,0.1)
        xgb_estimators=st.slider("XGBoost estimators",100,700,350,50)
        min_precision=st.slider("Minimum validation precision floor",0.01,0.90,0.10,0.01)
        st.markdown("**Leakage safeguards**")
        st.markdown("- Split before scaling/resampling\n- Fit scaler on training rows only\n- SMOTE training rows only\n- Threshold tuned on a validation subset of training rows\n- Test set remains untouched")
    Xtr,Xte,ytr,yte,scaler=get_splits(df.to_json(orient="split"))
    counts_before=ytr.value_counts().reindex([0,1],fill_value=0)
    st.subheader("Class counts before SMOTE")
    st.write(pd.DataFrame({"Class":["Legitimate","Fraud"],"Training rows":counts_before.values}))
    if st.button("🚀 Train and evaluate models", use_container_width=True):
        with st.spinner("Training models. This may take a few minutes..."):
            # Validation split is carved from the training partition only.
            from sklearn.model_selection import train_test_split
            X_fit,X_val,y_fit,y_val=train_test_split(Xtr,ytr,test_size=.2,stratify=ytr,random_state=42)
            iforest=train_isolation_forest(X_fit,contamination,n_estimators,max_samples)
            if_val=iforest_scores(iforest,X_val)
            if_thr=choose_threshold(y_val,if_val,min_precision)
            if_test=iforest_scores(iforest,Xte)
            if_metrics=evaluate_scores(yte,if_test,if_thr)
            xgb, X_res, y_res=train_xgboost(X_fit,y_fit,sampling,max_depth,learning_rate,xgb_estimators)
            xgb_val=xgb.predict_proba(X_val)[:,1]
            xgb_thr=choose_threshold(y_val,xgb_val,min_precision)
            xgb_test=xgb.predict_proba(Xte)[:,1]
            xgb_metrics=evaluate_scores(yte,xgb_test,xgb_thr)
            st.session_state.models={"Isolation Forest":iforest,"XGBoost + SMOTE":xgb}
            st.session_state.results={"Isolation Forest":if_metrics,"XGBoost + SMOTE":xgb_metrics}
            st.session_state.thresholds={"Isolation Forest":if_thr,"XGBoost + SMOTE":xgb_thr}
            st.session_state.scaler=scaler
            st.session_state.trained=True
            st.session_state.smote_counts=y_res.value_counts().to_dict()
            st.session_state.feature_names=list(Xtr.columns)
            joblib.dump({"models":st.session_state.models,"scaler":scaler,
                         "thresholds":st.session_state.thresholds,
                         "feature_names":list(Xtr.columns)},ROOT/"models/fraud_models.joblib")
        st.success("Training complete. Models and thresholds saved to models/fraud_models.joblib.")
        st.rerun()
    if st.session_state.trained:
        st.subheader("Class counts after SMOTE (XGBoost training partition)")
        st.dataframe(pd.DataFrame({"Class":["Legitimate","Fraud"],
            "Rows":[st.session_state.smote_counts.get(0,0),st.session_state.smote_counts.get(1,0)]}),use_container_width=True)
        st.success("Models are ready. Open Evaluation Report or Live Predictor.")

elif page == "Evaluation Report":
    st.title("Evaluation Report")
    if ensure_models():
        results=st.session_state.results
        table=[]
        for name,m in results.items():
            table.append({"Model":name,"Recall":m["recall"],"Precision":m["precision"],
                          "F1":m["f1"],"PR-AUC":m["pr_auc"],"ROC-AUC":m["roc_auc"],"Accuracy":m["accuracy"]})
        st.subheader("Side-by-side model comparison")
        st.dataframe(pd.DataFrame(table).style.format({k:"{:.4f}" for k in ["Recall","Precision","F1","PR-AUC","ROC-AUC","Accuracy"]})
                     .highlight_max(subset=["Recall","F1"],color="#244D55"),use_container_width=True)
        st.caption("Accuracy is included only as context; it is not the selection metric.")
        mdf=pd.DataFrame(table).melt(id_vars="Model",value_vars=["Recall","Precision","F1"],var_name="Metric",value_name="Score")
        fig=px.bar(mdf,x="Model",y="Score",color="Metric",barmode="group",
                   color_discrete_map={"Recall":TEAL,"Precision":AMBER,"F1":VIOLET},text_auto=".3f")
        st.plotly_chart(plot_theme(fig),use_container_width=True)
        cols=st.columns(2)
        for col,(name,m) in zip(cols,results.items()):
            with col:
                st.subheader(name)
                cm=np.array(m["confusion_matrix"])
                fig=px.imshow(cm,text_auto=True,x=["Predicted legit","Predicted fraud"],
                              y=["Actual legit","Actual fraud"],color_continuous_scale="Blues")
                st.plotly_chart(plot_theme(fig,300),use_container_width=True)
                st.metric("Fraud recall",f"{m['recall']:.3f}")
                st.metric("Fraud F1",f"{m['f1']:.3f}")
        st.subheader("Precision–Recall curves")
        Xtr,Xte,ytr,yte,scaler=get_splits(df.to_json(orient="split"))
        fig=go.Figure()
        for name,model in st.session_state.models.items():
            scores=( -model.score_samples(Xte) if name=="Isolation Forest"
                    else model.predict_proba(Xte)[:,1])
            p,r,_=precision_recall_curve(yte,scores)
            fig.add_trace(go.Scatter(x=r,y=p,mode="lines",name=name))
        fig.update_layout(xaxis_title="Recall",yaxis_title="Precision")
        st.plotly_chart(plot_theme(fig),use_container_width=True)
        st.subheader("Feature importance — XGBoost")
        xgb=st.session_state.models["XGBoost + SMOTE"]
        imp=pd.Series(xgb.feature_importances_,index=st.session_state.feature_names).sort_values().tail(15)
        fig=px.bar(x=imp.values,y=imp.index,orientation="h",labels={"x":"Importance","y":"Feature"})
        fig.update_traces(marker_color=VIOLET)
        st.plotly_chart(plot_theme(fig,420),use_container_width=True)
        st.subheader("Business interpretation")
        st.markdown("- **False negatives:** missed fraudulent transactions can create direct losses and chargebacks.\n- **False positives:** legitimate customers may be declined, creating friction and lost revenue.\n- Choose the operating threshold with fraud-loss estimates, customer friction costs and analyst review capacity.")
        best=max(results,key=lambda k:results[k]["f1"])
        recommendation=f"Based on the current held-out test F1, **{best}** is the leading candidate (F1={results[best]['f1']:.4f}, recall={results[best]['recall']:.4f}). Confirm the trade-off against business costs and validate on time-separated, real transactions before deployment."
        report=build_markdown(results,recommendation,len(df),int(df.Class.sum()))
        st.download_button("⬇️ Download Markdown report",report,file_name="evaluation_report.md",mime="text/markdown")
        pdf_path=ROOT/"reports/evaluation_report.pdf"
        save_pdf(report,pdf_path)
        st.download_button("⬇️ Download PDF report",pdf_path.read_bytes(),file_name="evaluation_report.pdf",mime="application/pdf")
        (ROOT/"reports/evaluation_report.md").write_text(report)
    else:
        st.info("Train models first to generate the evaluation report.")

elif page == "Live Predictor":
    st.title("Live Transaction Risk Predictor")
    st.caption("Score a single transaction or batch-upload a CSV with the Kaggle feature schema.")
    if ensure_models():
        model_name=st.selectbox("Model",list(st.session_state.models.keys()))
        model=st.session_state.models[model_name]
        threshold=st.session_state.thresholds[model_name]
        st.caption(f"Operating threshold: {threshold:.6f} (selected on validation data)")
        tab1,tab2=st.tabs(["Single transaction","Batch CSV"])
        with tab1:
            with st.form("single_tx"):
                values={}
                c1,c2=st.columns(2)
                with c1:
                    values["Time"]=st.number_input("Time",0.0,172800.0,50000.0)
                    values["Amount"]=st.number_input("Amount",0.0,1000000.0,120.0)
                with c2:
                    st.write("Anonymized PCA features")
                    st.caption("Defaults are zero; enter known V1–V28 values when available.")
                with st.expander("Enter V1–V28"):
                    for i in range(1,29):
                        values[f"V{i}"]=st.number_input(f"V{i}",value=0.0,key=f"v{i}")
                submitted=st.form_submit_button("Analyze transaction")
            if submitted:
                row=pd.DataFrame([values],columns=FEATURES)
                row[["Time","Amount"]]=st.session_state.scaler.transform(row[["Time","Amount"]])
                score=float(-model.score_samples(row)[0] if model_name=="Isolation Forest" else model.predict_proba(row)[0,1])
                fraud=score>=threshold
                st.markdown(f'<span class="{"fraud-badge" if fraud else "legit-badge"}">{"⚠ FRAUD" if fraud else "✓ LEGIT"}</span>',unsafe_allow_html=True)
                st.metric("Model score",f"{score:.5f}")
                fig=go.Figure(go.Indicator(mode="gauge+number",value=score,
                    gauge={"axis":{"range":[0,1]},"bar":{"color":RED if fraud else TEAL},
                           "steps":[{"range":[0,threshold],"color":"#1C2A40"},{"range":[threshold,1],"color":"#422338"}]}))
                st.plotly_chart(plot_theme(fig,280),use_container_width=True)
                st.caption("This score is a model score, not necessarily a calibrated probability—especially for Isolation Forest.")
        with tab2:
            uploaded=st.file_uploader("Upload CSV (Time, Amount, V1–V28; Class optional)",type=["csv"])
            if uploaded:
                batch=pd.read_csv(uploaded)
                missing=set(FEATURES)-set(batch.columns)
                if missing: st.error(f"Missing columns: {sorted(missing)}")
                else:
                    X=batch[FEATURES].copy()
                    X[["Time","Amount"]]=st.session_state.scaler.transform(X[["Time","Amount"]])
                    scores=(-model.score_samples(X) if model_name=="Isolation Forest" else model.predict_proba(X)[:,1])
                    batch["fraud_score"]=scores
                    batch["prediction"]=np.where(scores>=threshold,"FRAUD","LEGIT")
                    st.dataframe(batch,use_container_width=True)
                    st.download_button("Download scored CSV",batch.to_csv(index=False),file_name="scored_transactions.csv",mime="text/csv")
