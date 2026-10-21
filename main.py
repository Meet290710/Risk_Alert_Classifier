import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from Model.model import load_dataset, train_and_evaluate, prepare_features, TARGET

st.set_page_config(page_title="Risk Alert Classifier", page_icon="🎯", layout="wide")

st.title("🎯 Risk Alert Classifier")
st.write(
    "A machine-learning dashboard to identify potentially high-risk customers. "
    "Upload the supplied dataset, train a classifier, and review recall, F1-score and ROC-AUC."
)
st.info("Target: `risk_status` — 0 = Low Risk, 1 = High Risk. This demo is for educational use, not real credit decisions.")

with st.sidebar:
    st.header("Model settings")
    model_name = st.selectbox("Classification model", [
        "Logistic Regression", "Decision Tree", "Random Forest"
    ], index=2)
    sampling_method = st.selectbox("Imbalance handling", [
        "None", "Random Under-Sampling", "Random Over-Sampling", "SMOTE", "ADASYN"
    ], index=3)
    tune = st.checkbox("Tune hyperparameters (slower)", value=False)
    test_size = st.slider("Test set size", min_value=0.15, max_value=0.35, value=0.20, step=0.05)

uploaded = st.file_uploader("Upload your risk-alert CSV dataset", type=["csv"])
if uploaded is None:
    st.warning("Upload `Risk_Alert_Classifier_Dataset_4600(1).csv` to begin.")
    st.stop()

try:
    data = load_dataset(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

st.subheader("Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(data):,}")
c2.metric("Columns", f"{len(data.columns):,}")
c3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
with st.expander("Preview data"):
    st.dataframe(data.head(20), use_container_width=True)
with st.expander("Target class distribution"):
    if TARGET in data.columns:
        counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)
        st.bar_chart(pd.DataFrame({"Customers": counts.values}, index=["Low Risk (0)", "High Risk (1)"]))
        st.write("Class counts:", {"Low Risk (0)": int(counts.loc[0]), "High Risk (1)": int(counts.loc[1])})
    else:
        st.error(f"Required target column `{TARGET}` is missing.")

if st.button("🚀 Train and evaluate model", type="primary", use_container_width=True):
    with st.spinner("Training model and evaluating on a held-out test set..."):
        try:
            result = train_and_evaluate(
                data, model_name=model_name, sampling_method=sampling_method,
                tune=tune, test_size=test_size
            )
            st.session_state["risk_result"] = result
            st.session_state["risk_config"] = {
                "model_name": model_name, "sampling_method": sampling_method, "tune": tune
            }
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.stop()

if "risk_result" in st.session_state:
    result = st.session_state["risk_result"]
    metrics = result["metrics"]
    st.divider()
    st.subheader("Evaluation results")
    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    b.metric("Precision", f"{metrics['precision']:.3f}")
    c.metric("Recall / TPR", f"{metrics['recall']:.3f}")
    d.metric("F1-score", f"{metrics['f1_score']:.3f}")
    e.metric("ROC-AUC", "N/A" if pd.isna(metrics["auc_roc"]) else f"{metrics['auc_roc']:.3f}")

    st.caption("For high-risk detection, pay particular attention to recall and false negatives; accuracy alone can be misleading on imbalanced data.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix=metrics["confusion_matrix"],
            display_labels=["Low Risk", "High Risk"]
        ).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### ROC curve")
        if metrics["roc_curve"] is not None:
            fpr, tpr, _ = metrics["roc_curve"]
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {metrics['auc_roc']:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            ax.grid(True, alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.write("ROC curve is unavailable for this model.")
    st.markdown("#### Classification report")
    report = pd.DataFrame(metrics["classification_report"]).transpose()
    st.dataframe(report.round(3), use_container_width=True)
    if metrics["best_params"]:
        st.markdown("#### Best hyperparameters")
        st.json(metrics["best_params"])
    st.markdown("#### Class counts in training split")
    st.write({
        "Before sampling - Low Risk": int(metrics["train_class_counts"].get(0, 0)),
        "Before sampling - High Risk": int(metrics["train_class_counts"].get(1, 0)),
        "Test Low Risk": int(metrics["test_class_counts"].get(0, 0)),
        "Test High Risk": int(metrics["test_class_counts"].get(1, 0)),
    })

    st.divider()
    st.subheader("Predict risk for customer records")
    st.write("Upload a separate CSV containing the same feature columns. The target `risk_status` and `customer_id` are optional and will not be used as model inputs.")
    prediction_file = st.file_uploader("Upload customer records for prediction", type=["csv"], key="prediction_csv")
    if prediction_file is not None:
        try:
            new_data = pd.read_csv(prediction_file)
            input_features = prepare_features(new_data.drop(columns=[TARGET], errors="ignore"))
            # Align columns to the training feature schema; absent columns are filled with missing values.
            expected = metrics["feature_columns"]
            input_features = input_features.reindex(columns=expected)
            predicted = result["pipeline"].predict(input_features)
            try:
                probability = result["pipeline"].predict_proba(input_features)[:, 1]
            except (AttributeError, IndexError):
                probability = [float("nan")] * len(predicted)
            output = new_data.copy()
            output["predicted_risk_status"] = predicted.astype(int)
            output["predicted_label"] = ["High Risk" if p == 1 else "Low Risk" for p in predicted]
            output["high_risk_probability"] = probability
            st.dataframe(output.head(100), use_container_width=True)
            st.download_button(
                "Download predictions CSV",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="risk_predictions.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Prediction failed. Check that the uploaded file has the same feature columns. Details: {exc}")

st.divider()
st.caption("Project scope: Logistic Regression baseline, Decision Tree and Random Forest, imbalance sampling, model evaluation, ROC analysis and optional hyperparameter tuning.")
