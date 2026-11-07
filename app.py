
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Forest Intelligence | AI Classifier",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. LOAD TRAINED MODEL AND METADATA
# --------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

MODEL_PATH = PROJECT / "models" / "stacking_ensemble.joblib"
META_PATH = PROJECT / "results" / "model_metadata.json"
COMPARISON_PATH = PROJECT / "results" / "model_comparison.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(META_PATH, "r") as f:
        return json.load(f)

MODEL = load_model()
META = load_metadata()

features = META["feature_names"]
class_names = {
    int(k): v for k, v in META["class_names"].items()
}

# --------------------------------------------------
# 3. PROFESSIONAL STYLING
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: #101a17;
        color: #edf5ef;
    }

    [data-testid="stSidebar"] {
        background: #172820;
        border-right: 1px solid #294536;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #e9f5ec;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #a8c2b0;
        font-size: 1rem;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #d9f2df;
        margin-top: 12px;
    }

    .info-card {
        background: linear-gradient(135deg, #1d382b, #244b35);
        border: 1px solid #386448;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .prediction-card {
        background: linear-gradient(135deg, #174d32, #246b43);
        border: 1px solid #4b9a68;
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
    }

    div.stButton > button[kind="primary"] {
        background: #25834c;
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 700;
        min-height: 45px;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #31a45e;
        color: white;
    }

    div[data-testid="stMetric"] {
        background: #1a2c23;
        border: 1px solid #345340;
        padding: 15px;
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: #e9f5ec;
    }

    .stCaption {
        color: #a8c2b0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 4. HEADER
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🌲 Forest Intelligence</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'AI-powered forest cover classification using terrain and soil data.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# 5. LOAD MODEL COMPARISON RESULTS
# --------------------------------------------------
if COMPARISON_PATH.exists():
    comparison = pd.read_csv(COMPARISON_PATH)
else:
    comparison = pd.DataFrame()

# --------------------------------------------------
# 6. NAVIGATION
# --------------------------------------------------
overview_tab, predict_tab, analytics_tab = st.tabs([
    "📊 Overview",
    "🌳 Forest Prediction",
    "📈 Model Analytics"
])

# --------------------------------------------------
# 7. OVERVIEW
# --------------------------------------------------
with overview_tab:
    st.markdown("## Project Overview")
    st.write(
        "This application predicts forest cover types using "
        "the UCI Covertype dataset and a trained Stacking Ensemble."
    )

    if not comparison.empty:
        best = comparison.sort_values(
            "Test Macro F1", ascending=False
        ).iloc[0]

        best_accuracy = comparison["Test Accuracy"].max()
        best_f1 = comparison["Test Macro F1"].max()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Algorithms evaluated", len(comparison))
        c2.metric("Best test accuracy", f"{best_accuracy:.2%}")
        c3.metric("Best test Macro F1", f"{best_f1:.4f}")
        c4.metric("Best model", str(best["Algorithm"]))

    st.markdown("### Dataset information")

    a, b, c = st.columns(3)
    a.metric("Input features", len(features))
    b.metric("Forest cover classes", len(class_names))
    c.metric("Primary algorithm", "Stacking Ensemble")

    st.markdown("### Algorithms in this study")

    st.write(
        "- Decision Tree\n"
        "- Random Forest\n"
        "- Extra Trees\n"
        "- Stacking Ensemble"
    )

    st.info(
        "The displayed performance values are measured results "
        "from the notebook evaluation, not assumed accuracy."
    )

# --------------------------------------------------
# 8. FOREST COVER PREDICTION
# --------------------------------------------------
with predict_tab:
    st.markdown("## Forest Cover Prediction")
    st.write(
        "Enter terrain measurements and select the relevant "
        "wilderness and soil indicators."
    )

    numeric_features = [
        f for f in features
        if not f.startswith("Wilderness_Area")
        and not f.startswith("Soil_Type")
    ]

    wilderness_features = [
        f for f in features if f.startswith("Wilderness_Area")
    ]

    soil_features = [
        f for f in features if f.startswith("Soil_Type")
    ]

    defaults = {
        "Elevation": 2500,
        "Aspect": 180,
        "Slope": 10,
        "Horizontal_Distance_To_Hydrology": 200,
        "Vertical_Distance_To_Hydrology": 30,
        "Horizontal_Distance_To_Roadways": 1000,
        "Hillshade_9am": 220,
        "Hillshade_Noon": 230,
        "Hillshade_3pm": 140,
        "Horizontal_Distance_To_Fire_Points": 1500
    }

    st.markdown("### 1. Terrain measurements")

    input_values = {}

    with st.form("forest_prediction_form"):

        with st.expander("Terrain and environmental measurements",
                         expanded=True):

            cols = st.columns(2)

            for i, feature in enumerate(numeric_features):
                with cols[i % 2]:
                    value = defaults.get(feature, 0)

                    if feature == "Aspect":
                        minimum, maximum = 0.0, 360.0
                    elif feature == "Slope":
                        minimum, maximum = 0.0, 90.0
                    elif feature.startswith("Hillshade"):
                        minimum, maximum = 0.0, 255.0
                    else:
                        minimum, maximum = -1000.0, 10000.0

                    input_values[feature] = st.number_input(
                        feature.replace("_", " "),
                        min_value=minimum,
                        max_value=maximum,
                        value=float(value),
                        step=1.0,
                        help=f"Enter the measured value for {feature}."
                    )

        st.markdown("### 2. Wilderness area")

        with st.expander("Select wilderness area", expanded=False):
            st.caption("Select the applicable area indicator.")
            cols = st.columns(2)

            for i, feature in enumerate(wilderness_features):
                with cols[i % 2]:
                    input_values[feature] = int(
                        st.checkbox(feature.replace("_", " "))
                    )

        st.markdown("### 3. Soil type")

        with st.expander("Select soil type", expanded=False):
            st.caption("Select the applicable soil type indicator.")
            cols = st.columns(3)

            for i, feature in enumerate(soil_features):
                with cols[i % 3]:
                    input_values[feature] = int(
                        st.checkbox(feature.replace("_", " "))
                    )

        submitted = st.form_submit_button(
            "🌲 Predict Forest Cover",
            type="primary",
            use_container_width=True
        )

    if submitted:
        sample = pd.DataFrame(
            [[input_values[f] for f in features]],
            columns=features
        )

        with st.spinner("Analyzing terrain and predicting forest cover..."):
            prediction = int(MODEL.predict(sample)[0])
            predicted_name = class_names.get(
                prediction, "Unknown class"
            )

        st.markdown(
            f"""
            <div class="prediction-card">
                <h3>Predicted Forest Cover</h3>
                <h1>Class {prediction}</h1>
                <h2>{predicted_name}</h2>
                <p>Prediction generated by the Stacking Ensemble model.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if hasattr(MODEL, "predict_proba"):
            probabilities = MODEL.predict_proba(sample)[0]

            probability_df = pd.DataFrame({
                "Forest Cover": [
                    f"{int(label)} - "
                    f"{class_names.get(int(label), 'Unknown')}"
                    for label in MODEL.classes_
                ],
                "Probability": probabilities
            }).sort_values("Probability", ascending=False)

            st.markdown("### Class probability distribution")

            chart_df = probability_df.set_index("Forest Cover")
            st.bar_chart(chart_df)

            st.dataframe(
                probability_df.style.format({
                    "Probability": "{:.2%}"
                }),
                use_container_width=True,
                hide_index=True
            )

        prediction_record = sample.copy()
        prediction_record["Predicted_Class"] = prediction
        prediction_record["Predicted_Cover_Type"] = predicted_name

        st.download_button(
            "⬇️ Download prediction",
            data=prediction_record.to_csv(index=False),
            file_name="forest_cover_prediction.csv",
            mime="text/csv"
        )

# --------------------------------------------------
# 9. MODEL ANALYTICS
# --------------------------------------------------
with analytics_tab:
    st.markdown("## Model Performance Analytics")

    if comparison.empty:
        st.warning("Run the notebook evaluation section first.")

    else:
        st.markdown("### Model comparison")

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        if "Test Accuracy" in comparison.columns:
            st.markdown("### Test accuracy comparison")

            accuracy_chart = comparison[
                ["Algorithm", "Test Accuracy"]
            ].set_index("Algorithm")

            st.bar_chart(accuracy_chart)

        if "Test Macro F1" in comparison.columns:
            st.markdown("### Test Macro F1 comparison")

            f1_chart = comparison[
                ["Algorithm", "Test Macro F1"]
            ].set_index("Algorithm")

            st.bar_chart(f1_chart)

        st.download_button(
            "⬇️ Download model comparison",
            data=comparison.to_csv(index=False),
            file_name="model_comparison.csv",
            mime="text/csv"
        )

        st.caption(
            "All models should be evaluated on the same held-out test set. "
            "Use measured values in your academic report."
        )

# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Forest Intelligence | Academic Machine Learning Project | "
    "UCI Covertype Dataset"
)
