import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="Maison Mode | Fashion Insights",
    page_icon="✦",
    layout="wide",
)

# ----------------------------- CUSTOM THEME ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

    :root {
        --page: #FFF9F6;
        --ink: #282329;
        --ink-soft: #4C414A;
        --plum: #594052;
        --rose: #E8C5D1;
        --lavender: #E8DFF0;
        --sage: #DCE8DE;
        --card: #F4E8ED;
        --border: #E5D3DB;
        --accent: #9B526F;
    }

    .stApp {
        background: var(--page);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 1.7rem;
        padding-bottom: 2.5rem;
    }

    /* Make all Streamlit markdown headings readable, not pale/white. */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3,
    h1, h2, h3, h4 {
        color: var(--ink) !important;
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"] {
        color: var(--ink-soft);
    }

    .hero {
        background: linear-gradient(120deg, #F3DCE4 0%, #EAE1F0 52%, #DCE9E0 100%);
        border: 1px solid var(--border);
        border-radius: 25px;
        padding: 30px 28px;
        text-align: center;
        margin-bottom: 25px;
    }

    .hero .eyebrow {
        color: #493541 !important;
        letter-spacing: 4px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #382B36 !important;
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: clamp(2.2rem, 4vw, 2.8rem);
        line-height: 1.2;
        margin: 0.2rem 0 0.9rem;
    }

    .hero p {
        color: #493541 !important;
        font-weight: 500;
    }

    .hero .tagline {
        margin-top: 1.15rem;
        letter-spacing: 0.5px;
        font-size: 0.93rem;
    }

    /* Labels and input text stay dark on the pastel background. */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    label,
    .stSelectbox label,
    .stSlider label {
        color: #332A32 !important;
        font-weight: 600 !important;
    }

    [data-baseweb="select"] > div {
        border-color: #D8C7D0 !important;
    }

    [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
        color: #332A32 !important;
    }

    div.stButton > button {
        background: #594052 !important;
        color: #FFFFFF !important;
        border: 1px solid #594052 !important;
        border-radius: 11px !important;
        width: 100%;
        min-height: 46px;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background: #704F64 !important;
        border-color: #704F64 !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stMetric"] {
        background: #F2E5EB;
        border: 1px solid #E5D3DB;
        border-radius: 13px;
        padding: 16px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #332A32 !important;
    }

    .prediction-label {
        display: inline-block;
        background: #E8C5D1;
        color: #382B36 !important;
        padding: 7px 11px;
        border: 1px solid #D7AEBE;
        border-radius: 7px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .prediction-value {
        background: #F3E8ED;
        border: 1px solid #E5D3DB;
        border-radius: 9px;
        padding: 13px;
        color: #282329 !important;
        font-weight: 600;
        min-height: 45px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E5D3DB;
        border-radius: 10px;
        overflow: hidden;
    }

    hr {
        border-color: #E5D3DB !important;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- HERO BANNER -----------------------------
st.markdown(
    """
    <div class="hero">
        <p class="eyebrow">THE PERSONAL STYLE EDIT</p>
        <h1>Maison Mode ✦</h1>
        <p>Find your style. Discover your match.</p>
        <p class="tagline">PERSONALIZED FASHION · SMART INSIGHTS · YOUR AESTHETIC</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- DATA LOADING -----------------------------
FEATURES = [
    "price_usd",
    "trend_score",
    "comfort_score",
    "quality_score",
    "fit_score",
]
REQUIRED_COLUMNS = [
    "price_usd",
    "clothing_type",
    "occasion",
    "trend_score",
    "comfort_score",
    "quality_score",
    "fit_score",
    "purchased",
    "satisfaction_score",
]


@st.cache_data
def load_dataset():
    """Find and validate the project CSV from common project locations."""
    candidates = [
        Path(__file__).resolve().parent / "fashion_dataset_complete.csv",
        Path(__file__).resolve().parent / "data" / "fashion_dataset_complete.csv",
        Path("/content/fashion_dataset_complete.csv"),  # Google Colab fallback
        Path("fashion_dataset_complete.csv"),
    ]

    csv_path = next((path for path in candidates if path.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "Could not find fashion_dataset_complete.csv. Add the CSV to the same "
            "GitHub folder as app.py, or place it inside a data/ folder."
        )

    data = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing)
            + ". Please upload the complete fashion dataset."
        )

    for column in FEATURES + ["satisfaction_score", "purchased"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["price_usd", "clothing_type", "occasion"])
    if data.empty:
        raise ValueError("The dataset has no usable rows after cleaning.")
    return data


try:
    df = load_dataset()
except Exception as error:
    st.error(f"Unable to load the fashion dataset: {error}")
    st.stop()

# ----------------------------- USER PREFERENCES -------------------------
st.markdown("### ♡ Curate your look")
st.write(
    "Tell us what matters to you. We'll match your preferences with products "
    "in the fashion dataset."
)

col1, col2 = st.columns(2)
clothing_options = ["Any"] + sorted(
    df["clothing_type"].dropna().astype(str).unique().tolist()
)
occasion_options = ["Any"] + sorted(
    df["occasion"].dropna().astype(str).unique().tolist()
)

with col1:
    clothing = st.selectbox("What are you shopping for?", clothing_options)
with col2:
    occasion = st.selectbox("The occasion", occasion_options)

price_min = max(0, int(np.floor(df["price_usd"].min())))
price_max = max(price_min + 1, int(np.ceil(df["price_usd"].max())))
budget_default = int(np.clip(df["price_usd"].median(), price_min, price_max))
budget = st.slider(
    "Your maximum budget (USD)",
    min_value=price_min,
    max_value=price_max,
    value=budget_default,
)

st.markdown("### ✨ Your style priorities")
style_col1, style_col2 = st.columns(2)

with style_col1:
    trend = st.slider("Trendiness", min_value=0, max_value=100, value=70)
    quality = st.slider("Quality", min_value=0, max_value=10, value=8)

with style_col2:
    comfort = st.slider("Comfort", min_value=0, max_value=10, value=8)
    fit = st.slider("Fit", min_value=0, max_value=10, value=8)

# ----------------------------- RECOMMENDATIONS --------------------------
if st.button("Find My Fashion Match  ♡"):
    products = df[df["price_usd"] <= budget].copy()

    if clothing != "Any":
        products = products[products["clothing_type"].astype(str) == clothing]
    if occasion != "Any":
        products = products[products["occasion"].astype(str) == occasion]

    purchase_result = "Not enough data to predict"
    satisfaction_result = "Not enough data to predict"

    if products.empty:
        st.warning(
            "No matching items found. Try increasing your budget or selecting "
            "'Any' for clothing type or occasion."
        )
    else:
        # Dataset trend scores are assumed to use a 0–10 scale.
        preferences = {
            "trend_score": trend / 10.0,
            "comfort_score": float(comfort),
            "quality_score": float(quality),
            "fit_score": float(fit),
        }

        score = pd.Series(0.0, index=products.index)
        used_features = 0

        for column, preference in preferences.items():
            low = df[column].min()
            high = df[column].max()
            if pd.notna(low) and pd.notna(high) and high > low:
                normalized = (products[column] - low) / (high - low)
                target = (preference - low) / (high - low)
                score += 1 - (normalized - target).abs().clip(0, 1)
                used_features += 1

        products["Style Match (%)"] = (
            100 * score / max(used_features, 1)
        ).round(1)
        products = products.sort_values(
            "Style Match (%)", ascending=False
        ).head(8)

        st.caption(
            f"Found {len(products)} recommended styles within your budget. "
            "Ranked by similarity to your preferences."
        )
        st.markdown("#### Your Curated Style Matches")

        display_columns = [
            "clothing_type",
            "occasion",
            "price_usd",
            "trend_score",
            "comfort_score",
            "quality_score",
            "fit_score",
            "Style Match (%)",
        ]
        st.dataframe(
            products[display_columns].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        input_row = pd.DataFrame(
            [
                {
                    "price_usd": budget,
                    "trend_score": trend / 10.0,
                    "comfort_score": comfort,
                    "quality_score": quality,
                    "fit_score": fit,
                }
            ],
            columns=FEATURES,
        )

        # Purchase likelihood model
        purchase_data = df[FEATURES + ["purchased"]].dropna().copy()
        purchase_data = purchase_data[
            purchase_data["purchased"].isin([0, 1])
        ]

        if len(purchase_data) >= 10 and purchase_data["purchased"].nunique() == 2:
            purchase_model = make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=120,
                    random_state=42,
                    class_weight="balanced",
                ),
            )
            purchase_model.fit(
                purchase_data[FEATURES],
                purchase_data["purchased"].astype(int),
            )
            probabilities = purchase_model.predict_proba(input_row)[0]
            classes = list(purchase_model[-1].classes_)
            if 1 in classes:
                probability = probabilities[classes.index(1)]
                purchase_result = f"{probability:.1%} estimated purchase likelihood"

        # Satisfaction score model
        satisfaction_data = df[FEATURES + ["satisfaction_score"]].dropna()
        if len(satisfaction_data) >= 10:
            satisfaction_model = make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(
                    n_estimators=120,
                    random_state=42,
                ),
            )
            satisfaction_model.fit(
                satisfaction_data[FEATURES],
                satisfaction_data["satisfaction_score"],
            )
            prediction = float(satisfaction_model.predict(input_row)[0])
            satisfaction_result = f"{prediction:.2f} predicted satisfaction score"

    st.markdown("### ♡ Your style predictions")
    prediction_col1, prediction_col2 = st.columns(2)

    with prediction_col1:
        st.markdown(
            '<div class="prediction-label">Purchase Likelihood</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="prediction-value">{purchase_result}</div>',
            unsafe_allow_html=True,
        )

    with prediction_col2:
        st.markdown(
            '<div class="prediction-label">Predicted Satisfaction</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="prediction-value">{satisfaction_result}</div>',
            unsafe_allow_html=True,
        )

# ----------------------------- FOOTER -----------------------------------
st.divider()
st.markdown("*MAISON MODE · Fashion Insights Studio*")
st.caption(
    "Recommendations are ranked using product attributes in the dataset. "
    "Predictions are estimates learned from historical data, not guarantees."
)
