from __future__ import annotations

import io
import os
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =============================================================================
# CARWISE | Intelligent Used-Car Valuation & Market Analytics
# Single-file Streamlit application. Keep this file at <project_root>/app/main.py
# =============================================================================

# -- Paths (pure Python, safe to run before set_page_config) -------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


def first_existing(paths: list[Path]) -> Path | None:
    for candidate in paths:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _logo_candidates() -> list[Path]:
    names = ["logo.jfif", "logo.jpg", "logo.jpeg", "logo.png"]
    folders = [PROJECT_ROOT, APP_DIR, PROJECT_ROOT / "assets", APP_DIR / "assets"]
    paths: list[Path] = []
    env_logo = os.environ.get("CARWISE_LOGO")
    if env_logo:
        paths.append(Path(env_logo))
    # Original machine location first, then portable locations.
    paths.append(Path(r"D:\INSTANT\ML_SV_PROJ\logo.jfif"))
    paths.extend(folder / name for folder in folders for name in names)
    return paths


LOGO_PATH = first_existing(_logo_candidates())
LOGO_BYTES: bytes | None = None
LOGO_IMAGE = None  # PIL image used as the browser-tab icon
if LOGO_PATH is not None:
    try:
        LOGO_BYTES = LOGO_PATH.read_bytes()
        from PIL import Image  # Pillow ships with Streamlit

        # .jfif is a JPEG; decoding from bytes avoids relying on the extension.
        LOGO_IMAGE = Image.open(io.BytesIO(LOGO_BYTES)).convert("RGB")
        LOGO_IMAGE.thumbnail((128, 128))
    except Exception:
        LOGO_BYTES = LOGO_BYTES  # keep bytes if only the favicon decode failed
        LOGO_IMAGE = None

st.set_page_config(
    page_title="CarWise | Intelligent Vehicle Valuation",
    page_icon=LOGO_IMAGE if LOGO_IMAGE is not None else "🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Streamlit API compatibility (use_container_width -> width="stretch") ------
def _st_version() -> tuple[int, int]:
    try:
        major, minor = st.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return (0, 0)


STRETCH: dict = {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}

# -- Design system ------------------------------------------------------------
BG = "#0B1220"
PANEL = "#111C2E"
PANEL_ALT = "#192A43"
GOLD = "#5B9BFF"
GOLD_LIGHT = "#8CB9FF"
TEXT = "#E8EEF7"
MUTED = "#9AAAC0"
BORDER = "rgba(140,185,255,.18)"
GOLD_SCALE = [[0, "#24436C"], [0.55, "#3979D4"], [1, GOLD]]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus Jakarta Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {{ color-scheme: dark; }}
    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
    .stApp {{
        background:
          radial-gradient(ellipse at 78% -8%, rgba(66,133,220,.13), transparent 31%),
          radial-gradient(ellipse at 0% 45%, rgba(37,75,125,.10), transparent 35%),
          {BG};
        color: {TEXT};
    }}
    [data-testid="stHeader"] {{ background: rgba(11,18,32,.82); backdrop-filter: blur(12px); }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #101B2D 0%, #0B1220 100%);
        border-right: 1px solid rgba(140,185,255,.13);
    }}
    [data-testid="stSidebar"] > div:first-child {{ padding: 1.35rem 1rem 1.6rem; }}

    /* Sidebar Logo */
    [data-testid="stSidebar"] [data-testid="stImage"] img {{
        width: 100%;
        height: auto;
        object-fit: contain;
        border: none;
        border-radius: 0;
        box-shadow: none;
    }}

    .block-container {{ padding-top: 2.15rem; padding-bottom: 3.5rem; max-width: 1500px; }}
    h1,h2,h3,h4 {{ font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; color: {TEXT}; }}
    h1 {{ font-weight: 800 !important; }}
    p, label, .stMarkdown {{ color: {TEXT}; }}

    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, #17243A 0%, #111C2E 100%);
        border: 1px solid rgba(140,185,255,.13); padding: 20px 21px;
        border-radius: 17px; box-shadow: 0 12px 30px rgba(0,0,0,.16);
        transition: transform .2s ease, border-color .2s ease; overflow: hidden;
    }}
    [data-testid="stMetric"]:hover {{ transform: translateY(-2px); border-color: rgba(140,185,255,.35); }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: .70rem !important; text-transform: uppercase; letter-spacing: .10em; font-weight: 700; }}
    [data-testid="stMetricValue"] {{ color: #E8F1FF !important; font-family: 'Manrope',sans-serif; font-weight: 800; font-size: clamp(1.25rem, 2vw, 1.9rem); }}
    [data-testid="stMetricValue"] > div {{ white-space: normal !important; overflow-wrap: anywhere; }}
    [data-testid="stMetricDelta"] {{ color: #AAB9CD !important; }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: rgba(140,185,255,.13); border-radius: 17px; background: rgba(17,28,46,.48); box-shadow: 0 12px 32px rgba(0,0,0,.12); }}

    .hero {{
        background: radial-gradient(circle at 88% 15%, rgba(75,145,235,.17), transparent 27%),
                    linear-gradient(115deg, #192A43 0%, #142238 56%, #101B2D 100%);
        border: 1px solid rgba(140,185,255,.20); border-radius: 23px;
        padding: 36px 38px; margin: 4px 0 28px; position: relative; overflow: hidden;
        box-shadow: 0 22px 55px rgba(0,0,0,.22);
    }}
    .hero:after {{ content:''; position:absolute; width:280px; height:280px; right:-100px; top:-150px;
        border:1px solid rgba(140,185,255,.22); border-radius:50%;
        box-shadow:0 0 0 28px rgba(140,185,255,.035),0 0 0 62px rgba(140,185,255,.022); pointer-events:none; }}

    .eyebrow {{ color:{GOLD}; font-size:.68rem; font-weight:700; letter-spacing:.19em; text-transform:uppercase; margin-bottom:12px; }}
    .hero-title {{ color:{TEXT}; font-family:'Manrope',sans-serif; font-size:clamp(1.8rem,3.2vw,2.9rem); line-height:1.1; font-weight:800; letter-spacing:-.055em; margin:0 0 12px; }}
    .hero-copy {{ color:#B8C6D9; font-size:.95rem; line-height:1.75; max-width:790px; margin:0; }}
    .section-kicker {{ color:{GOLD}; font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.15em; margin:4px 0 7px; }}
    .subtle {{ color:{MUTED}; font-size:.84rem; line-height:1.65; margin-bottom:8px; }}
    .price-output {{ font-family:'Manrope',sans-serif; font-size:clamp(2.3rem,5vw,4.3rem); font-weight:800; letter-spacing:-.06em; color:{GOLD_LIGHT}; line-height:1.05; }}
    .gold-rule {{ height:1px; background:linear-gradient(90deg,{GOLD},rgba(91,155,255,.08) 72%,transparent); margin:21px 0 26px; opacity:.9; }}
    .nav-label {{ color:#8799B2; font-size:.66rem; font-weight:700; letter-spacing:.16em; margin:0 0 9px 8px; }}

    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
        border:1px solid rgba(140,185,255,.55) !important; border-radius:11px !important;
        background:linear-gradient(135deg,#6CA8F5,#3979D4) !important; color:#071426 !important;
        font-weight:750 !important; min-height:2.85rem; padding:.58rem 1.2rem;
        box-shadow:0 7px 18px rgba(57,121,212,.18); transition:transform .18s ease,box-shadow .18s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
        border-color:#B7D4FF !important; background:linear-gradient(135deg,#91BEFF,#5A98ED) !important;
        transform:translateY(-1px); box-shadow:0 10px 22px rgba(57,121,212,.24);
    }}
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible, [data-testid="stFormSubmitButton"] > button:focus-visible {{ outline:2px solid {GOLD_LIGHT}; outline-offset:2px; }}

    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, textarea {{ background:#142238 !important; border:1px solid #30445F !important; border-radius:11px !important; }}
    div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"] > div:focus-within, textarea:focus {{ border-color:{GOLD} !important; box-shadow:0 0 0 1px rgba(91,155,255,.25) !important; }}

    [data-testid="stTabs"] button {{ color:{MUTED}; font-weight:650; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{ color:{GOLD_LIGHT}; border-bottom-color:{GOLD}; }}
    [data-testid="stRadio"] label {{ padding:.5rem .68rem; border-radius:9px; transition:background .15s ease; }}
    [data-testid="stRadio"] label:hover {{ background:rgba(91,155,255,.10); }}
    [data-testid="stExpander"] {{ border:1px solid rgba(140,185,255,.14); border-radius:13px; background:rgba(17,28,46,.48); }}
    [data-testid="stDataFrame"] {{ border:1px solid rgba(140,185,255,.14); border-radius:12px; overflow:hidden; }}
    [data-testid="stPlotlyChart"] {{ border:1px solid rgba(140,185,255,.10); border-radius:15px; background:rgba(17,28,46,.22); padding:7px; }}
    [data-testid="stAlert"] {{ border-radius:12px; }}
    [data-testid="stCaptionContainer"] {{ color:{MUTED}; }}
    [data-testid="stForm"] {{ border:1px solid rgba(140,185,255,.14); border-radius:17px; background:rgba(17,28,46,.5); }}
    hr {{ border-color:rgba(255,255,255,.08); }}
    footer {{ visibility:hidden; }}

    @media (max-width:900px) {{
        .block-container {{ padding:1.2rem 1rem 3rem; }}
        .hero {{ padding:27px 25px; margin-bottom:21px; }}
    }}
    @media (max-width:640px) {{
        .block-container {{ padding:.9rem .75rem 2.5rem; }}
        .hero {{ padding:21px 18px; border-radius:18px; }}
        .hero:after {{ display:none; }}
        .hero-copy {{ font-size:.89rem; line-height:1.65; }}
        [data-testid="stMetric"] {{ padding:14px 15px; border-radius:14px; }}
        .gold-rule {{ margin:14px 0 18px; }}
    }}
    @media (prefers-reduced-motion:reduce) {{
        * {{ transition:none !important; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)



# -- Required project files ---------------------------------------------------
MODEL_PATH = first_existing([
    PROJECT_ROOT / "artifacts" / "final_model.joblib",
    PROJECT_ROOT / "artifacts" / "final_model.pkl",
])
DATASET_PATH = first_existing([
    PROJECT_ROOT / "data" / "processed" / "vehicles_cleaned.csv",
    PROJECT_ROOT / "data" / "vehicles_cleaned.csv",
    PROJECT_ROOT / "data" / "vehicles_cleaned.csv",
])
COMPARISON_PATH = first_existing([
    PROJECT_ROOT / "artifacts" / "model_comparison.csv",
    PROJECT_ROOT / "data" / "model_comparison.csv",
    PROJECT_ROOT / "data" / "model_comparison.csv",
])

missing = []
if MODEL_PATH is None:
    missing.append("artifacts/final_model.joblib")
if DATASET_PATH is None:
    missing.append("data/processed/vehicles_cleaned.csv or data/vehicles_cleaned.csv")
if COMPARISON_PATH is None:
    missing.append("model_comparison.csv (artifacts/, data/, or data/)")

if missing:
    st.error("CarWise could not load one or more required project files.")
    for item in missing:
        st.markdown(f"- `{item}`")
    st.caption(f"Application file: {APP_DIR}\n\nDetected project root: {PROJECT_ROOT}")
    st.info("Place app/main.py inside the project root's app/ folder, beside artifacts/ and the dataset folder.")
    st.stop()


# -- Cached original artifacts ------------------------------------------------
NUMERIC_COLUMNS = {"year", "price", "mileage", "tax", "mpg", "engineSize"}
TEXT_COLUMNS = ["manufacturer", "model", "transmission", "fuelType"]
REQUIRED_DATA_COLUMNS = {"manufacturer", "model", "year", "price", "mileage", "transmission", "engineSize"}


@st.cache_resource(show_spinner="Loading the supplied trained model…")
def load_model(path_string: str):
    # The model is the original saved Pipeline. No retraining or replacement.
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        return joblib.load(path_string)


@st.cache_data(show_spinner="Loading vehicle dataset…")
def load_dataset(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string, low_memory=False)


@st.cache_data(show_spinner="Loading model evaluation results…")
def load_comparison(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string)


@st.cache_data(show_spinner="Preparing analytics view…")
def prepare_dataset(path_string: str) -> pd.DataFrame:
    """Cleaned working view for analytics (original columns preserved)."""
    work = load_dataset(path_string).copy()
    for col in work.columns:
        if col in NUMERIC_COLUMNS:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in TEXT_COLUMNS:
        if col in work.columns:
            work[col] = work[col].astype("string").str.strip()
    needed = [c for c in REQUIRED_DATA_COLUMNS if c in work.columns]
    work = work.dropna(subset=needed)
    work = work[work["price"] > 0].copy()
    work["year"] = work["year"].astype(int)
    for col in TEXT_COLUMNS:
        if col in work.columns:
            work[col] = work[col].astype(str)
    return work


@st.cache_data(show_spinner=False)
def build_catalogue(path_string: str) -> dict[str, list[str]]:
    work = prepare_dataset(path_string)
    return {
        make: sorted(models.unique().tolist(), key=str.casefold)
        for make, models in work.groupby("manufacturer")["model"]
    }


try:
    pipeline = load_model(str(MODEL_PATH))
    raw_df = load_dataset(str(DATASET_PATH))
    comparison_raw = load_comparison(str(COMPARISON_PATH))
except Exception as exc:
    st.error(f"A project artifact could not be loaded: {exc}")
    st.caption("Check the installed package versions against the project's existing requirements.txt.")
    st.stop()

TARGET = "price"
missing_columns = sorted(REQUIRED_DATA_COLUMNS - set(raw_df.columns))
if missing_columns:
    st.error(f"The supplied dataset is missing required columns: {', '.join(missing_columns)}")
    st.stop()

df = prepare_dataset(str(DATASET_PATH))
if df.empty:
    st.error("The dataset contains no usable rows after cleaning.")
    st.stop()
CATALOGUE = build_catalogue(str(DATASET_PATH))

# The saved pipeline defines the real prediction contract. Never infer or
# silently substitute a different set/order of features.
if hasattr(pipeline, "feature_names_in_"):
    MODEL_FEATURES = list(pipeline.feature_names_in_)
else:
    MODEL_FEATURES = []
    try:
        pre = pipeline.named_steps.get("preprocessor")
        for _, _, cols in pre.transformers_:
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                MODEL_FEATURES.extend(list(cols))
    except Exception:
        pass

if not MODEL_FEATURES:
    st.error("Could not determine the saved model's expected input features. Prediction is disabled to avoid incorrect inputs.")
    st.stop()

missing_model_features = [c for c in MODEL_FEATURES if c not in df.columns]
if missing_model_features:
    st.error(f"Dataset is missing features required by the trained model: {missing_model_features}")
    st.stop()

MANUFACTURERS = sorted(CATALOGUE.keys(), key=str.casefold)


# -- Helpers ------------------------------------------------------------------
def money(value: float | int) -> str:
    return f"£{float(value):,.0f}"


def pretty(value: str) -> str:
    return str(value).replace("_", " ").strip().title()


def display_manufacturer(value: str) -> str:
    known = {"mercedes": "Mercedes-Benz", "skoda": "Škoda", "bmw": "BMW", "vw": "Volkswagen"}
    return known.get(str(value).casefold(), pretty(str(value)))


def manufacturer_models(manufacturer: str) -> list[str]:
    return CATALOGUE.get(manufacturer, [])


def local_subset(manufacturer: str, model: str | None = None) -> pd.DataFrame:
    subset = df[df["manufacturer"] == manufacturer]
    if model:
        subset = subset[subset["model"] == str(model)]
    return subset


def model_input_from_values(values: dict) -> pd.DataFrame:
    missing_values = [feature for feature in MODEL_FEATURES if feature not in values]
    if missing_values:
        raise ValueError(f"Missing model inputs: {', '.join(missing_values)}")
    row = {feature: values[feature] for feature in MODEL_FEATURES}
    return pd.DataFrame([row], columns=MODEL_FEATURES)


def predict_price(values: dict) -> float:
    input_frame = model_input_from_values(values)
    result = np.asarray(pipeline.predict(input_frame)).reshape(-1)
    if result.size != 1 or not np.isfinite(result[0]):
        raise ValueError("The trained model returned an invalid prediction.")
    return max(0.0, float(result[0]))


def label_for_feature(feature: str) -> str:
    labels = {
        "model": "Car model", "manufacturer": "Manufacturer", "year": "Registration year",
        "transmission": "Transmission", "mileage": "Mileage (miles)", "fuelType": "Fuel type",
        "tax": "Road tax (£)", "mpg": "Fuel economy (MPG)", "engineSize": "Engine size (L)",
    }
    return labels.get(feature, pretty(feature))


def feature_widget(feature: str, manufacturer: str, selected_model: str, key_prefix: str):
    """Create an input from real dataset values and the model's exact feature schema.

    Widget keys include the manufacturer and model, so defaults refresh whenever
    the vehicle selection changes (otherwise Streamlit keeps stale values).
    """
    if feature == "manufacturer":
        return manufacturer
    if feature == "model":
        return selected_model

    widget_key = f"{key_prefix}_{manufacturer}_{selected_model}_{feature}"
    series = df[feature].dropna() if feature in df.columns else pd.Series(dtype=object)
    if series.empty:
        raise ValueError(f"No dataset values are available for model feature '{feature}'.")

    subset = local_subset(manufacturer, selected_model)
    if subset.empty:
        subset = local_subset(manufacturer)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            raise ValueError(f"No numeric values are available for '{feature}'.")
        low, high = float(numeric.min()), float(numeric.max())
        local = pd.to_numeric(subset[feature], errors="coerce").dropna() if feature in subset.columns else pd.Series(dtype=float)
        default = float(local.median()) if not local.empty else float(numeric.median())
        default = float(np.clip(default, low, high))

        if feature in {"year", "mileage"}:
            return int(st.number_input(
                label_for_feature(feature),
                min_value=int(np.floor(low)), max_value=int(np.ceil(high)),
                value=int(np.clip(round(default), np.floor(low), np.ceil(high))),
                step=1 if feature == "year" else 1000, key=widget_key,
            ))
        step = {"engineSize": 0.1, "mpg": 0.5, "tax": 5.0}.get(feature, 1.0)
        fmt = "%.1f" if feature == "engineSize" else "%.2f"
        return float(st.number_input(
            label_for_feature(feature), min_value=low, max_value=high,
            value=default, step=step, format=fmt, key=widget_key,
        ))

    options = sorted(series.astype(str).unique().tolist(), key=str.casefold)
    if feature in subset.columns:
        local_options = sorted(subset[feature].dropna().astype(str).unique().tolist(), key=str.casefold)
        if local_options:
            options = local_options
    return st.selectbox(label_for_feature(feature), options, key=widget_key)


def hero(title: str, copy: str, eyebrow: str = "CARWISE / INTELLIGENT VEHICLE VALUATION"):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><div class="hero-title">{title}</div><p class="hero-copy">{copy}</p></div>',
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str | None = None):
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f'<div class="subtle">{subtitle}</div>', unsafe_allow_html=True)


def rule():
    st.markdown("<div class='gold-rule'></div>", unsafe_allow_html=True)


def plotly_dark(fig, height: int | None = None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=TEXT),
        colorway=[GOLD, "#8CB9FF", "#7BC8B2", "#B5A4E8", "#E2A66F", "#91A7C4"],
        margin=dict(l=12, r=12, t=48, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PANEL_ALT, font_color=TEXT, bordercolor=GOLD),
    )
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(gridcolor="rgba(150,175,210,.12)", zerolinecolor="rgba(150,175,210,.14)")
    fig.update_yaxes(gridcolor="rgba(150,175,210,.12)", zerolinecolor="rgba(150,175,210,.14)")
    if height:
        fig.update_layout(height=height)
    return fig


def gbp(fig, axis: str = "y"):
    """Format a price axis as £ with thousands separators."""
    getattr(fig, f"update_{axis}axes")(tickprefix="£", tickformat=",")
    return fig


def show_chart(fig, height: int | None = None):
    st.plotly_chart(plotly_dark(fig, height), **STRETCH)


def show_table(data: pd.DataFrame, hide_index: bool = True):
    st.dataframe(data, hide_index=hide_index, **STRETCH)


def show_logo():
    if LOGO_BYTES is not None:
        st.image(LOGO_BYTES, width=260)
    else:
        st.markdown(
            f"""
            <div style="
                font-family:Plus Jakarta Sans,sans-serif;
                font-weight:800;
                font-size:2.4rem;
                color:{TEXT};
                letter-spacing:-.07em;
                padding:15px 0;
            ">
                CAR<span style="color:{GOLD}">WISE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )




# -- Navigation ---------------------------------------------------------------
PAGES = ["Dashboard", "Price Prediction", "Market Analysis", "Car Comparison", "Model Insights"]

with st.sidebar:
    show_logo()
    st.markdown(
        "<div style='color:#A6AFBC;font-size:.67rem;font-weight:700;letter-spacing:.16em;margin-top:10px'>SMARTER AUTOMOTIVE DECISIONS</div>",
        unsafe_allow_html=True,
    )
    rule()
    st.markdown("<div class='nav-label'>DATA OVERVIEW</div>", unsafe_allow_html=True)
    page = st.radio("Workspace", PAGES, label_visibility="collapsed")
    rule()
    st.markdown("<div class='nav-label'>LIVE PROJECT DATA</div>", unsafe_allow_html=True)
    st.metric("Vehicles in dataset", f"{len(df):,}")
    st.metric("Manufacturers", f"{df['manufacturer'].nunique():,}")
    st.caption(f"Model features: {len(MODEL_FEATURES)}")
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# =============================================================================
# A. DASHBOARD
# =============================================================================
if page == "Dashboard":
    hero(
        "Intelligence behind every price.",
        "Smarter valuations. Deeper market insights.",
        "CARWISE / INTELLIGENT VEHICLE VALUATION"
    )
    total = len(df)
    avg_price = float(df[TARGET].mean())
    median_price = float(df[TARGET].median())
    min_price = float(df[TARGET].min())
    max_price = float(df[TARGET].max())
    avg_mileage = float(df["mileage"].mean())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Vehicles", f"{total:,}")
    m2.metric("Manufacturers", f"{df['manufacturer'].nunique():,}")
    m3.metric("Average Market Price", money(avg_price))
    m4.metric("Lowest Price", money(min_price))
    m5.metric("Highest Price", money(max_price))

    rule()
    left, right = st.columns([1.35, 1])
    with left:
        section_title("Vehicle Price Distribution", "How vehicle prices are distributed across the dataset.")
        fig = px.histogram(df, x="price", nbins=55, marginal="box", labels={"price": "Vehicle price (£)"}, color_discrete_sequence=[GOLD])
        fig.update_traces(opacity=.88)
        show_chart(gbp(fig, "x"), 370)
    with right:
        section_title("Manufacturer Overview", "Vehicle inventory count, ranked by manufacturer.")
        counts = df["manufacturer"].value_counts().head(12).rename_axis("Manufacturer").reset_index(name="Vehicles")
        counts["Manufacturer"] = counts["Manufacturer"].map(display_manufacturer)
        fig = px.bar(counts.sort_values("Vehicles"), x="Vehicles", y="Manufacturer", orientation="h", color="Vehicles", color_continuous_scale=GOLD_SCALE)
        show_chart(fig, 370)

    c1, c2 = st.columns(2)
    with c1:
        section_title("Pricing by Fuel Type")
        if "fuelType" in df.columns:
            fuel = df.groupby("fuelType", dropna=True)["price"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(fuel, x="fuelType", y="price", color="price", color_continuous_scale=GOLD_SCALE, labels={"fuelType": "Fuel type", "price": "Average price (£)"})
            show_chart(gbp(fig), 320)
        else:
            st.info("Fuel type is not present in this dataset.")
    with c2:
        section_title("Price Trends Over Time")
        yearly = df.groupby("year", as_index=False)["price"].median()
        fig = px.line(yearly, x="year", y="price", markers=True, labels={"year": "Registration year", "price": "Median price (£)"})
        fig.update_traces(line=dict(color=GOLD, width=3), marker=dict(size=6))
        show_chart(gbp(fig), 320)

    rule()
    q1, q2, q3 = st.columns(3)
    q1.metric("Median vehicle price", money(median_price))
    q2.metric("Average mileage", f"{avg_mileage:,.0f} mi")
    q3.metric("Top Manufacturer by Listings", display_manufacturer(df["manufacturer"].value_counts().index[0]))


# =============================================================================
# B. PRICE PREDICTION
# =============================================================================
elif page == "Price Prediction":
    hero("A clearer picture of your car's value.", "Enter your vehicle details to receive an AI-powered price estimate.", "AI VALUATION / PREDICTION ENGINE")

    if not MANUFACTURERS:
        st.error("No manufacturers are available in the dataset.")
        st.stop()

    # Manufacturer and model live outside the form so changing them reruns the
    # app and refreshes the real model catalogue and field defaults immediately.
    top = st.columns([1, 2])
    with top[0]:
        manufacturer = st.selectbox("Manufacturer", MANUFACTURERS, format_func=display_manufacturer, key="valuation_manufacturer")
    available_models = manufacturer_models(manufacturer)
    if not available_models:
        st.warning("No models were found for this manufacturer in the supplied dataset.")
        st.stop()
    with top[1]:
        selected_model = st.selectbox("Vehicle model", available_models, key=f"valuation_model_{manufacturer}")

    rule()
    st.markdown("#### Vehicle specification")
    st.caption("Only fields required by the loaded model are requested. Options and defaults are derived from the real dataset for the selected model.")

    input_values = {"manufacturer": manufacturer, "model": selected_model}
    other_features = [f for f in MODEL_FEATURES if f not in {"manufacturer", "model"}]

    submitted = False
    with st.form("carwise_prediction_form", border=True):
        cols = st.columns(2)
        try:
            for i, feature in enumerate(other_features):
                with cols[i % 2]:
                    input_values[feature] = feature_widget(feature, manufacturer, selected_model, "prediction")
            submitted = st.form_submit_button("Get Vehicle Valuation", **STRETCH)
        except Exception as exc:
            st.error(f"Could not build a valid input for the trained model: {exc}")

    if submitted:
        try:
            estimate = predict_price(input_values)

            # Keep only the latest prediction for price comparison
            st.session_state["carwise_last_estimate"] = estimate
            st.session_state["carwise_last_vehicle"] = (
                f"{display_manufacturer(manufacturer)} {selected_model}"
            )
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **{feature: input_values.get(feature) for feature in MODEL_FEATURES},
                "predicted_price": estimate,
            }


            st.success("Your vehicle valuation is ready.")
            st.markdown(
                f"<div class='hero'><div class='eyebrow'>ESTIMATED MARKET VALUE</div><div class='price-output'>{money(estimate)}</div><p class='hero-copy'>Model-generated estimate · GBP</p></div>",
                unsafe_allow_html=True,
            )

            # Market context from real dataset listings (does not alter the prediction).
            comparable = local_subset(manufacturer, selected_model)
            scope = f"{display_manufacturer(manufacturer)} {selected_model}"
            if "year" in input_values and len(comparable) > 0:
                near = comparable[(comparable["year"] - int(input_values["year"])).abs() <= 1]
                if len(near) >= 10:
                    comparable, scope = near, f"{scope}, {int(input_values['year']) - 1}–{int(input_values['year']) + 1}"
            if len(comparable) >= 3:
                st.markdown("#### Market context")
                st.caption(f"Compared against {len(comparable):,} real dataset listings: {scope}.")
                median_listing = float(comparable["price"].median())
                q25, q75 = float(comparable["price"].quantile(.25)), float(comparable["price"].quantile(.75))
                diff_pct = (estimate - median_listing) / median_listing * 100 if median_listing else 0.0
                x1, x2, x3, x4 = st.columns(4)
                x1.metric("Comparable listings", f"{len(comparable):,}")
                x2.metric("Median listing", money(median_listing))
                x3.metric("Typical range", f"{money(q25)} – {money(q75)}")
                x4.metric("Valuation vs median", f"{diff_pct:+.1f}%")
                fig = px.histogram(comparable, x="price", nbins=30, color_discrete_sequence=["#7795B8"], labels={"price": "Listing price (£)"})
                fig.update_traces(opacity=.8)
                fig.add_vline(x=estimate, line_color=GOLD_LIGHT, line_dash="dash", line_width=2, annotation_text="Your estimate", annotation_font_color=GOLD_LIGHT)
                show_chart(gbp(fig, "x"), 300)
            else:
                st.caption("Too few comparable listings in the dataset to build a market-context view for this selection.")

            details = pd.DataFrame([record]).drop(columns=["timestamp", "predicted_price"])
            details = details.rename(columns={c: label_for_feature(c) for c in details.columns}).astype(str)
            st.markdown("#### Valuation details")
            show_table(details)
            st.caption("This is a model estimate based on the supplied training data, not a guaranteed sale or dealer offer.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    # =============================================================================
    # ASKING PRICE ANALYSIS
    # =============================================================================
    if "carwise_last_estimate" in st.session_state:
        estimate_for_comparison = float(
            st.session_state["carwise_last_estimate"]
        )
        vehicle_for_comparison = st.session_state.get(
            "carwise_last_vehicle", "Selected vehicle"
        )

        rule()

        st.markdown("#### Asking Price Analysis")
        st.caption(
            f"Enter the asking price you found for {vehicle_for_comparison}. "
            "CarWise will evaluate it against the estimated market value."
        )

        with st.form("carwise_asking_price_form", border=True):
            asking_price = st.number_input(
                "Asking Price (£)",
                min_value=1.0,
                value=float(max(1, round(estimate_for_comparison))),
                step=100.0,
                format="%.2f",
            )

            compare_submitted = st.form_submit_button(
                "Analyze This Price",
                **STRETCH,
            )

        if compare_submitted:
            difference = asking_price - estimate_for_comparison
            difference_pct = (
                difference / estimate_for_comparison * 100
                if estimate_for_comparison > 0
                else 0.0
            )

            st.markdown("#### Pricing Assessment")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "CarWise Estimate",
                money(estimate_for_comparison),
            )

            col2.metric(
                "Asking Price",
                money(asking_price),
            )

            col3.metric(
                "Price Difference",
                money(difference),
                f"{difference_pct:+.1f}%",
                delta_color="inverse",
            )

            if difference_pct < -15:
                st.success("Great Deal")
                st.write("Check the car before buying.")

            elif difference_pct < -5:
                st.success("Good Deal")
                st.write(
                    "Good price. Consider buying after inspection."
                )

            elif difference_pct <= 5:
                st.info("Fair Price")
                st.write("The price looks reasonable.")

            elif difference_pct <= 10:
                st.warning("Slightly Expensive")
                st.write("Try negotiating a lower price.")

            elif difference_pct <= 20:
                st.warning("Overpriced")
                st.write(
                    "Negotiate or compare other cars."
                )

            else:
                st.error("Too Expensive")
                st.write(
                    "Look for a better-priced car."
                )

            st.caption(
                "This recommendation is based on the model estimate only. "
                "Always inspect the vehicle's condition and history, "
                "and compare similar listings before making a decision."
            )


# =============================================================================
# C. MARKET ANALYSIS
# =============================================================================
elif page == "Market Analysis":
    hero("Explore the Automotive Market.", "Discover pricing patterns, compare manufacturers, and explore real vehicle listings.", "MARKET INSIGHTS / DATA ANALYTICS")

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    price_min, price_max = int(df["price"].min()), int(df["price"].max())
    fuel_options = sorted(df["fuelType"].dropna().astype(str).unique()) if "fuelType" in df.columns else []

    filter_cols = st.columns(4)
    with filter_cols[0]:
        selected_manufacturers = st.multiselect("Manufacturers", MANUFACTURERS, default=[], format_func=display_manufacturer, placeholder="All manufacturers")
    with filter_cols[1]:
        if year_min < year_max:
            year_range = st.slider("Registration year", year_min, year_max, (year_min, year_max))
        else:
            year_range = (year_min, year_max)
            st.caption(f"Registration year: {year_min}")
    with filter_cols[2]:
        if price_min < price_max:
            price_range = st.slider("Price range (£)", price_min, price_max, (price_min, price_max), step=max(1, (price_max - price_min) // 200))
        else:
            price_range = (price_min, price_max)
            st.caption(f"Price: {money(price_min)}")
    with filter_cols[3]:
        selected_fuels = st.multiselect("Fuel type", fuel_options, default=[], placeholder="All fuel types")

    filtered = df[df["year"].between(*year_range) & df["price"].between(*price_range)]
    if selected_manufacturers:
        filtered = filtered[filtered["manufacturer"].isin(selected_manufacturers)]
    if selected_fuels and "fuelType" in filtered.columns:
        filtered = filtered[filtered["fuelType"].astype(str).isin(selected_fuels)]

    if filtered.empty:
        st.warning("No vehicles match the current filters. Widen the selected ranges to see results.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Vehicles in view", f"{len(filtered):,}")
        k2.metric("Average Market Price", money(filtered["price"].mean()))
        k3.metric("Median mileage", f"{filtered['mileage'].median():,.0f} mi")
        k4.metric("Manufacturers", f"{filtered['manufacturer'].nunique():,}")

        a, b = st.columns(2)
        with a:
            section_title("Price distribution")
            fig = px.histogram(filtered, x="price", nbins=45, color_discrete_sequence=[GOLD], labels={"price": "Price (£)"})
            show_chart(gbp(fig, "x"), 340)
        with b:
            section_title("Average price by manufacturer")
            by_make = filtered.groupby("manufacturer", as_index=False)["price"].mean().sort_values("price", ascending=False)
            by_make["manufacturer"] = by_make["manufacturer"].map(display_manufacturer)
            fig = px.bar(by_make, x="manufacturer", y="price", color="price", color_continuous_scale=GOLD_SCALE, labels={"manufacturer": "Manufacturer", "price": "Average price (£)"})
            show_chart(gbp(fig), 340)

        scatter_data = filtered.sample(4500, random_state=42) if len(filtered) > 4500 else filtered
        c, d = st.columns(2)
        with c:
            section_title("Price vs. mileage", "Sampled to 4,500 points for speed when many vehicles match." if len(filtered) > 4500 else None)
            color_col = "fuelType" if "fuelType" in scatter_data.columns else "manufacturer"
            fig = px.scatter(scatter_data, x="mileage", y="price", color=color_col, opacity=.58, hover_data=["manufacturer", "model", "year"], labels={"mileage": "Mileage (miles)", "price": "Price (£)"})
            show_chart(gbp(fig), 350)
        with d:
            section_title("Price vs. registration year")
            fig = px.scatter(scatter_data, x="year", y="price", color="transmission", opacity=.55, hover_data=["manufacturer", "model", "mileage"], labels={"year": "Registration year", "price": "Price (£)"})
            show_chart(gbp(fig), 350)

        e, f = st.columns(2)
        pie_colors = [GOLD, "#8CB9FF", "#7BC8B2", "#B5A4E8", "#E2A66F"]
        with e:
            section_title("Transmission mix")
            trans = filtered["transmission"].value_counts().rename_axis("Transmission").reset_index(name="Vehicles")
            fig = px.pie(trans, names="Transmission", values="Vehicles", hole=.62, color_discrete_sequence=pie_colors)
            show_chart(fig, 310)
        with f:
            section_title("Fuel type mix")
            if "fuelType" in filtered.columns:
                fuels = filtered["fuelType"].value_counts().rename_axis("Fuel type").reset_index(name="Vehicles")
                fig = px.pie(fuels, names="Fuel type", values="Vehicles", hole=.62, color_discrete_sequence=pie_colors)
                show_chart(fig, 310)
            else:
                st.info("Fuel type is unavailable in this dataset.")

        with st.expander("Inspect filtered source records"):
            show_table(filtered.head(5000))
            if len(filtered) > 5000:
                st.caption(f"Showing the first 5,000 of {len(filtered):,} matching records. Download the full filtered set below.")
            st.download_button("Download filtered records (CSV)", data=filtered.to_csv(index=False).encode("utf-8-sig"), file_name="carwise_filtered_records.csv", mime="text/csv")


# =============================================================================
# D. CAR COMPARISON
# =============================================================================
elif page == "Car Comparison":
    hero("Find the Right Car for You.", "Compare vehicle specifications, mileage, and asking prices side by side.", "VEHICLE COMPARISON / SIDE-BY-SIDE")

    st.caption("Choose a manufacturer, model and exact dataset record for each vehicle. Additional comparison slots are optional.")
    extra_cols = st.columns(2)
    with extra_cols[0]:
        include_3 = st.checkbox("Include a third vehicle", value=False, key="compare_include_3")
    with extra_cols[1]:
        include_4 = st.checkbox("Include a fourth vehicle", value=False, key="compare_include_4")
    active_slots = [True, True, include_3, include_4]

    selected_records = []
    slot_cols = st.columns(2)
    for slot in range(4):
        if not active_slots[slot]:
            continue
        with slot_cols[slot % 2]:
            with st.container(border=True):
                st.markdown(f"**Vehicle {slot + 1}**")
                make = st.selectbox("Manufacturer", MANUFACTURERS, key=f"compare_make_{slot + 1}", format_func=display_manufacturer)
                model_options = manufacturer_models(make)
                model_choice = st.selectbox("Model", model_options, key=f"compare_model_{slot + 1}_{make}") if model_options else None
                if model_choice is not None:
                    candidates = local_subset(make, model_choice).sort_values(["year", "price"], ascending=[False, True])
                    candidates = candidates.reset_index(names="source_row")
                    candidates["record_label"] = [
                        f"{int(r.year)} · {money(r.price)} · {int(r.mileage):,} mi · row {int(r.source_row)}"
                        for r in candidates.itertuples()
                    ]
                    chosen_label = st.selectbox("Vehicle Listing", candidates["record_label"].tolist(), key=f"compare_record_{slot + 1}_{make}_{model_choice}")
                    selected_records.append(candidates.loc[candidates["record_label"] == chosen_label].iloc[0])

    if len(selected_records) < 2:
        st.info("Select at least two vehicles to build a comparison.")
    else:
        selected = pd.DataFrame(selected_records).reset_index(drop=True)
        if selected["source_row"].duplicated().any():
            st.info("The same dataset record is selected more than once. Pick different records for a more useful comparison.")
        spec_cols = [c for c in ["manufacturer", "model", "year", "price", "mileage", "transmission", "fuelType", "engineSize", "tax", "mpg"] if c in selected.columns]

        # Keep comparison labels unique: the same model/year can be selected more than once.
        base_labels = [f"{display_manufacturer(r['manufacturer'])} {r['model']} ({int(r['year'])})" for _, r in selected.iterrows()]
        label_counts = {label: base_labels.count(label) for label in set(base_labels)}
        vehicle_labels = [f"{label} — Vehicle {i + 1}" if label_counts[label] > 1 else label for i, label in enumerate(base_labels)]

        rule()
        best = st.columns(3)
        best[0].metric("Lowest Price", money(selected["price"].min()), vehicle_labels[int(selected["price"].idxmin())], delta_color="off")
        best[1].metric("Lowest mileage", f"{selected['mileage'].min():,.0f} mi", vehicle_labels[int(selected["mileage"].idxmin())], delta_color="off")
        best[2].metric("Newest registration", str(int(selected["year"].max())), vehicle_labels[int(selected["year"].idxmax())], delta_color="off")

        def fmt_cell(col: str, value) -> str:
            if pd.isna(value):
                return "—"
            if col == "price" or col == "tax":
                return money(value)
            if col == "mileage":
                return f"{float(value):,.0f} mi"
            if col == "year":
                return str(int(value))
            if col == "engineSize":
                return f"{float(value):.1f} L"
            if col == "mpg":
                return f"{float(value):.1f}"
            if col == "manufacturer":
                return display_manufacturer(value)
            return str(value)

        specs = pd.DataFrame({label: [fmt_cell(c, selected.loc[i, c]) for c in spec_cols] for i, label in enumerate(vehicle_labels)}, index=[label_for_feature(c) if c in MODEL_FEATURES else pretty(c) for c in spec_cols])
        st.markdown("#### Specification comparison")
        show_table(specs, hide_index=False)

        x, y = st.columns(2)
        with x:
            section_title("Price Comparison")
            price_chart = pd.DataFrame({"Vehicle": vehicle_labels, "Price (£)": selected["price"].astype(float)})
            fig = px.bar(price_chart, x="Vehicle", y="Price (£)", color="Price (£)", color_continuous_scale=GOLD_SCALE, text_auto=",.0f")
            fig.update_layout(showlegend=False)
            show_chart(gbp(fig), 350)
        with y:
            section_title("Mileage Comparison")
            mileage_chart = pd.DataFrame({"Vehicle": vehicle_labels, "Mileage (miles)": selected["mileage"].astype(float)})
            fig = px.bar(mileage_chart, x="Vehicle", y="Mileage (miles)", color_discrete_sequence=[GOLD], text_auto=",.0f")
            show_chart(fig, 350)

        st.caption("All compared specifications and prices are read directly from the selected dataset rows.")


# =============================================================================
# E. MODEL INSIGHTS
# =============================================================================
elif page == "Model Insights":
    hero("Inside the Intelligence.", "Explore model performance, evaluation metrics, and the factors influencing predictions.", "MODEL PERFORMANCE / EVALUATION")

    comparison = comparison_raw.copy()
    if "Unnamed: 0" in comparison.columns:
        comparison = comparison.rename(columns={"Unnamed: 0": "Model"})
    elif "Model" not in comparison.columns:
        comparison = comparison.rename(columns={comparison.columns[0]: "Model"})

    metric_aliases = {"Median AE": "Median Absolute Error", "Explained Variance": "Explained Variance", "R2": "R²"}
    for metric in [c for c in comparison.columns if c != "Model"]:
        comparison[metric] = pd.to_numeric(comparison[metric], errors="coerce")

    if "R2" in comparison.columns and comparison["R2"].notna().any():
        best_r2 = comparison.loc[comparison["R2"].idxmax()]
        metric_cards = st.columns(4)
        metric_cards[0].metric("Top R² model", str(best_r2["Model"]))
        metric_cards[1].metric("R²", f"{best_r2['R2']:.4f}")
        if "MAE" in comparison.columns:
            metric_cards[2].metric("MAE", money(best_r2["MAE"]))
        if "RMSE" in comparison.columns:
            metric_cards[3].metric("RMSE", money(best_r2["RMSE"]))

    rule()
    available_plot_metrics = [m for m in ["R2", "MAE", "RMSE", "Median AE", "MAPE", "Explained Variance"] if m in comparison.columns]
    if available_plot_metrics:
        selected_metric = st.selectbox("Evaluation metric", available_plot_metrics, format_func=lambda x: metric_aliases.get(x, x))
        higher_is_better = selected_metric in {"R2", "Explained Variance"}
        ordered = comparison.sort_values(selected_metric, ascending=not higher_is_better)
        st.caption("Higher is better for this metric." if higher_is_better else "Lower is better for this metric.")
        fig = px.bar(ordered, x="Model", y=selected_metric, color=selected_metric, color_continuous_scale=GOLD_SCALE, labels={selected_metric: metric_aliases.get(selected_metric, selected_metric)})
        fig.update_layout(xaxis_tickangle=-35, showlegend=False)
        show_chart(fig, 430)

    st.markdown("#### Complete evaluation results")
    show_table(comparison)
    st.caption("Metric values are read from the supplied model_comparison.csv. They are not regenerated or altered.")

    # Genuine feature importances, only when the saved estimator exposes them.
    rule()
    section_title("Feature Importance", "Available only when the loaded estimator exposes fitted feature_importances_.")

    def source_feature(transformed_name: str) -> str:
        """Map a transformed column (e.g. 'cat__model_A3') back to its source feature."""
        base = str(transformed_name).split("__", 1)[-1]
        for feat in sorted(MODEL_FEATURES, key=len, reverse=True):
            if base == feat or base.startswith(f"{feat}_"):
                return feat
        return base

    try:
        estimator = pipeline.steps[-1][1] if hasattr(pipeline, "steps") and pipeline.steps else pipeline
        # Unwrap target-transforming wrappers (e.g. log-price TransformedTargetRegressor).
        for _ in range(3):
            if getattr(estimator, "feature_importances_", None) is None and hasattr(estimator, "regressor_"):
                estimator = estimator.regressor_
        importances = getattr(estimator, "feature_importances_", None)

        transformed_names = None
        try:
            if hasattr(pipeline, "steps") and len(pipeline.steps) > 1:
                transformed_names = pipeline[:-1].get_feature_names_out()
        except Exception:
            transformed_names = None

        if importances is None:
            st.info("The loaded estimator does not expose feature_importances_. No feature-importance chart is available for this model.")
        else:
            importances = np.asarray(importances, dtype=float).reshape(-1)
            if transformed_names is not None and len(transformed_names) == len(importances):
                grouped: dict[str, float] = {}
                for name, value in zip(transformed_names, importances):
                    key = source_feature(name)
                    grouped[key] = grouped.get(key, 0.0) + float(value)
                importance_df = pd.DataFrame({"Feature": [label_for_feature(k) for k in grouped], "Importance": list(grouped.values())})
            else:
                importance_df = pd.DataFrame({"Feature": [f"Transformed feature {i + 1}" for i in range(len(importances))], "Importance": importances})
            importance_df = importance_df.sort_values("Importance", ascending=False)
            fig = px.bar(importance_df.head(15).sort_values("Importance"), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=GOLD_SCALE)
            show_chart(fig, 390)
    except Exception as exc:
        st.info(f"Feature importance could not be extracted from this pipeline: {exc}")

    with st.expander("Loaded model contract"):
        st.write("Pipeline type:", type(pipeline).__name__)
        st.write("Expected input features (exact saved order):", MODEL_FEATURES)
        if hasattr(pipeline, "steps"):
            st.write("Pipeline stages:", [name for name, _ in pipeline.steps])