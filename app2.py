from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
HISTORICAL_PATH = DATA_DIR / "f1_ml_training.csv"
LATEST_PATH = DATA_DIR / "F1_2025_2026_Cleaned_Dataset.csv"
TEAM_FILE_EXCLUSIONS = {HISTORICAL_PATH.name, LATEST_PATH.name}

FEATURES = [
    "total_points",
    "races_started",
    "average_grid",
    "average_finish",
    "podiums",
    "wins",
    "best_finish",
    "championship_position",
    "championship_points",
    "championship_wins",
]
TEAM_FEATURES = ["Points", "Wins", "Podiums", "AvgFinish"]

RED = "#E10600"
SLATE = "#111318"
TEXT = "#F4F4F5"
MUTED = "#8F98A8"
GREEN = "#62E6A8"
AMBER = "#FFC857"

st.set_page_config(
    page_title="Formula One Championship Predictor",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');
:root { --red: #E10600; --bg: #0b0d10; --panel: #15191f; --panel2: #1c222b; --line: #303744; --text: #f4f4f5; --muted: #8f98a8; }
.stApp { background: radial-gradient(circle at 20% -10%, rgba(225,6,0,.13), transparent 30%), linear-gradient(135deg, #0b0d10 0%, #12161c 55%, #0b0d10 100%); color: var(--text); font-family: 'DM Sans', system-ui, sans-serif; }
.stApp:before { content: ''; position: fixed; inset: 0; pointer-events: none; opacity: .09; background-image: linear-gradient(45deg, #fff 1px, transparent 1px), linear-gradient(-45deg, #fff 1px, transparent 1px); background-size: 14px 14px; mask-image: linear-gradient(to bottom, black, transparent 74%); }
.block-container { max-width: 1420px; padding: 2.2rem 2rem 4rem; }
[data-testid="stSidebar"] { background: #0d1014; border-right: 1px solid #242a34; }
[data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stSlider { margin-bottom: .8rem; }
.stMarkdown, p, label, div { color: var(--text); }
.hero { position: relative; overflow: hidden; border: 1px solid #343a45; border-radius: 24px; padding: 2rem 2.2rem 1.7rem; background: linear-gradient(105deg, rgba(225,6,0,.22), rgba(25,29,36,.96) 44%, rgba(9,12,15,.96)); box-shadow: 0 20px 60px rgba(0,0,0,.35); margin-bottom: 1.2rem; }
.hero:after { content: ''; position: absolute; width: 58%; height: 4px; left: -6%; bottom: 18px; background: var(--red); transform: skewX(-28deg); box-shadow: 0 0 28px rgba(225,6,0,.75); animation: scan 1.8s ease-out .35s both; }
.hero-kicker { font-family: 'Barlow Condensed', sans-serif; letter-spacing: .22em; color: #ff6a63; font-size: .85rem; text-transform: uppercase; }
.hero h1 { margin: .45rem 0 .35rem; font-family: 'Barlow Condensed', sans-serif; letter-spacing: .03em; line-height: .93; font-size: clamp(2.35rem, 6vw, 5.7rem); font-weight: 800; color: white; max-width: 800px; }
.hero p { color: #c3cbd5; max-width: 680px; font-size: 1rem; margin: 0; }
.car-card { position: absolute; left: -250px; bottom: 18px; width: 230px; height: 88px; border: 1px solid rgba(255,255,255,.2); border-radius: 16px 16px 7px 7px; background: linear-gradient(160deg, #f21a13, #870600); transform: rotate(-4deg); animation: drive-across 6.5s cubic-bezier(.2,.7,.25,1) .15s infinite; box-shadow: 0 18px 35px rgba(0,0,0,.35), 0 0 34px rgba(225,6,0,.28); z-index: 2; }
.car-card:before { content: 'F1//PREDICTOR'; position: absolute; left: 18px; top: 12px; font: 700 12px 'Barlow Condensed', sans-serif; letter-spacing: .18em; color: rgba(255,255,255,.92); }
.car-card:after { content: ''; position: absolute; left: 30px; right: 28px; bottom: 18px; height: 16px; border-radius: 18px 28px 6px 6px; background: #101318; box-shadow: -22px 8px 0 -3px #0b0d10, 68px 8px 0 -3px #0b0d10, 35px -14px 0 -3px #f5f5f5; }
@keyframes drive-across { 0% { left: -250px; opacity: 0; } 10% { opacity: 1; } 82% { opacity: 1; } 100% { left: calc(100% + 40px); opacity: 0; } }
@keyframes scan { 0% { transform: translateX(-80%) skewX(-28deg); opacity: 0; } 100% { transform: translateX(0) skewX(-28deg); opacity: 1; } }
.section-kicker { color: #ff7069; font-family: 'Barlow Condensed', sans-serif; text-transform: uppercase; letter-spacing: .16em; font-size: .76rem; font-weight: 700; }
.section-title { font-family: 'Barlow Condensed', sans-serif; text-transform: uppercase; letter-spacing: .03em; font-size: 2rem; font-weight: 700; margin: .15rem 0 .5rem; }
.panel { background: rgba(21,25,31,.85); border: 1px solid #2d3440; border-radius: 18px; padding: 1.05rem 1.15rem; }
.metric-card { background: linear-gradient(145deg, #1b2028, #101318); border: 1px solid #303744; border-top: 3px solid var(--red); border-radius: 14px; padding: 1rem 1.05rem; min-height: 106px; }
.metric-label { color: var(--muted); text-transform: uppercase; font: 700 .72rem 'Barlow Condensed', sans-serif; letter-spacing: .14em; }
.metric-value { color: white; font: 700 2rem 'Barlow Condensed', sans-serif; margin-top: .18rem; }
.metric-note { color: #bcc4cf; font-size: .78rem; margin-top: .2rem; }
.stButton > button { background: var(--red); color: white; border: none; border-radius: 9px; font-weight: 700; }
.stButton > button:hover { background: #ff241b; color: white; }
.stTabs [data-baseweb="tab-list"] { gap: .25rem; background: transparent; border-bottom: 1px solid #303744; }
.stTabs [data-baseweb="tab"] { color: #9aa4b4; font-family: 'Barlow Condensed', sans-serif; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
.stTabs [aria-selected="true"] { color: white; border-bottom-color: var(--red); }
div[data-baseweb="select"] > div { background: #15191f; border-color: #3a424e; }
.stSlider [data-baseweb="slider"] { color: var(--red); }
[data-testid="stDataFrame"] { border: 1px solid #303744; border-radius: 12px; overflow: hidden; }
.podium-wrap { display: flex; align-items: flex-end; justify-content: center; gap: 10px; height: 220px; padding: 1.1rem 0 .4rem; }
.podium { width: min(30%, 170px); min-width: 92px; text-align: center; animation: lift .65s ease both; }
.podium:nth-child(1) { animation-delay: .15s; } .podium:nth-child(2) { animation-delay: .3s; } .podium:nth-child(3) { animation-delay: .45s; }
.podium .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 700; font-size: .83rem; padding: .55rem .25rem; }
.podium .bar { border-radius: 9px 9px 3px 3px; display: flex; align-items: center; justify-content: center; font: 800 2rem 'Barlow Condensed', sans-serif; color: #0b0d10; }
.podium.first .bar { height: 142px; background: linear-gradient(#f7d774,#af7c14); } .podium.second .bar { height: 103px; background: linear-gradient(#dce3e9,#7b8999); } .podium.third .bar { height: 75px; background: linear-gradient(#dc9e70,#8a4a25); }
@keyframes lift { from { transform: translateY(35px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
.fine-print { color: var(--muted); font-size: .78rem; line-height: 1.45; }
.alert-strip { border-left: 4px solid var(--red); background: rgba(225,6,0,.1); border-radius: 8px; padding: .75rem .9rem; color: #f2c4c2; }
@media (max-width: 800px) { .block-container { padding: 1rem .75rem 3rem; } .hero { padding: 1.35rem 1rem 1.1rem; min-height: 270px; } .car-card { width: 190px; height: 66px; bottom: 18px; } .hero h1 { font-size: 3rem; max-width: 90%; } .podium-wrap { height: 180px; } }
@media (prefers-reduced-motion: reduce) { *, *:before, *:after { animation: none !important; transition: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)


def _coalesce(frame: pd.DataFrame, columns: Iterable[str], default: float = 0.0) -> pd.Series:
    present = [c for c in columns if c in frame.columns]
    if not present:
        return pd.Series(default, index=frame.index, dtype="float64")
    result = frame[present[0]].copy()
    for column in present[1:]:
        result = result.fillna(frame[column])
    return pd.to_numeric(result, errors="coerce").fillna(default)


@st.cache_data(show_spinner=False)
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not HISTORICAL_PATH.exists() or not LATEST_PATH.exists():
        raise FileNotFoundError("Place both supplied CSV files in the data/ folder before launching the app.")
    return pd.read_csv(HISTORICAL_PATH), pd.read_csv(LATEST_PATH)


@st.cache_data(show_spinner=False)
def load_supplemental_csvs() -> list[pd.DataFrame]:
    frames = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        if path.name in TEAM_FILE_EXCLUSIONS:
            continue
        try:
            frame = pd.read_csv(path)
            frame.attrs["source_file"] = path.name
            frames.append(frame)
        except Exception:
            continue
    return frames


def canonicalize(historical: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    # Process historical dataset
    hist = pd.DataFrame(index=historical.index)
    f_name = historical["driver_first_name"].astype(str).str.strip() if "driver_first_name" in historical.columns else ""
    l_name = historical["driver_last_name"].astype(str).str.strip() if "driver_last_name" in historical.columns else ""
    hist["Driver"] = (f_name + " " + l_name).str.strip()
    if "driver" in historical.columns:
        hist["Driver"] = hist["Driver"].replace("", historical["driver"].astype(str).str.strip())
        
    hist["Season"] = pd.to_numeric(historical["year"] if "year" in historical.columns else historical.get("Season"), errors="coerce").astype("Int64")
    hist["driverId"] = historical.get("driverId", pd.Series(pd.NA, index=historical.index))
    hist["total_points"] = _coalesce(historical, ["total_points", "championship_points", "Points"])
    hist["races_started"] = _coalesce(historical, ["races_started", "races", "Races"])
    hist["average_grid"] = _coalesce(historical, ["average_grid", "average_qualifying", "AverageQualifying"], np.nan)
    hist["average_finish"] = _coalesce(historical, ["average_finish", "AverageFinish"], np.nan)
    hist["podiums"] = _coalesce(historical, ["podiums", "Podiums"])
    hist["wins"] = _coalesce(historical, ["wins", "Wins"])
    hist["best_finish"] = _coalesce(historical, ["best_finish", "BestFinish"], np.nan)
    hist["championship_position"] = _coalesce(historical, ["championship_position"], np.nan)
    hist["championship_points"] = _coalesce(historical, ["championship_points", "total_points", "Points"])
    hist["championship_wins"] = _coalesce(historical, ["championship_wins"], 0)
    
    if "constructor_name" in historical.columns:
        hist["Team"] = historical["constructor_name"]
    elif "team" in historical.columns:
        hist["Team"] = historical["team"]
    else:
        hist["Team"] = "Unknown"
    hist["source"] = "historical"

    # Process latest/supplemental dataset (e.g. 2025_2026 dataset)
    recent = pd.DataFrame(index=latest.index)
    driver_col = "Driver" if "Driver" in latest.columns else ("driver" if "driver" in latest.columns else latest.columns[0])
    recent["Driver"] = latest[driver_col].astype(str).str.strip()
    
    season_col = "Season" if "Season" in latest.columns else ("year" if "year" in latest.columns else None)
    if season_col:
        recent["Season"] = pd.to_numeric(latest[season_col], errors="coerce").astype("Int64")
    else:
        recent["Season"] = 2026  # Default fallback for current active dataset
        
    recent["driverId"] = pd.NA
    recent["total_points"] = _coalesce(latest, ["TotalPoints", "total_points", "Points", "AveragePoints", "championship_points"])
    recent["races_started"] = _coalesce(latest, ["Races", "races_started", "races"])
    recent["average_grid"] = _coalesce(latest, ["AverageQualifying", "average_grid", "average_qualifying"], np.nan)
    recent["average_finish"] = _coalesce(latest, ["AverageFinish", "average_finish"], np.nan)
    recent["podiums"] = _coalesce(latest, ["Podiums", "podiums"])
    recent["wins"] = _coalesce(latest, ["Wins", "wins"])
    recent["best_finish"] = _coalesce(latest, ["BestFinish", "best_finish"], np.nan)
    recent["championship_position"] = np.nan
    recent["championship_points"] = recent["total_points"]
    recent["championship_wins"] = recent["wins"]
    
    t_col = detect_constructor_column(latest)
    recent["Team"] = latest[t_col].astype(str).str.strip() if t_col else (latest["Team"] if "Team" in latest.columns else "Unknown")
    recent["source"] = "supplemental"

    combined = pd.concat([hist, recent], ignore_index=True)
    combined["Season"] = pd.to_numeric(combined["Season"], errors="coerce")
    for column in FEATURES:
        if column in combined.columns:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined.dropna(subset=["Season", "Driver"]).sort_values(["Season", "Driver"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def train_driver_model(historical: pd.DataFrame) -> tuple[Pipeline | None, dict]:
    if "driverId" not in historical or "year" not in historical:
        return None, {"samples": 0, "mae": None, "r2": None, "status": "missing linkage fields"}
    work = historical.copy()
    work["driverId"] = pd.to_numeric(work["driverId"], errors="coerce")
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    points = pd.to_numeric(work.get("championship_points", work.get("total_points")), errors="coerce")
    lookup = pd.DataFrame({"driverId": work["driverId"], "year": work["year"], "next_points": points})
    lookup["year"] = lookup["year"] - 1
    work = work.merge(lookup, on=["driverId", "year"], how="left")
    
    source = pd.DataFrame(index=work.index)
    for feat in FEATURES:
        if feat in work.columns:
            source[feat] = _coalesce(work, [feat], np.nan)
    target = pd.to_numeric(work["next_points"], errors="coerce")
    valid = target.notna() & source["total_points"].notna()
    available_feats = [f for f in FEATURES if f in source.columns]
    X, y = source.loc[valid, available_feats], target.loc[valid]
    if len(X) < 10:
        return None, {"samples": int(len(X)), "mae": None, "r2": None, "status": "not enough linked seasons"}

    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=260, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    evaluator = Pipeline(base.steps)
    evaluator.fit(X_train, y_train)
    predictions = evaluator.predict(X_test)
    metrics = {
        "samples": int(len(X)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "status": "ready",
    }
    base.fit(X, y)
    return base, metrics


def relative_probabilities(values: pd.Series, temperature: float = 1.0) -> pd.Series:
    """Robust softmax probability distribution calculation."""
    values = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0)
    if values.sum() <= 0:
        return pd.Series(np.repeat(100 / max(len(values), 1), len(values)), index=values.index)
    
    max_val = values.max()
    scaled = (values / max_val) if max_val > 0 else values
    logits = scaled / max(temperature, 0.05)
    exp_values = np.exp(np.clip(logits, -20, 20))
    return exp_values / exp_values.sum() * 100


def forecast_driver_season(canonical: pd.DataFrame, model: Pipeline | None, target_year: int) -> tuple[pd.DataFrame, int, str]:
    # DIRECT LIVE SEASON MATCH: If exact target year data exists directly in canonical (e.g. current 2026 dataset), use it directly!
    exact_match = canonical[canonical["Season"] == target_year]
    if not exact_match.empty and ("total_points" in exact_match.columns and exact_match["total_points"].sum() > 0):
        out = exact_match.copy()
        out["PredictedPoints"] = pd.to_numeric(out["total_points"], errors="coerce").fillna(0)
        out["TargetYear"] = target_year
        out["SourceYear"] = target_year
        out["Probability"] = relative_probabilities(out["PredictedPoints"])
        return out.sort_values(["PredictedPoints", "wins", "podiums"], ascending=False).reset_index(drop=True), target_year, "current live standings"

    # Otherwise fallback to predictive modeling using prior/nearest data
    prior = canonical[canonical["Season"] == target_year - 1]
    source_year = target_year - 1
    source_label = "prior season"
    if prior.empty:
        earlier = canonical[canonical["Season"] < target_year]
        if not earlier.empty:
            source_year = int(earlier["Season"].max())
            prior = earlier[earlier["Season"] == source_year]
            source_label = "nearest available season"
        else:
            prior = exact_match
            source_year = target_year
            source_label = "same-season fallback"

    if prior.empty:
        return pd.DataFrame(), source_year, source_label

    available_feats = [f for f in FEATURES if f in prior.columns]
    X = prior[available_feats].copy()
    if model is not None and len(available_feats) >= len(FEATURES):
        try:
            predicted = model.predict(X)
        except Exception:
            predicted = pd.to_numeric(prior["total_points"], errors="coerce").fillna(0).to_numpy()
        carry = pd.to_numeric(prior["total_points"], errors="coerce").fillna(0).to_numpy()
        predicted = .72 * predicted + .28 * carry
    else:
        predicted = pd.to_numeric(prior["total_points"], errors="coerce").fillna(0).to_numpy()

    output = prior.copy()
    output["PredictedPoints"] = np.maximum(0, np.nan_to_num(predicted, nan=0.0))
    output["TargetYear"] = target_year
    output["SourceYear"] = source_year
    output["Probability"] = relative_probabilities(output["PredictedPoints"])
    return output.sort_values(["PredictedPoints", "wins", "podiums"], ascending=False).reset_index(drop=True), source_year, source_label


def detect_constructor_column(*frames: pd.DataFrame) -> str | None:
    candidates = {"constructor", "constructor_name", "constructorname", "team", "team_name", "teamname"}
    for frame in frames:
        for column in frame.columns:
            normalized = str(column).lower().replace(" ", "_")
            if normalized in candidates:
                return column
    return None


def constructor_frame(canonical_df: pd.DataFrame, *raw_frames: pd.DataFrame) -> pd.DataFrame:
    if "Team" in canonical_df.columns and "Season" in canonical_df.columns:
        sub = canonical_df.dropna(subset=["Team", "Season"]).copy()
        if not sub.empty and len(sub["Team"].unique()) > 1:
            agg = sub.groupby(["Season", "Team"], as_index=False).agg({
                "total_points": "sum",
                "wins": "sum",
                "podiums": "sum",
                "average_finish": "mean"
            }).rename(columns={"total_points": "Points", "wins": "Wins", "podiums": "Podiums", "average_finish": "AvgFinish"})
            return agg

    frames = []
    for raw in raw_frames:
        column = detect_constructor_column(raw)
        if not column:
            continue
        work = raw.copy()
        season_col = "Season" if "Season" in work.columns else "year" if "year" in work.columns else None
        if season_col:
            work["Season"] = pd.to_numeric(work[season_col], errors="coerce")
        else:
            source_year = re.search(r"(?:19|20)\d{2}", str(raw.attrs.get("source_file", "")))
            if "Team" not in work.columns or not source_year:
                continue
            work["Season"] = int(source_year.group(0))
        work["Team"] = work[column].astype(str).str.strip()
        work["Points"] = _coalesce(work, ["TotalPoints", "total_points", "championship_points", "Points", "Team Points", "team_points"])
        work["Wins"] = _coalesce(work, ["Wins", "wins"])
        work["Podiums"] = _coalesce(work, ["Podiums", "podiums"])
        work["AvgFinish"] = _coalesce(work, ["AverageFinish", "average_finish", "AvgFinish"], np.nan)
        frames.append(work[["Season", "Team", "Points", "Wins", "Podiums", "AvgFinish"]])
    if not frames:
        return pd.DataFrame()
    all_rows = pd.concat(frames, ignore_index=True)
    return all_rows.groupby(["Season", "Team"], as_index=False).agg({"Points": "sum", "Wins": "sum", "Podiums": "sum", "AvgFinish": "mean"})


def chart_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=TEXT),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#29303a", zerolinecolor="#29303a")
    fig.update_yaxes(gridcolor="#29303a", zerolinecolor="#29303a")
    return fig


def podium_markup(frame: pd.DataFrame, name_column: str = "Driver") -> str:
    if frame.empty:
        return "<div class='panel fine-print'>No podium forecast is available for this selection.</div>"
    top = frame.head(3).reset_index(drop=True)
    order = [1, 0, 2] if len(top) >= 3 else list(range(len(top)))
    classes = ["second", "first", "third"]
    items = []
    for slot, idx in enumerate(order):
        if idx >= len(top):
            continue
        row = top.iloc[idx]
        name = html.escape(str(row.get(name_column, "—")))
        points = float(row.get("PredictedPoints", row.get("total_points", 0)))
        items.append(f"<div class='podium {classes[slot]}'><div class='name'>{name}<br><span class='fine-print'>{points:,.0f} pts</span></div><div class='bar'>{idx + 1}</div></div>")
    return "<div class='panel'><div class='section-kicker'>Championship standings podium</div><div class='podium-wrap'>" + "".join(items) + "</div></div>"


def metric_card(label: str, value: str, note: str = "") -> str:
    return f"<div class='metric-card'><div class='metric-label'>{html.escape(label)}</div><div class='metric-value'>{html.escape(value)}</div><div class='metric-note'>{html.escape(note)}</div></div>"


st.markdown(
    """
<div class='hero'>
  <div class='hero-kicker'>Race control / live predictive telemetry</div>
  <h1>FORMULA ONE<br>CHAMPIONSHIP PREDICTOR</h1>
  <p>Live current-season analysis, probability distributions, constructor standings, model explainability, and head-to-head battle simulations.</p>
  <div class='car-card' aria-label='Animated stylized racing car card'></div>
</div>
""",
    unsafe_allow_html=True,
)

try:
    historical_raw, latest_raw = load_raw_data()
    supplemental_raw = load_supplemental_csvs()
    canonical = canonicalize(historical_raw, latest_raw)
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.stop()

model, model_metrics = train_driver_model(historical_raw)
data_frames = [historical_raw, latest_raw, *supplemental_raw]
teams = constructor_frame(canonical, *data_frames)

all_seasons = sorted({int(x) for x in canonical["Season"].dropna().unique()})
min_year = 1950
max_year = 2026
all_drivers = sorted(canonical["Driver"].dropna().unique().tolist())
preferred_driver = "Andrea Kimi Antonelli" if "Andrea Kimi Antonelli" in all_drivers else ("Max Verstappen" if "Max Verstappen" in all_drivers else all_drivers[0])

team_options = ["No constructor field detected"]
if not teams.empty and "Team" in teams.columns:
    team_options = sorted(teams["Team"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("<div class='section-kicker'>Controls / inputs</div>", unsafe_allow_html=True)
    st.markdown("### Forecast setup")
    selected_year = st.slider("Year", min_value=min_year, max_value=max_year, value=max_year, step=1)
    selected_driver = st.selectbox("Driver", all_drivers, index=all_drivers.index(preferred_driver) if preferred_driver in all_drivers else 0)
    selected_team = st.selectbox("Constructor / Team", team_options)
    st.divider()
    st.markdown("### Model telemetry")
    if selected_year == 2026:
        st.success("Live 2026 dataset active")
        st.caption("Using exact cleaned dataset metrics for current standings.")
    elif model_metrics.get("status") == "ready":
        st.success(f"Model online · {model_metrics['samples']:,} linked rows")
        st.caption(f"Holdout MAE: {model_metrics['mae']:.1f} pts · R²: {model_metrics['r2']:.2f}")
    else:
        st.warning("Heuristic fallback active")
    st.caption("Probabilities reflect relative championship confidence share.")

# REPLACED SANDBOX TAB WITH MODEL EXPLAINABILITY
season_tab, driver_tab, constructor_tab, explain_tab, battle_tab = st.tabs([
    "Season Predictor",
    "Driver Analysis",
    "Constructor's Cup",
    "Model Explainability",
    "Head-to-Head Battle",
])

with season_tab:
    st.markdown("<div class='section-kicker'>Module A / season forecast</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{selected_year} championship simulation & probabilities</div>", unsafe_allow_html=True)
    forecast, source_year, source_label = forecast_driver_season(canonical, model, selected_year)
    if forecast.empty:
        st.warning("No driver rows are available for this year.")
    else:
        leader = forecast.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Current leader / Favorite", str(leader["Driver"]), f"{selected_year} standings"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Points tally", f"{leader.get('PredictedPoints', leader.get('total_points', 0)):,.0f}", f"source: {source_year}"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Championship probability", f"{leader['Probability']:.1f}%", "relative share"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Field size", f"{len(forecast)}", "drivers in grid"), unsafe_allow_html=True)
        st.markdown(podium_markup(forecast), unsafe_allow_html=True)
        left, right = st.columns([1.4, 1])
        with left:
            pts_col = "PredictedPoints" if "PredictedPoints" in forecast.columns else "total_points"
            display = forecast[["Driver", pts_col, "Probability", "wins", "podiums"]].copy()
            display.columns = ["Driver", "Points / Score", "Probability %", "Wins", "Podiums"]
            display["Points / Score"] = display["Points / Score"].round(1)
            display["Probability %"] = display["Probability %"].round(1)
            st.dataframe(display, use_container_width=True, hide_index=True)
        with right:
            fig = px.bar(forecast.head(10).sort_values(pts_col), x=pts_col, y="Driver", orientation="h", color="Probability", color_continuous_scale=[[0, "#5c1411"], [1, RED]], title=f"Top 10 Championship Projections ({selected_year})")
            st.plotly_chart(chart_layout(fig, 470), use_container_width=True)

with driver_tab:
    st.markdown("<div class='section-kicker'>Module B / individual deep dive</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{selected_driver} telemetry</div>", unsafe_allow_html=True)
    driver_history = canonical[canonical["Driver"] == selected_driver].sort_values("Season").copy()
    if driver_history.empty:
        st.info("No history found for this driver.")
    else:
        season_totals = canonical.groupby("Season")["championship_points"].sum().replace(0, np.nan)
        driver_history["SeasonProbability"] = driver_history.apply(lambda row: float(row["championship_points"] / season_totals.get(row["Season"], np.nan) * 100) if season_totals.get(row["Season"], np.nan) else 0, axis=1)
        wins = int(driver_history["wins"].sum())
        podiums = int(driver_history["podiums"].sum())
        best_points = float(driver_history["championship_points"].max())
        a, b, c = st.columns(3)
        with a: st.markdown(metric_card("Cumulative wins", f"{wins}", "historical total"), unsafe_allow_html=True)
        with b: st.markdown(metric_card("Cumulative podiums", f"{podiums}", "historical total"), unsafe_allow_html=True)
        with c: st.markdown(metric_card("Best points season", f"{best_points:,.0f}", "single-season peak"), unsafe_allow_html=True)
        
        left, right = st.columns([1.35, 1])
        with left:
            trajectory = driver_history[["Season", "championship_points", "wins", "podiums"]].melt(id_vars="Season", var_name="Metric", value_name="Value")
            trajectory["Metric"] = trajectory["Metric"].map({"championship_points": "Points", "wins": "Wins", "podiums": "Podiums"})
            fig = px.line(trajectory, x="Season", y="Value", color="Metric", markers=True, title="Historical performance trajectory", color_discrete_sequence=[RED, AMBER, GREEN])
            st.plotly_chart(chart_layout(fig, 390), use_container_width=True)
        with right:
            probability = driver_history[["Season", "SeasonProbability"]].rename(columns={"SeasonProbability": "Probability"})
            fig = px.area(probability, x="Season", y="Probability", markers=True, title="Relative season probability share", color_discrete_sequence=[RED])
            st.plotly_chart(chart_layout(fig, 390), use_container_width=True)
        st.dataframe(driver_history[["Season", "championship_points", "wins", "podiums"]].rename(columns={"championship_points": "Points"}).round(1), use_container_width=True, hide_index=True)

with constructor_tab:
    st.markdown("<div class='section-kicker'>Module C / constructor forecast</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{selected_year} Constructor's Cup Standings</div>", unsafe_allow_html=True)
    if teams.empty:
        st.markdown("<div class='alert-strip'><strong>Constructor team data not detected in current frame.</strong></div>", unsafe_allow_html=True)
    else:
        year_teams = teams[teams["Season"] == selected_year]
        if year_teams.empty:
            year_teams = teams[teams["Season"] == teams["Season"].max()]
        year_teams = year_teams.sort_values("Points", ascending=False).reset_index(drop=True)
        year_teams["Probability"] = relative_probabilities(year_teams["Points"])
        
        if not year_teams.empty:
            top_team = year_teams.iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(metric_card("Leading Constructor", str(top_team["Team"]), f"{selected_year} season"), unsafe_allow_html=True)
            with c2: st.markdown(metric_card("Team Points", f"{top_team['Points']:,.0f}", f"Wins: {int(top_team['Wins'])}"), unsafe_allow_html=True)
            with c3: st.markdown(metric_card("Championship Probability", f"{top_team['Probability']:.1f}%", "share"), unsafe_allow_html=True)
            
            st.dataframe(year_teams[["Team", "Points", "Probability", "Wins", "Podiums"]].rename(columns={"Points": "Total Points", "Probability": "Championship Probability %"}).round(1), use_container_width=True, hide_index=True)
            
            fig = px.bar(year_teams, x="Points", y="Team", orientation="h", color="Probability", color_continuous_scale=[[0, "#5c1411"], [1, RED]], title=f"Constructor Standings & Probabilities ({selected_year})")
            st.plotly_chart(chart_layout(fig, 400), use_container_width=True)

with explain_tab:
    st.markdown("<div class='section-kicker'>Advanced / Machine Learning Interpretability</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Random Forest Feature Importance</div>", unsafe_allow_html=True)
    st.write("This chart explains what statistical factors our Scikit-Learn model weighs most heavily when forecasting championship performance.")
    
    if model is not None and hasattr(model.named_steps["model"], "feature_importances_"):
        importances = model.named_steps["model"].feature_importances_
        feat_df = pd.DataFrame({
            "Feature": FEATURES[:len(importances)],
            "Importance": importances
        }).sort_values("Importance", ascending=True)
        
        fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h", 
                     title="Feature Importance Gini Coefficients", 
                     color="Importance", color_continuous_scale=[[0, "#5c1411"], [1, RED]])
        st.plotly_chart(chart_layout(fig, 420), use_container_width=True)
    else:
        st.info("Trainable model metrics are currently running on heuristic fallback; ensure historical data is linked to view feature importances.")

with battle_tab:
    st.markdown("<div class='section-kicker'>Advanced / matchup lab</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Head-to-Head Driver Battle</div>", unsafe_allow_html=True)
    battle_forecast, _, _ = forecast_driver_season(canonical, model, selected_year)
    if len(all_drivers) >= 2 and not battle_forecast.empty:
        d1, d2 = st.columns(2)
        default_a = all_drivers.index("Andrea Kimi Antonelli") if "Andrea Kimi Antonelli" in all_drivers else 0
        default_b = all_drivers.index("George Russell") if "George Russell" in all_drivers else min(1, len(all_drivers) - 1)
        with d1: driver_a = st.selectbox("Driver A", all_drivers, index=default_a, key="battle_a")
        with d2: driver_b = st.selectbox("Driver B", all_drivers, index=default_b, key="battle_b")
        
        row_a = battle_forecast[battle_forecast["Driver"] == driver_a]
        row_b = battle_forecast[battle_forecast["Driver"] == driver_b]
        if row_a.empty or row_b.empty:
            st.info("One or both drivers are missing from the selected year's dataset.")
        else:
            left, right = st.columns(2)
            ra = row_a.iloc[0]
            rb = row_b.iloc[0]
            with left:
                st.markdown(metric_card(driver_a, f"{ra['Probability']:.1f}%", f"{ra.get('PredictedPoints', ra.get('total_points', 0)):,.0f} pts"), unsafe_allow_html=True)
                st.write(f"**Wins:** {int(ra.get('wins', 0))} · **Podiums:** {int(ra.get('podiums', 0))}")
            with right:
                st.markdown(metric_card(driver_b, f"{rb['Probability']:.1f}%", f"{rb.get('PredictedPoints', rb.get('total_points', 0)):,.0f} pts"), unsafe_allow_html=True)
                st.write(f"**Wins:** {int(rb.get('wins', 0))} · **Podiums:** {int(rb.get('podiums', 0))}")
                
            winner = driver_a if float(ra["Probability"]) >= float(rb["Probability"]) else driver_b
            st.markdown(f"<div class='panel' style='margin-top:1rem'><span class='section-kicker'>Matchup Verdict</span><br><strong>{html.escape(winner)}</strong> holds the stronger probability projection in this head-to-head comparison.</div>", unsafe_allow_html=True)
            
            battle_df = pd.DataFrame({
                "Driver": [driver_a, driver_b],
                "Probability": [ra["Probability"], rb["Probability"]],
                "Points": [ra.get('PredictedPoints', ra.get('total_points', 0)), rb.get('PredictedPoints', rb.get('total_points', 0))]
            })
            fig = px.bar(battle_df, x="Driver", y="Probability", color="Driver", text="Probability", title="Head-to-Head Win Probability Comparison", color_discrete_sequence=[RED, "#747f91"])
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(chart_layout(fig, 340), use_container_width=True)
    else:
        st.info("At least two drivers are required for a battle comparison.")

st.divider()
st.markdown("<div class='fine-print'>Model scope: Integrates current live 2026 dataset tracking with supervised machine learning and model explainability charts.</div>", unsafe_allow_html=True)
