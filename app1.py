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
    """Load any additional CSVs, such as a constructor/team dataset, without hard-coding its filename."""
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
    hist = pd.DataFrame(index=historical.index)
    hist["Driver"] = historical["driver_first_name"].astype(str).str.strip() + " " + historical["driver_last_name"].astype(str).str.strip()
    hist["Season"] = pd.to_numeric(historical["year"], errors="coerce").astype("Int64")
    hist["driverId"] = historical.get("driverId", pd.Series(pd.NA, index=historical.index))
    hist["total_points"] = _coalesce(historical, ["total_points", "championship_points"])
    hist["races_started"] = _coalesce(historical, ["races_started", "races"])
    hist["average_grid"] = _coalesce(historical, ["average_grid", "average_qualifying"], np.nan)
    hist["average_finish"] = _coalesce(historical, ["average_finish", "AverageFinish"], np.nan)
    hist["podiums"] = _coalesce(historical, ["podiums"])
    hist["wins"] = _coalesce(historical, ["wins"])
    hist["best_finish"] = _coalesce(historical, ["best_finish", "BestFinish"], np.nan)
    hist["championship_position"] = _coalesce(historical, ["championship_position"], np.nan)
    hist["championship_points"] = _coalesce(historical, ["championship_points", "total_points"])
    hist["championship_wins"] = _coalesce(historical, ["championship_wins"], 0)
    hist["source"] = "historical"

    recent = pd.DataFrame(index=latest.index)
    recent["Driver"] = latest["Driver"].astype(str).str.strip()
    recent["Season"] = pd.to_numeric(latest["Season"], errors="coerce").astype("Int64")
    recent["driverId"] = pd.NA
    recent["total_points"] = _coalesce(latest, ["TotalPoints", "AveragePoints"])
    recent["races_started"] = _coalesce(latest, ["Races"])
    recent["average_grid"] = _coalesce(latest, ["AverageQualifying"], np.nan)
    recent["average_finish"] = _coalesce(latest, ["AverageFinish"], np.nan)
    recent["podiums"] = _coalesce(latest, ["Podiums"])
    recent["wins"] = _coalesce(latest, ["Wins"])
    recent["best_finish"] = _coalesce(latest, ["BestFinish"], np.nan)
    recent["championship_position"] = np.nan
    recent["championship_points"] = _coalesce(latest, ["TotalPoints", "AveragePoints"])
    recent["championship_wins"] = 0.0
    recent["source"] = "supplemental"

    combined = pd.concat([hist, recent], ignore_index=True)
    combined["Season"] = pd.to_numeric(combined["Season"], errors="coerce")
    for column in FEATURES:
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
    source["total_points"] = _coalesce(work, ["total_points", "championship_points"])
    source["races_started"] = _coalesce(work, ["races_started"])
    source["average_grid"] = _coalesce(work, ["average_grid"], np.nan)
    source["average_finish"] = _coalesce(work, ["average_finish"], np.nan)
    source["podiums"] = _coalesce(work, ["podiums"])
    source["wins"] = _coalesce(work, ["wins"])
    source["best_finish"] = _coalesce(work, ["best_finish"], np.nan)
    source["championship_position"] = _coalesce(work, ["championship_position"], np.nan)
    source["championship_points"] = _coalesce(work, ["championship_points", "total_points"])
    source["championship_wins"] = _coalesce(work, ["championship_wins"], 0)
    target = pd.to_numeric(work["next_points"], errors="coerce")
    valid = target.notna() & source["total_points"].notna()
    X, y = source.loc[valid, FEATURES], target.loc[valid]
    if len(X) < 30:
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


def source_rows_for_year(canonical: pd.DataFrame, target_year: int) -> tuple[pd.DataFrame, int, str]:
    prior = canonical[canonical["Season"] == target_year - 1]
    if not prior.empty:
        return prior.copy(), target_year - 1, "prior season"
    earlier = canonical[canonical["Season"] < target_year]
    if not earlier.empty:
        nearest = int(earlier["Season"].max())
        return earlier[earlier["Season"] == nearest].copy(), nearest, "nearest available season"
    exact = canonical[canonical["Season"] == target_year]
    return exact.copy(), target_year, "same-season fallback"


def relative_probabilities(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0)
    if values.sum() <= 0:
        return pd.Series(np.repeat(100 / max(len(values), 1), len(values)), index=values.index)
    centered = (values - values.max()) / max(values.std(ddof=0), 1.0)
    exp_values = np.exp(np.clip(centered, -20, 20))
    return exp_values / exp_values.sum() * 100


def apply_sandbox(
    frame: pd.DataFrame,
    point_multiplier: float = 1.0,
    finish_weight: float = 0.0,
    podium_weight: float = 0.0,
    win_bonus: float = 0.0,
) -> pd.DataFrame:
    result = frame.copy()
    finish_signal = (22 - pd.to_numeric(result["average_finish"], errors="coerce").fillna(22)).clip(0, 22) / 22
    podium_signal = pd.to_numeric(result["podiums"], errors="coerce").fillna(0)
    if podium_signal.max() > 0:
        podium_signal = podium_signal / podium_signal.max()
    wins = pd.to_numeric(result["wins"], errors="coerce").fillna(0)
    result["ScenarioPoints"] = (
        pd.to_numeric(result["PredictedPoints"], errors="coerce").fillna(0) * point_multiplier
        + finish_signal * finish_weight * 15
        + podium_signal * podium_weight * 10
        + wins * win_bonus * 5
    ).clip(lower=0)
    result["Probability"] = relative_probabilities(result["ScenarioPoints"])
    return result.sort_values(["ScenarioPoints", "Probability"], ascending=False).reset_index(drop=True)


def forecast_driver_season(canonical: pd.DataFrame, model: Pipeline | None, target_year: int) -> tuple[pd.DataFrame, int, str]:
    source, source_year, source_label = source_rows_for_year(canonical, target_year)
    if source.empty:
        return pd.DataFrame(), source_year, source_label
    X = source[FEATURES].copy()
    if model is not None:
        predicted = model.predict(X)
        carry = pd.to_numeric(source["total_points"], errors="coerce").fillna(0).to_numpy()
        predicted = .72 * predicted + .28 * carry
    else:
        predicted = pd.to_numeric(source["total_points"], errors="coerce").fillna(0).to_numpy()
    output = source.copy()
    output["PredictedPoints"] = np.maximum(0, np.nan_to_num(predicted, nan=0.0))
    output["TargetYear"] = target_year
    output["SourceYear"] = source_year
    output["Probability"] = relative_probabilities(output["PredictedPoints"])
    return output.sort_values("PredictedPoints", ascending=False).reset_index(drop=True), source_year, source_label


def detect_constructor_column(*frames: pd.DataFrame) -> str | None:
    candidates = {"constructor", "constructor_name", "constructorname", "team", "team_name", "teamname"}
    for frame in frames:
        for column in frame.columns:
            normalized = str(column).lower().replace(" ", "_")
            if normalized in candidates:
                return column
    return None


def constructor_frame(*raw_frames: pd.DataFrame) -> pd.DataFrame:
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


@st.cache_resource(show_spinner=False)
def train_constructor_model(teams: pd.DataFrame) -> tuple[Pipeline | None, dict]:
    """Train a team-level next-season points model when constructor data is available."""
    if teams.empty or not set(TEAM_FEATURES).issubset(teams.columns):
        return None, {"samples": 0, "status": "no constructor data"}
    work = teams.copy().sort_values(["Team", "Season"])
    lookup = work[["Team", "Season", "Points"]].rename(columns={"Points": "next_points"}).copy()
    lookup["Season"] = lookup["Season"] - 1
    work = work.merge(lookup, on=["Team", "Season"], how="left")
    X = work[TEAM_FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(work["next_points"], errors="coerce")
    valid = y.notna()
    if int(valid.sum()) < 10:
        return None, {"samples": int(valid.sum()), "status": "not enough linked team seasons"}
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=180, max_depth=10, min_samples_leaf=1, random_state=42, n_jobs=-1)),
    ])
    model.fit(X.loc[valid], y.loc[valid])
    return model, {"samples": int(valid.sum()), "status": "ready"}


def forecast_constructors(teams: pd.DataFrame, target_year: int, model: Pipeline | None = None) -> tuple[pd.DataFrame, int]:
    if teams.empty:
        return teams, target_year - 1
    prior = teams[teams["Season"] == target_year - 1]
    if prior.empty:
        prior = teams[teams["Season"] == teams[teams["Season"] < target_year]["Season"].max()]
    if prior.empty:
        prior = teams[teams["Season"] == target_year]
    out = prior.copy()
    if model is not None:
        learned = model.predict(out[TEAM_FEATURES].apply(pd.to_numeric, errors="coerce"))
        out["PredictedPoints"] = np.maximum(0, 0.78 * learned + 0.22 * out["Points"].to_numpy())
    else:
        out["PredictedPoints"] = out["Points"] + out["Wins"] * 5 + out["Podiums"] * 1.5
    out["Probability"] = relative_probabilities(out["PredictedPoints"])
    return out.sort_values("PredictedPoints", ascending=False).reset_index(drop=True), int(prior["Season"].iloc[0]) if not prior.empty else target_year


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
        points = float(row.get("ScenarioPoints", row.get("PredictedPoints", 0)))
        items.append(f"<div class='podium {classes[slot]}'><div class='name'>{name}<br><span class='fine-print'>{points:,.0f} pts</span></div><div class='bar'>{idx + 1}</div></div>")
    return "<div class='panel'><div class='section-kicker'>Championship podium</div><div class='podium-wrap'>" + "".join(items) + "</div></div>"


def metric_card(label: str, value: str, note: str = "") -> str:
    return f"<div class='metric-card'><div class='metric-label'>{html.escape(label)}</div><div class='metric-value'>{html.escape(value)}</div><div class='metric-note'>{html.escape(note)}</div></div>"


st.markdown(
    """
<div class='hero'>
  <div class='hero-kicker'>Race control / predictive telemetry</div>
  <h1>FORMULA ONE<br>CHAMPIONSHIP PREDICTOR</h1>
  <p>Run the season before lights out. Forecast the championship, inspect driver form, and stress-test the points system with an explainable Scikit-Learn model.</p>
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
constructor_column = detect_constructor_column(*data_frames)
teams = constructor_frame(*data_frames)
team_model, team_metrics = train_constructor_model(teams)
constructor_results_loaded = any({"constructorId", "raceId"}.issubset(frame.columns) for frame in supplemental_raw)

all_seasons = sorted({int(x) for x in canonical["Season"].dropna().unique()})
min_year = 1950
max_year = 2026
all_drivers = sorted(canonical["Driver"].dropna().unique().tolist())
preferred_driver = "Max Verstappen" if "Max Verstappen" in all_drivers else all_drivers[0]
team_options = ["No constructor field detected"]
if constructor_column:
    team_values = []
    for frame in data_frames:
        if constructor_column in frame.columns:
            team_values.extend(frame[constructor_column].dropna().astype(str).unique().tolist())
    team_options = sorted(set(team_values)) or team_options

with st.sidebar:
    st.markdown("<div class='section-kicker'>Controls / inputs</div>", unsafe_allow_html=True)
    st.markdown("### Forecast setup")
    selected_year = st.slider("Year", min_value=min_year, max_value=max_year, value=max_year, step=1)
    selected_driver = st.selectbox("Driver", all_drivers, index=all_drivers.index(preferred_driver))
    selected_team = st.selectbox("Constructor / Team", team_options)
    st.divider()
    st.markdown("### Model telemetry")
    if model_metrics.get("status") == "ready":
        st.success(f"Model online · {model_metrics['samples']:,} linked rows")
        st.caption(f"Holdout MAE: {model_metrics['mae']:.1f} pts · R²: {model_metrics['r2']:.2f}")
    else:
        st.warning("Heuristic fallback active")
        st.caption(model_metrics.get("status", "Model unavailable"))
    st.caption("Probabilities are relative model confidence scores, not betting odds.")

season_tab, driver_tab, constructor_tab, sandbox_tab, battle_tab = st.tabs([
    "Season Predictor",
    "Driver Analysis",
    "Constructor's Cup",
    "What-If Sandbox",
    "Head-to-Head Battle",
])

with season_tab:
    st.markdown("<div class='section-kicker'>Module A / season forecast</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{selected_year} championship simulation</div>", unsafe_allow_html=True)
    forecast, source_year, source_label = forecast_driver_season(canonical, model, selected_year)
    forecast = apply_sandbox(forecast)
    if forecast.empty:
        st.warning("No driver rows are available for this year.")
    else:
        champion = forecast.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Predicted champion", str(champion["Driver"]), f"{selected_year} forecast"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Forecast points", f"{champion['ScenarioPoints']:,.0f}", f"source: {source_year}"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Relative probability", f"{champion['Probability']:.1f}%", "model-derived"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Field size", f"{len(forecast)}", "drivers in source season"), unsafe_allow_html=True)
        st.caption(f"Inputs use the {source_year} {source_label} rows to forecast {selected_year}. The supplemental file is included for the latest seasons.")
        st.markdown(podium_markup(forecast), unsafe_allow_html=True)
        left, right = st.columns([1.4, 1])
        with left:
            display = forecast[["Driver", "ScenarioPoints", "Probability", "PredictedPoints", "wins", "podiums", "average_finish"]].copy()
            display.columns = ["Driver", "Scenario points", "Win probability %", "Base forecast points", "Wins", "Podiums", "Avg finish"]
            display["Scenario points"] = display["Scenario points"].round(1)
            display["Win probability %"] = display["Win probability %"].round(1)
            display["Base forecast points"] = display["Base forecast points"].round(1)
            st.dataframe(display, use_container_width=True, hide_index=True)
        with right:
            fig = px.bar(forecast.head(10).sort_values("ScenarioPoints"), x="ScenarioPoints", y="Driver", orientation="h", color="Probability", color_continuous_scale=[[0, "#5c1411"], [1, RED]], title="Top 10 forecast points")
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
        titles = int((driver_history["championship_position"] == 1).fillna(False).sum())
        if titles == 0:
            titles = int((driver_history["championship_wins"] > 0).sum())
        wins = int(driver_history["wins"].sum())
        podiums = int(driver_history["podiums"].sum())
        best_points = float(driver_history["championship_points"].max())
        success_rate = titles / max(len(driver_history), 1) * 100
        a, b, c, d = st.columns(4)
        with a: st.markdown(metric_card("Cumulative wins", f"{wins}", "historical total"), unsafe_allow_html=True)
        with b: st.markdown(metric_card("Cumulative podiums", f"{podiums}", "historical total"), unsafe_allow_html=True)
        with c: st.markdown(metric_card("Best points season", f"{best_points:,.0f}", "single-season peak"), unsafe_allow_html=True)
        with d: st.markdown(metric_card("Title success rate", f"{success_rate:.1f}%", f"{titles} title(s) / {len(driver_history)} seasons"), unsafe_allow_html=True)
        left, right = st.columns([1.35, 1])
        with left:
            trajectory = driver_history[["Season", "championship_points", "wins", "podiums"]].melt(id_vars="Season", var_name="Metric", value_name="Value")
            trajectory["Metric"] = trajectory["Metric"].map({"championship_points": "Points", "wins": "Wins", "podiums": "Podiums"})
            fig = px.line(trajectory, x="Season", y="Value", color="Metric", markers=True, title="Historical performance trajectory", color_discrete_sequence=[RED, AMBER, GREEN])
            st.plotly_chart(chart_layout(fig, 390), use_container_width=True)
        with right:
            probability = driver_history[["Season", "SeasonProbability"]].rename(columns={"SeasonProbability": "Probability"})
            fig = px.area(probability, x="Season", y="Probability", markers=True, title="Relative season probability", color_discrete_sequence=[RED])
            st.plotly_chart(chart_layout(fig, 390), use_container_width=True)
        st.dataframe(driver_history[["Season", "championship_points", "championship_position", "wins", "podiums", "SeasonProbability"]].rename(columns={"championship_points": "Points", "championship_position": "Championship position", "SeasonProbability": "Relative probability %"}).round(1), use_container_width=True, hide_index=True)

with constructor_tab:
    st.markdown("<div class='section-kicker'>Module C / constructor forecast</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Constructor's Cup</div>", unsafe_allow_html=True)
    if not constructor_column:
        if constructor_results_loaded:
            st.markdown("<div class='alert-strip'><strong>Constructor results file detected, but it is results-only.</strong><br><code>constructor_results.csv</code> contains race/constructor IDs and points, but no season or constructor-name fields. Add matching <code>races.csv</code> and <code>constructors.csv</code> files, or a team-season CSV with <code>Season</code>/<code>year</code> plus <code>constructor</code>/<code>team</code>, to activate named 1950–2026 team forecasts.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-strip'><strong>Constructor data not detected.</strong><br>The supplied driver files contain no constructor/team column. The selector is kept in the interface and the forecasting code is schema-ready; add a column named <code>constructor</code>, <code>constructor_name</code>, <code>team</code>, or <code>team_name</code> to activate team predictions.</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel fine-print'>No team affiliations are fabricated from driver names. Once named team-season data is available, the app trains a dedicated next-season constructor model and uses it for team rankings and filters.</div>", unsafe_allow_html=True)
    else:
        team_forecast, team_source_year = forecast_constructors(teams, selected_year, team_model)
        team_champion = team_forecast.iloc[0] if not team_forecast.empty else None
        if team_champion is not None:
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(metric_card("Predicted constructor champion", str(team_champion["Team"]), f"{selected_year} forecast"), unsafe_allow_html=True)
            with c2: st.markdown(metric_card("Predicted team points", f"{team_champion['PredictedPoints']:,.0f}", f"source: {team_source_year}"), unsafe_allow_html=True)
            with c3: st.markdown(metric_card("Relative probability", f"{team_champion['Probability']:.1f}%", "model-derived"), unsafe_allow_html=True)
            st.dataframe(team_forecast[["Team", "PredictedPoints", "Probability", "Points", "Wins", "Podiums"]].rename(columns={"PredictedPoints": "Predicted points", "Probability": "Championship probability %"}).round(1), use_container_width=True, hide_index=True)
            if selected_team in set(team_forecast["Team"]):
                st.markdown(f"#### {selected_team} historical tracking")
                st.dataframe(teams[teams["Team"] == selected_team].sort_values("Season", ascending=False), use_container_width=True, hide_index=True)

with sandbox_tab:
    st.markdown("<div class='section-kicker'>Advanced / scenario engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>What-If Sandbox</div>", unsafe_allow_html=True)
    st.write("Change the scoring philosophy and watch the predicted order update. This is a transparent scenario layer over the base model, not a hidden re-training step.")
    s1, s2, s3, s4 = st.columns(4)
    with s1: point_multiplier = st.slider("Points multiplier", 0.50, 1.50, 1.00, 0.05, key="sandbox_points")
    with s2: finish_weight = st.slider("Finish-performance weight", 0.0, 5.0, 0.0, 0.25, key="sandbox_finish")
    with s3: podium_weight = st.slider("Podium consistency weight", 0.0, 5.0, 0.0, 0.25, key="sandbox_podium")
    with s4: win_bonus = st.slider("Win bonus weight", 0.0, 5.0, 0.0, 0.25, key="sandbox_wins")
    sandbox_forecast, sandbox_source_year, _ = forecast_driver_season(canonical, model, selected_year)
    sandbox_forecast = apply_sandbox(sandbox_forecast, point_multiplier, finish_weight, podium_weight, win_bonus)
    if not sandbox_forecast.empty:
        champ = sandbox_forecast.iloc[0]
        st.markdown(f"<div class='panel'><span class='section-kicker'>Scenario result</span><br><strong style='font-size:1.5rem'>{html.escape(str(champ['Driver']))}</strong> leads the {selected_year} field with <strong>{champ['ScenarioPoints']:,.1f} scenario points</strong> and {champ['Probability']:.1f}% relative probability.</div>", unsafe_allow_html=True)
        st.markdown(podium_markup(sandbox_forecast), unsafe_allow_html=True)
        fig = px.bar(sandbox_forecast.head(12).sort_values("ScenarioPoints"), x="ScenarioPoints", y="Driver", orientation="h", color="Probability", color_continuous_scale=[[0, "#5c1411"], [1, RED]], title="Scenario standings")
        st.plotly_chart(chart_layout(fig, 520), use_container_width=True)

with battle_tab:
    st.markdown("<div class='section-kicker'>Advanced / matchup lab</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Head-to-Head Driver Battle</div>", unsafe_allow_html=True)
    battle_forecast, _, _ = forecast_driver_season(canonical, model, selected_year)
    if len(all_drivers) >= 2 and not battle_forecast.empty:
        d1, d2 = st.columns(2)
        default_a = all_drivers.index("Max Verstappen") if "Max Verstappen" in all_drivers else 0
        default_b = all_drivers.index("Lewis Hamilton") if "Lewis Hamilton" in all_drivers else min(1, len(all_drivers) - 1)
        with d1: driver_a = st.selectbox("Driver A", all_drivers, index=default_a, key="battle_a")
        with d2: driver_b = st.selectbox("Driver B", all_drivers, index=default_b, key="battle_b")
        row_a = battle_forecast[battle_forecast["Driver"] == driver_a]
        row_b = battle_forecast[battle_forecast["Driver"] == driver_b]
        if row_a.empty or row_b.empty:
            st.info("One or both drivers are not present in the selected season's source field. Try another year.")
        else:
            left, right = st.columns(2)
            for holder, row, name in ((left, row_a.iloc[0], driver_a), (right, row_b.iloc[0], driver_b)):
                with holder:
                    st.markdown(metric_card(name, f"{row['Probability']:.1f}%", f"{row['PredictedPoints']:,.0f} predicted points"), unsafe_allow_html=True)
                    st.write(f"**Wins:** {int(row['wins'])} · **Podiums:** {int(row['podiums'])} · **Avg finish:** {row['average_finish']:.1f}")
            winner = driver_a if float(row_a.iloc[0]["Probability"]) >= float(row_b.iloc[0]["Probability"]) else driver_b
            st.markdown(f"<div class='panel' style='margin-top:1rem'><span class='section-kicker'>Battle call</span><br><strong>{html.escape(winner)}</strong> has the higher model-derived relative probability for the {selected_year} forecast.</div>", unsafe_allow_html=True)
            battle_df = pd.DataFrame({"Driver": [driver_a, driver_b], "Probability": [row_a.iloc[0]["Probability"], row_b.iloc[0]["Probability"]], "Predicted points": [row_a.iloc[0]["PredictedPoints"], row_b.iloc[0]["PredictedPoints"]]})
            fig = px.bar(battle_df, x="Driver", y="Probability", color="Driver", text="Probability", title="Side-by-side relative probability", color_discrete_sequence=[RED, "#747f91"])
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(chart_layout(fig, 340), use_container_width=True)
    else:
        st.info("At least two drivers are required for a battle comparison.")

st.divider()
st.markdown("<div class='fine-print'>Model scope: driver-season prediction trained from the supplied historical file and blended with the latest supplemental driver metrics. Constructor predictions activate when a constructor/team field is available. For deployment instructions, open README.md in the project.</div>", unsafe_allow_html=True)
