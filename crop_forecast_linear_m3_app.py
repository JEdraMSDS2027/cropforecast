from pathlib import Path
import warnings

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_linear_regression.joblib"
DATA_PATH = APP_DIR / "crop_data_no_duplicate.csv"
LAST_COMPLETE_QUARTER = "2025Q3"
M3_FEATURES = [
    "province",
    "crop",
    "next_quarter_num",
    "production_t",
    "production_lag3",
    "production_lag7",
    "production_lag1",
    "production_lag2",
    "production_roll4_mean",
]
PRODUCTION_LAGS = (1, 2, 3, 7)
RESULT_KEY = "linear_m3_forecast"
BASELINE_KEY = "linear_m3_seasonal_baseline"

APP_STYLE = """
<style>
.block-container { max-width: 1440px; padding-top: 4.5rem; padding-bottom: 2rem; }
[data-testid="stCaptionContainer"] { color: #56685b; }
[data-testid="stSidebar"] { background-color: #edf2e9; }
[data-testid="stSidebar"] h1 { font-size: 1.65rem; padding-bottom: 0.25rem; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #244d37; font-size: 1.1rem; padding-top: 0.2rem; padding-bottom: 0.3rem;
}
[data-testid="stSidebar"] hr { margin: 0.65rem 0; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.7rem; }
[data-testid="stSidebar"] [data-testid="stButton"] button { border-radius: 9px; }
.st-key-outlook_header {
    background: linear-gradient(110deg, #153e34, #2d5e48);
    border-radius: 16px; padding: 1.55rem 1.8rem; margin-bottom: 0.8rem;
}
.st-key-outlook_header h1 { color: #ffffff; font-size: 2.35rem; padding-top: 0; }
.st-key-outlook_header h3 { color: #ffffff; font-size: 1.65rem; }
.st-key-outlook_header p { color: #dce9df; }
.st-key-outlook_header [data-testid="stCaptionContainer"] { color: #dce9df; }
.st-key-forecast_card, .st-key-reported_card, .st-key-baseline_card {
    background: #ffffff; border-radius: 12px; padding: 1.15rem; min-height: 180px;
}
.st-key-forecast_card { border-top: 4px solid #287a50; }
.st-key-summary [data-testid="stMetricValue"] { font-size: 2rem; }
.st-key-summary [data-testid="stMetricLabel"] { color: #52665a; }
.st-key-trend_panel {
    background: #ffffff; border-radius: 14px; padding: 1.15rem 1.3rem;
}
.st-key-trend_panel h3 { padding-top: 0; }
@media (max-width: 760px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .st-key-outlook_header { padding: 1.1rem; }
    .st-key-outlook_header h1 { font-size: 1.8rem; }
    .st-key-trend_panel { padding: 0.8rem; }
}
</style>
"""


# The notebook saved this class as __main__.FeatureSelector. Keep its definition
# available when joblib restores the fitted pipeline in the Streamlit script.
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, self.columns].copy()


@st.cache_resource
def load_model(path, modified_at):
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(path)

    if not isinstance(model, Pipeline):
        raise ValueError("The model file must contain the complete fitted pipeline.")
    check_is_fitted(model)
    selector = model.named_steps.get("select")
    columns = getattr(selector, "columns", None)
    if columns is None:
        columns = getattr(model, "feature_names_in_", None)
    if columns is None or set(columns) != set(M3_FEATURES):
        raise ValueError("The saved pipeline must use the nine M3 production features.")
    return model, list(columns)


@st.cache_data
def load_history(path, modified_at, last_complete_quarter):
    columns = ["province", "crop", "year", "quarter", "production"]
    history = pd.read_csv(path, usecols=columns).drop_duplicates()
    if history[columns].isna().any().any():
        raise ValueError("The production history contains missing values.")
    history["province"] = history["province"].astype(str).str.strip().str.upper()
    history["crop"] = history["crop"].astype(str).str.strip()
    quarter = history["quarter"].astype(str).str.strip().str.upper()
    if not quarter.str.fullmatch(r"Q[1-4]").all():
        raise ValueError("Quarter labels must be Q1, Q2, Q3, or Q4.")
    year = pd.to_numeric(history["year"], errors="raise")
    if not year.mod(1).eq(0).all():
        raise ValueError("Years must be whole numbers.")
    history["period"] = pd.PeriodIndex(year.astype(int).astype(str) + quarter, freq="Q-DEC")
    history["production"] = pd.to_numeric(history["production"], errors="raise")
    if not np.isfinite(history["production"]).all() or history["production"].lt(0).any():
        raise ValueError("Production must contain finite, nonnegative values.")
    if history.duplicated(["province", "crop", "period"]).any():
        raise ValueError("Conflicting province-crop-quarter production records remain.")
    cutoff = pd.Period(last_complete_quarter, freq="Q-DEC")
    history = history.loc[history["period"].le(cutoff)].copy()
    if history.empty:
        raise ValueError("There are no production records in the selected date range.")
    return history.sort_values(["province", "crop", "period"])


def build_features(province, crop, actual, origin, current_production):
    row = {
        "province": province,
        "crop": crop,
        "next_quarter_num": str((origin + 1).quarter),
        "production_t": float(current_production),
    }
    for lag in PRODUCTION_LAGS:
        row[f"production_lag{lag}"] = float(actual.get(origin - lag, np.nan))
    # Match the notebook's four-quarter mean, including its missing-value rule.
    row["production_roll4_mean"] = float(np.mean([
        row["production_t"],
        row["production_lag1"],
        row["production_lag2"],
        row["production_lag3"],
    ]))
    return pd.DataFrame([row], columns=M3_FEATURES)


def restore_defaults(widget_key, value):
    st.session_state[widget_key] = value
    st.session_state.pop(RESULT_KEY, None)
    st.session_state.pop(BASELINE_KEY, None)


def quarter_label(period):
    return f"Q{period.quarter} {period.year}"


def change_label(value, reference, period):
    if reference == 0:
        return f"{value - reference:+,.1f} tons vs {quarter_label(period)}"
    return f"{(value / reference - 1) * 100:+.1f}% vs {quarter_label(period)}"


def main():
    st.set_page_config(
        page_title="Crop Production Outlook",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.html(APP_STYLE)

    for path in (MODEL_PATH, DATA_PATH):
        if not path.is_file():
            st.error(f"Place {path.name} in the same folder as this app.")
            st.stop()

    model_modified = MODEL_PATH.stat().st_mtime_ns
    data_modified = DATA_PATH.stat().st_mtime_ns
    try:
        model, feature_columns = load_model(str(MODEL_PATH), model_modified)
        history = load_history(str(DATA_PATH), data_modified, LAST_COMPLETE_QUARTER)
    except InconsistentVersionWarning as error:
        st.error(
            f"This model was saved with scikit-learn {error.original_sklearn_version}. "
            "Install the matching dependencies, then restart the app."
        )
        st.code("python -m pip install -r requirements.txt")
        st.stop()
    except Exception as error:
        st.error(f"Could not load the saved model or production history. {error}")
        st.stop()

    with st.sidebar:
        st.title("Crop outlook")
        st.caption("Philippine provincial crop planning")
        st.divider()
        st.subheader("Planning area")
        province = st.selectbox(
            "Province", sorted(history["province"].unique()), format_func=str.title, key="m3_province",
        )
    crops = history.loc[history["province"].eq(province), "crop"].unique()
    with st.sidebar:
        crop = st.selectbox("Crop", sorted(crops), key=f"m3_crop_{province}")
    actual = history.loc[
        history["province"].eq(province) & history["crop"].eq(crop)
    ].set_index("period")["production"].sort_index()
    origin = actual.index.max()
    forecast_quarter = origin + 1
    reported_production = float(actual.loc[origin])

    widget_key = f"m3_production_{province}_{crop}_{origin}_{data_modified}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = reported_production
    with st.sidebar:
        st.divider()
        st.subheader("Production scenario")
        st.caption(f"Latest observed quarter · {quarter_label(origin)}")
        current_production = st.number_input(
            "Production (metric tons)",
            min_value=0.0,
            step=0.1,
            format="%.2f",
            key=widget_key,
            help="Adjust this value to explore a production scenario. Earlier quarters use recorded data.",
        )
        st.caption(f"Recorded production is {reported_production:,.1f} metric tons.")
        calculate = st.button("Forecast", type="primary", width="stretch")
        show_baseline = st.button(
            "Show seasonal baseline", width="stretch",
            help="Use production from the forecast quarter in the previous year as a comparison.",
        )
        st.button(
            "Restore defaults", width="stretch",
            on_click=restore_defaults, args=(widget_key, reported_production),
        )

    features = build_features(province, crop, actual, origin, current_production)
    baseline_quarter = forecast_quarter - 4
    baseline_production = actual.get(baseline_quarter, np.nan)
    baseline_context = (data_modified, province, crop, str(forecast_quarter))
    if show_baseline:
        if pd.isna(baseline_production):
            st.warning(f"Production for {baseline_quarter} is unavailable for this seasonal baseline.")
        else:
            st.session_state[BASELINE_KEY] = baseline_context
    baseline = (
        float(baseline_production)
        if st.session_state.get(BASELINE_KEY) == baseline_context
        else None
    )
    scenario = (model_modified, data_modified, province, crop, str(origin), current_production)
    if calculate:
        st.session_state.pop(RESULT_KEY, None)
        try:
            prediction = float(model.predict(features[feature_columns])[0])
            if not np.isfinite(prediction):
                raise ValueError("The model returned a non-finite forecast.")
            st.session_state[RESULT_KEY] = {
                "scenario": scenario,
                "raw_prediction": prediction,
                "production": max(0.0, prediction),
            }
        except (ValueError, TypeError, FloatingPointError) as error:
            st.error(f"Could not calculate the forecast. {error}")

    result = st.session_state.get(RESULT_KEY)
    forecast = result["production"] if result and result["scenario"] == scenario else None

    with st.container(key="outlook_header"):
        heading, period = st.columns([3, 1])
        with heading:
            st.caption("PHILIPPINE AGRICULTURE")
            st.title("Crop production outlook")
            st.write(f"{crop} in {province.title()}")
        with period:
            st.caption("Forecast quarter")
            st.subheader(quarter_label(forecast_quarter))
            st.caption(f"Data snapshot through {quarter_label(origin)}")

    st.caption(
        "Explore a next-quarter production estimate for provincial crop planning. "
        "All production values are in metric tons."
    )
    with st.container(key="summary"):
        forecast_card, reported_card, baseline_card = st.columns([1.15, 1, 1])
        with forecast_card, st.container(border=True, key="forecast_card"):
            delta = change_label(forecast, reported_production, origin) if forecast is not None else None
            st.metric(
                "Forecast production", None if forecast is None else f"{forecast:,.1f}",
                delta=delta, delta_color="off",
                help="The model's next-quarter estimate, using the production scenario in the sidebar.",
            )
            st.caption(f"Metric tons for {quarter_label(forecast_quarter)}")
            if forecast is None:
                st.caption("Select Forecast in the sidebar to calculate.")
        with reported_card, st.container(border=True, key="reported_card"):
            st.metric("Latest reported production", f"{reported_production:,.1f}")
            st.caption(f"Metric tons recorded in {quarter_label(origin)}")
        with baseline_card, st.container(border=True, key="baseline_card"):
            st.metric(
                "Seasonal baseline", None if baseline is None else f"{baseline:,.1f}",
                help="Repeats reported production from the same calendar quarter last year.",
            )
            if baseline is None:
                st.caption("Select Show seasonal baseline to compare.")
            else:
                st.caption(f"Metric tons from {quarter_label(baseline_quarter)}")
                if forecast is not None:
                    st.caption(f"Model forecast {change_label(forecast, baseline, baseline_quarter)}")

    if current_production != reported_production:
        st.info(
            f"Adjusted scenario. The forecast uses {current_production:,.1f} metric tons "
            f"for {quarter_label(origin)}. The reported value is {reported_production:,.1f}."
        )
    if forecast is not None and result["raw_prediction"] < 0:
        st.warning(
            f"Forecast requires review. The model returned {result['raw_prediction']:,.1f} metric tons. "
            "The displayed estimate is zero because production cannot be negative."
        )

    with st.container(border=True, key="trend_panel"):
        trend_heading, trend_control = st.columns([2, 1])
        with trend_heading:
            st.subheader("Production trend")
            st.caption(f"Reported production and estimates for {quarter_label(forecast_quarter)}")
        with trend_control:
            chart_range = st.selectbox(
                "History shown", ["Last 3 years", "Last 5 years", "All history"], index=1,
                key="m3_chart_range",
            )
        start = actual.index.min()
        if chart_range != "All history":
            count = 12 if chart_range == "Last 3 years" else 20
            start = max(start, origin - count + 1)
        quarters = pd.period_range(start, origin, freq="Q-DEC")
        series = actual.reindex(quarters)
        fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="white")
        ax.set_facecolor("white")
        ax.axvspan(forecast_quarter.start_time, forecast_quarter.end_time, color="#f1f5ef", zorder=0)
        ax.plot(quarters.to_timestamp(), series, color="#385f6b", linewidth=2, label="Reported production")
        ax.scatter([origin.start_time], [reported_production], color="#385f6b", s=28, zorder=3)
        if forecast is not None:
            ax.plot(
                [origin.start_time, forecast_quarter.start_time], [current_production, forecast],
                "o--", color="#bd7523", linewidth=2, label="Model forecast",
            )
        if baseline is not None:
            ax.plot(
                [origin.start_time, forecast_quarter.start_time], [reported_production, baseline],
                "s:", color="#34815d", linewidth=2, label="Seasonal baseline",
            )
        ax.set_ylabel("Metric tons", color="#52665a")
        ax.set_ylim(bottom=0)
        ax.set_xlim(quarters[0].start_time, forecast_quarter.end_time)
        ax.xaxis.set_major_locator(mdates.YearLocator(2 if chart_range == "All history" else 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.tick_params(axis="both", colors="#65766b", labelsize=10, length=0, pad=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#dce5dc")
        ax.grid(axis="y", color="#e8ede6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"The shaded area marks {quarter_label(forecast_quarter)}, the forecast period.")

    with st.expander("Production inputs and scenario assumptions"):
        roles = ["Previous quarter", "Two quarters earlier", "Same quarter last year", "Same quarter two years earlier"]
        input_table = pd.DataFrame({
            "Production reference": ["Latest quarter (scenario)"] + roles,
            "Quarter": [quarter_label(origin)] + [quarter_label(origin - lag) for lag in PRODUCTION_LAGS],
            "Metric tons": [current_production] + [actual.get(origin - lag, np.nan) for lag in PRODUCTION_LAGS],
        })
        st.dataframe(input_table, hide_index=True, width="stretch")
        st.caption(
            f"The four-quarter average is {features.at[0, 'production_roll4_mean']:,.1f} metric tons. "
            "It includes the scenario value and the three preceding recorded quarters."
        )
    with st.expander("Method and data coverage"):
        st.write(
            "The Linear Regression model uses current, recent, and seasonal production, "
            "a four-quarter production average, province, crop, and forecast quarter. "
            "These inputs form the M3 Full Production feature set."
        )
        st.write(
            "The seasonal naïve baseline repeats production from the same quarter last year. "
            "It uses recorded history for the selected crop and province."
        )
        st.caption(
            f"Production history spans {quarter_label(history['period'].min())} through "
            f"{quarter_label(history['period'].max())}. Partial 2025 Q4 records are excluded. "
            "This view provides a planning estimate for the displayed data snapshot."
        )
        st.dataframe(features.T.astype(str).rename(columns={0: "Model input"}), width="stretch")


if __name__ == "__main__":
    main()
