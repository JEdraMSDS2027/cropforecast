from pathlib import Path
import warnings

import joblib
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


def main():
    st.set_page_config(page_title="Philippine Crop Forecast", layout="wide")
    st.title("Next-quarter crop production forecast")
    st.write(
        "Choose a province and crop to forecast next-quarter production in metric tons. "
        "The M3 model uses current, recent, and seasonal production, plus a four-quarter average."
    )

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
        st.code("python -m pip install -r requirements_linear_m3.txt")
        st.stop()
    except Exception as error:
        st.error(f"Could not load the saved model or production history. {error}")
        st.stop()

    st.caption(f"Saved model: {type(model.steps[-1][1]).__name__} · Feature set: M3 Full Production")
    province = st.selectbox("Province", sorted(history["province"].unique()), format_func=str.title)
    crops = history.loc[history["province"].eq(province), "crop"].unique()
    crop = st.selectbox("Crop", sorted(crops), key=f"m3_crop_{province}")
    actual = history.loc[
        history["province"].eq(province) & history["crop"].eq(crop)
    ].set_index("period")["production"].sort_index()
    origin = actual.index.max()
    forecast_quarter = origin + 1
    reported_production = float(actual.loc[origin])

    st.subheader(f"Forecast {forecast_quarter} using observations through {origin}")
    st.caption(
        f"The latest complete quarter in the source data is {LAST_COMPLETE_QUARTER}. "
        "The partial 2025Q4 records are excluded from the production history."
    )
    st.subheader(f"Production simulator for {origin}")
    st.caption("Edit the latest quarter's production, then click Forecast. Earlier quarters use recorded values.")
    widget_key = f"m3_production_{province}_{crop}_{origin}_{data_modified}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = reported_production
    current_production = st.number_input(
        f"Production in {origin} (metric tons)",
        min_value=0.0,
        step=0.1,
        format="%.2f",
        key=widget_key,
    )
    features = build_features(province, crop, actual, origin, current_production)
    with st.expander("Fixed historical production inputs"):
        fixed_inputs = pd.DataFrame({
            "Feature": [f"production_lag{lag}" for lag in PRODUCTION_LAGS],
            "Quarter": [str(origin - lag) for lag in PRODUCTION_LAGS],
            "Recorded production (metric tons)": [
                actual.get(origin - lag, np.nan) for lag in PRODUCTION_LAGS
            ],
        })
        st.dataframe(fixed_inputs, hide_index=True)
        st.caption("Seasonal lags 3 and 7 match the forecast quarter one and two years earlier.")
    with st.expander("M3 inputs passed to the saved pipeline"):
        st.dataframe(features.T.astype(str).rename(columns={0: "Value"}))
        st.caption("The four-quarter average updates when the simulator value changes.")

    forecast_button, baseline_button, reset_button = st.columns(3)
    calculate = forecast_button.button("Forecast", type="primary")
    show_baseline = baseline_button.button("Show seasonal baseline")
    reset_button.button(
        "Restore defaults",
        on_click=restore_defaults,
        args=(widget_key, reported_production),
    )
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
    model_display, baseline_display = st.columns(2)
    with model_display:
        if forecast is None:
            st.caption(f"Click Forecast to calculate production for {forecast_quarter}.")
        else:
            st.metric(f"{crop} forecast for {forecast_quarter}", f"{forecast:,.1f} metric tons")
            if result["raw_prediction"] < 0:
                st.warning(
                    f"The model returned {result['raw_prediction']:,.1f} metric tons. "
                    "The displayed forecast is set to zero because production cannot be negative."
                )
    with baseline_display:
        if baseline is not None:
            st.metric(f"Seasonal naïve baseline for {forecast_quarter}", f"{baseline:,.1f} metric tons")
            st.caption(f"Uses reported production from {baseline_quarter}, the same quarter last year.")

    quarters = pd.period_range(actual.index.min(), origin, freq="Q-DEC")
    series = actual.reindex(quarters)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(quarters.to_timestamp(), series, label="Reported production")
    if forecast is not None:
        ax.plot(
            [origin.to_timestamp(), forecast_quarter.to_timestamp()],
            [current_production, forecast],
            "o--", color="darkorange", label="Scenario forecast",
        )
    if baseline is not None:
        ax.plot(
            [origin.to_timestamp(), forecast_quarter.to_timestamp()],
            [reported_production, baseline],
            "s:", color="seagreen", label="Seasonal naïve baseline",
        )
    ax.set_title(f"{crop} production in {province.title()}")
    ax.set_ylabel("Production (metric tons)")
    ax.set_xlabel("Year")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
