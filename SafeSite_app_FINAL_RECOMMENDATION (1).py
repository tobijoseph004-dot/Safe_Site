import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(
    page_title="SafeSite",
    page_icon="🏗️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

# Change this to the exact name of your CSV if you renamed it.
DATASET_FILENAME = "OSHA_Data.csv"

MODEL_FILES = {
    "model": "model.pkl",
    "encoders": "encoders.pkl",
    "dropdown_options": "dropdown_options.pkl",
    "context_map": "context_map.pkl",
    "task_map": "task_map.pkl",
}


@st.cache_resource
def load_pickle(filename):
    path = BASE_DIR / filename
    if not path.exists():
        st.error(f"Missing file: `{filename}`. Put it in the same folder as this app.py.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_dataset():
    path = BASE_DIR / DATASET_FILENAME
    if not path.exists():
        for name in [
            "OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv",
            "osha_construction.csv",
            "construction_incidents.csv",
            "construction_data.csv",
        ]:
            candidate = BASE_DIR / name
            if candidate.exists():
                path = candidate
                break
    if not path.exists():
        st.error(
            f"Dataset not found. The app is looking for `{DATASET_FILENAME}`. "
            "Change DATASET_FILENAME near the top of this file to your exact CSV filename."
        )
        st.stop()
    return pd.read_csv(path)


model = load_pickle(MODEL_FILES["model"])
encoders = load_pickle(MODEL_FILES["encoders"])
dropdown_options = load_pickle(MODEL_FILES["dropdown_options"])
context_map = load_pickle(MODEL_FILES["context_map"])
task_map = load_pickle(MODEL_FILES["task_map"])
df = load_dataset()


def remove_other(series):
    s = series.dropna().astype(str).str.strip()
    return s[~s.str.lower().isin(
        ["other", "others", "other/unknown", "other / unknown"]
    )]


def find_year_range(data):
    preferred = ["year", "incident year", "date", "incident date", "event date"]
    lower = {str(c).strip().lower(): c for c in data.columns}

    for name in preferred:
        if name in lower:
            col = lower[name]
            parsed = pd.to_datetime(data[col], errors="coerce")
            years = parsed.dt.year.dropna()
            if len(years):
                return int(years.min()), int(years.max())

    for col in data.columns:
        name = str(col).lower()
        if "year" in name or "date" in name:
            parsed = pd.to_datetime(data[col], errors="coerce")
            years = parsed.dt.year.dropna()
            if len(years):
                return int(years.min()), int(years.max())
    return None


year_range = find_year_range(df)
year_text = f"{year_range[0]}–{year_range[1]}" if year_range else "the available historical records"


def explain_result(prediction, fatal_prob, nonfatal_prob):
    if prediction == "Fatal":
        return (
            f"The model estimates that a fatal outcome is more likely for the "
            f"incident details entered: {fatal_prob * 100:.1f}% fatal versus "
            f"{nonfatal_prob * 100:.1f}% non-fatal."
        )
    return (
        f"The model estimates that a non-fatal outcome is more likely for the "
        f"incident details entered: {nonfatal_prob * 100:.1f}% non-fatal versus "
        f"{fatal_prob * 100:.1f}% fatal."
    )


def make_recommendation(event_type, env_factor, human_factor, risk_level):
    event = str(event_type).lower()
    env = str(env_factor).lower()
    human = str(human_factor).lower()
    recs = []

    if any(x in env for x in ["gas", "vapor", "vapour", "fume", "smoke", "dust", "mist"]):
        recs.append("Review ventilation, exposure controls and appropriate respiratory protection.")
    if any(x in env for x in ["fall", "height", "elevation", "roof", "scaffold"]):
        recs.append("Review fall-prevention and fall-protection controls for the task.")
    if any(x in env for x in ["electric", "electrical", "current", "power"]):
        recs.append("Review electrical isolation, equipment condition and applicable safe-work procedures.")
    if any(x in env for x in ["vehicle", "mobile", "traffic", "moving equipment"]):
        recs.append("Review vehicle/equipment movement controls, worker positioning and separation from moving equipment.")
    if any(x in env for x in ["pinch", "caught", "between", "machine", "machinery"]):
        recs.append("Review machine guarding, isolation procedures and controls around pinch or caught-between points.")
    if any(x in env for x in ["chemical", "corrosive", "toxic"]):
        recs.append("Review chemical handling, exposure controls, storage and required protective equipment.")

    if any(x in human for x in ["respiratory", "respiration", "breathing"]):
        recs.append("Confirm that suitable respiratory protection is available, correctly selected and consistently used.")
    if any(x in human for x in ["training", "trained", "knowledge", "instruction"]):
        recs.append("Review task-specific training and confirm that workers understand the required safe procedure.")
    if any(x in human for x in ["housekeeping", "clean"]):
        recs.append("Strengthen housekeeping checks and remove hazards that could contribute to the incident.")
    if any(x in human for x in ["exposure", "monitoring"]):
        recs.append("Review exposure monitoring and make sure controls are adequate for the identified hazard.")

    if "fall" in event:
        recs.append("Review task setup, access arrangements and controls for falls.")
    if any(x in event for x in ["inhal", "respir", "absorption"]):
        recs.append("Review the source of exposure and the controls used to prevent worker contact with the hazard.")
    if any(x in event for x in ["shock", "electrical"]):
        recs.append("Review electrical hazard controls and isolation procedures relevant to the task.")
    if "burn" in event:
        recs.append("Review controls for heat, flame, hot surfaces or other sources of burns.")
    if any(x in event for x in ["struck", "caught"]):
        recs.append("Review worker positioning, exclusion zones and controls for moving or falling objects.")

    unique = []
    for r in recs:
        if r not in unique:
            unique.append(r)

    if risk_level == "High Risk":
        priority = (
            "Prioritise this incident profile for immediate HSE review. Focus on the identified "
            "hazards and strengthen the relevant controls before the task continues."
        )
    elif risk_level == "Medium Risk":
        priority = (
            "Review the identified hazards and confirm that the required safety controls are "
            "in place and being followed before the task continues."
        )
    else:
        priority = (
            "Maintain the required safety controls and continue routine monitoring of the "
            "identified hazards. A lower risk assessment does not mean the hazard can be ignored."
        )

    return [priority] + unique[:4]


st.markdown("""
<style>
.block-container {max-width: 1350px; padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 1.6rem 1.8rem; border-radius: 16px; background: linear-gradient(135deg,#17324d,#285b80); color:white; margin-bottom:1.3rem;}
.hero h1 {margin:0 0 .35rem 0;}
.hero p {margin:0; opacity:.92;}
.recommendation-box {padding:.9rem 1rem; border-left:4px solid #2f855a; background:#f4f8f6; border-radius:7px; margin-bottom:.65rem;}
</style>
""", unsafe_allow_html=True)

st.markdown(
    f'<div class="hero"><h1>🏗️ SafeSite</h1>'
    f'<p><strong>Construction Incident Risk Assessment</strong><br>'
    f'Use previous construction incident records to assess a new incident and estimate '
    f'whether it is more likely to result in a fatal or non-fatal outcome. '
    f'The results can help identify situations that may need stronger safety measures '
    f'and preventive action ({year_text}).</p></div>',
    unsafe_allow_html=True,
)

st.sidebar.title("🏗️ SafeSite")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Incident Insights", "🔎 Assess an Incident"],
)
st.sidebar.divider()
st.sidebar.metric("Historical incidents", f"{len(df):,}")
st.sidebar.caption(f"Historical period: {year_text}")


if page == "🏠 Overview":
    st.subheader("Construction Safety Overview")
    st.write(
        "SafeSite uses previous construction incident records to help estimate the "
        "likely outcome of a new incident. The aim is to support earlier safety "
        "action and help reduce serious workplace injuries."
    )

    total = len(df)
    fatal = int((df["Degree of Injury"] == "Fatal").sum())
    nonfatal = int((df["Degree of Injury"] == "Nonfatal").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total incidents", f"{total:,}")
    c2.metric("Fatal incidents", f"{fatal:,}")
    c3.metric("Non-fatal incidents", f"{nonfatal:,}")
    c4.metric("Fatal share", f"{fatal / total * 100:.1f}%" if total else "0.0%")

    st.divider()
    st.markdown("### What does the historical data show?")
    if fatal > nonfatal:
        st.write(
            f"Across {year_text}, the dataset contains more fatal incidents "
            f"({fatal:,}) than non-fatal incidents ({nonfatal:,})."
        )
    else:
        st.write(
            f"Across {year_text}, the dataset contains more non-fatal incidents "
            f"({nonfatal:,}) than fatal incidents ({fatal:,})."
        )

    left, right = st.columns(2)
    with left:
        st.markdown("### Most common incident types")
        st.bar_chart(remove_other(df["Event type"]).value_counts().head(10).rename("Incidents"))
    with right:
        st.markdown("### Fatal vs non-fatal incidents")
        outcomes = (
            df["Degree of Injury"].value_counts()
            .reindex(["Fatal", "Nonfatal"]).fillna(0).astype(int).rename("Incidents")
        )
        st.bar_chart(outcomes)

    st.info(
        "The Assess an Incident section uses the trained classification model to "
        "estimate whether a supplied incident profile is more likely to be fatal or non-fatal."
    )


elif page == "📊 Incident Insights":
    st.subheader("📊 Incident Insights")
    st.caption(
        "Broad 'Other' categories are excluded from the factor charts because "
        "they do not provide a specific HSE insight."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Incidents", f"{len(df):,}")
    c2.metric("Fatal", f"{(df['Degree of Injury'] == 'Fatal').sum():,}")
    c3.metric("Non-fatal", f"{(df['Degree of Injury'] == 'Nonfatal').sum():,}")

    st.divider()
    st.markdown("### Most common incident types")
    st.bar_chart(remove_other(df["Event type"]).value_counts().head(10).rename("Incidents"))

    st.divider()
    st.markdown("### Fatal share by incident type")

    temp = df.dropna(subset=["Event type"]).copy()
    temp["event_clean"] = temp["Event type"].astype(str).str.strip()
    temp = temp[~temp["event_clean"].str.lower().isin(
        ["other", "others", "other/unknown", "other / unknown"]
    )]

    grouped = temp.groupby("event_clean").agg(
        incidents=("Degree of Injury", "size"),
        fatal=("Degree of Injury", lambda x: (x == "Fatal").sum()),
    )
    grouped["fatal_share"] = grouped["fatal"] / grouped["incidents"]
    grouped = grouped.sort_values("fatal_share", ascending=False).head(10)

    table = grouped[["incidents", "fatal_share"]].copy()
    table["fatal_share"] = table["fatal_share"].map(lambda x: f"{x:.1%}")
    table.columns = ["Incidents", "Fatal share"]
    st.dataframe(table, use_container_width=True)

    st.divider()
    st.markdown("### Most common specific environmental factors")
    env = remove_other(df["Environmental Factor"])
    if len(env):
        st.bar_chart(env.value_counts().head(10).rename("Incidents"))
    else:
        st.info("No specific environmental-factor categories are available.")

    st.markdown("### Most common specific human factors")
    human = remove_other(df["Human Factor"])
    if len(human):
        st.bar_chart(human.value_counts().head(10).rename("Incidents"))
    else:
        st.info("No specific human-factor categories are available.")


else:
    st.subheader("🔎 Assess an Incident")
    st.caption(
        "Enter the characteristics of the incident. Environmental and human-factor "
        "choices are context-sensitive to the selected event type."
    )

    st.markdown("### Step 1 — What happened?")
    event_type = st.selectbox("Select the type of incident", dropdown_options["Event type"])

    st.markdown("### Step 2 — What was the surrounding condition?")
    env_factor = st.selectbox(
        "Select the environmental factor",
        context_map[event_type]["environmental_factors"],
    )

    st.markdown("### Step 3 — What human factor was involved?")
    human_factor = st.selectbox(
        "Select the human factor",
        context_map[event_type]["human_factors"],
    )

    st.markdown("### Step 4 — Was the worker regularly assigned to this task?")
    task_assigned = st.radio(
        "Was the worker regularly assigned to this task?",
        ["Regularly Assigned", "Not Regularly Assigned"],
        horizontal=True,
    )

    st.divider()

    if st.button("🔮 Check Outcome", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "Event type": event_type,
            "Environmental Factor": env_factor,
            "Human Factor": human_factor,
            "Task Assigned": task_assigned,
        }])

        unknown = []
        for col in input_data.columns:
            le = encoders[col]
            value = input_data[col].iloc[0]
            if value not in le.classes_:
                unknown.append((col, value))
            else:
                input_data[col] = le.transform(input_data[col])

        if unknown:
            st.error("A selected value is not recognised by the trained model.")
            for col, value in unknown:
                st.write(f"**{col}:** {value}")
            st.stop()

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        prob_dict = dict(zip(model.classes_, probabilities))

        fatal_prob = float(prob_dict.get("Fatal", 0))
        nonfatal_prob = float(prob_dict.get("Nonfatal", 0))
        top_prob = max(prob_dict.values())

        # Risk bands are applied to the model's estimated fatal-outcome probability.
        # The underlying trained model remains the existing Fatal/Nonfatal classifier.
        risk_score = fatal_prob
        if risk_score >= 0.70:
            risk_level = "High Risk"
            risk_icon = "🔴"
        elif risk_score >= 0.40:
            risk_level = "Medium Risk"
            risk_icon = "🟠"
        else:
            risk_level = "Low Risk"
            risk_icon = "🟢"

        st.markdown("## Risk Assessment")
        st.markdown(f"### {risk_icon} {risk_level}")
        st.progress(risk_score)
        st.caption(f"Risk score: {risk_score * 100:.1f}%")

        if top_prob < 0.60:
            st.warning("The model shows a less clear separation between the two historical outcomes. Use this result together with professional HSE judgement and site information.")
        elif top_prob < 0.75:
            st.info("The model shows a moderate preference for one historical outcome. Use the result together with professional HSE judgement and site information.")
        else:
            st.success("The model shows a stronger preference for the estimated historical outcome.")

        st.divider()
        st.markdown("### 🔎 Assessment Summary")
        summary_items = [
            ("Incident type", event_type),
            ("Environmental factor", env_factor),
            ("Human factor", human_factor),
            ("Task assignment", task_assigned),
            ("Risk level", f"{risk_icon} {risk_level}"),
        ]
        for label, value in summary_items:
            st.markdown(f"- **{label}:** {value}")

        st.divider()
        st.markdown("### 🔎 What the result shows")
        st.write(
            f"The selected incident profile has been assessed as **{risk_level.lower()}** "
            f"based on patterns in the historical construction incident records."
        )

        st.markdown("### 📌 What does this risk level mean?")
        if risk_level == "High Risk":
            st.write(
                "This result indicates a higher level of concern for the selected incident profile. "
                "The HSE officer should prioritise the identified hazards and review the relevant "
                "safety controls before the task continues."
            )
        elif risk_level == "Medium Risk":
            st.write(
                "This result indicates a moderate level of concern for the selected incident profile. "
                "The HSE officer should review the identified hazards and confirm that the relevant "
                "safety controls are in place."
            )
        else:
            st.write(
                "This result indicates a lower level of concern for the selected incident profile. "
                "Normal safety procedures should still be maintained and the identified hazards "
                "should continue to be monitored."
            )

        st.markdown("### 📊 Previous incidents of this type")
        history = df[df["Event type"].astype(str) == str(event_type)]
        event_total = len(history)
        event_fatal = int((history["Degree of Injury"] == "Fatal").sum())

        h1, h2, h3 = st.columns(3)
        h1.metric("Historical incidents for this event", f"{event_total:,}")
        h2.metric("Fatal incidents", f"{event_fatal:,}")
        h3.metric(
            "Historical fatal share",
            f"{event_fatal / event_total * 100:.1f}%" if event_total else "0.0%",
        )
        st.caption(
            f"Previous incidents of this type from {year_text}. These figures describe the selected "
            "event type and are separate from the model probability."
        )

        st.divider()
        st.markdown("### 🛡️ Recommended Safety Action")

        recommendations = make_recommendation(
            event_type, env_factor, human_factor, risk_level
        )

        # Always show the recommendations as normal Streamlit text.
        # This avoids HTML/CSS rendering problems and ensures the section
        # cannot accidentally display Python code in the app.
        if recommendations:
            for recommendation in recommendations:
                st.markdown(f"- {recommendation}")
        else:
            st.info(
                "Review the identified environmental and human factors and "
                "confirm that appropriate safety controls are in place."
            )



st.divider()
st.caption(
    f"SafeSite | Construction incident risk assessment | "
    f"Historical construction incident data: {year_text}"
)
