import streamlit as st
from dataclasses import dataclass
import pandas as pd
import openai
import json

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@dataclass
class RiskInput:
    name: str
    severity: int
    relevance: int
    directionality: int
    likelihood: int
    category: str

    def weighted_score(self) -> int:
        return self.severity + self.relevance * 2 + self.directionality + self.likelihood


def gpt_extract_risks(scenario_text):
    prompt = f"""
    You are a risk analyst AI. Given the following scenario, return a list of risks. For each risk, map it to one of the following higher-level risk categories, and estimate its severity (0-2), relevance (0-2), directionality (0-2), and likelihood (0-2). Use whole numbers only.

    Risk Categories:
    1. Threat Environment (e.g., Critical Incident, Sustained Civil Unrest, Anti-American Sentiment, Status of Government, History of Resolution, Actions Taken by Local Government, Key Populations Being Targeted, Police/Military Presence, Observance of Lawlessness, Likelihood of Regional Conflict Spillover, Other Assistance Companies Issuing Warnings, Other Higher Ed Clients Discussing Evacuation, Closure of Educational Institutions)

    2. Operational Disruption (e.g., Impact Considerations, Location Considerations, Immediacy Considerations, Event Lead Time, Road Closures, Curfews, Disruptions to Mobile Voice/SMS/Data Services, Observance of Power Outages, Access to Fuel, Access to Food and Clean Water, Transportation Infrastructure, Airlines Limiting or Canceling Flights)

    3. Health & Medical Risk (e.g., Severity of Health Situation [Self or Official Report], Crisis24 Medical Assessment, Deviation from Baseline Medical History, Availability of Medical/Mental Health Treatment, Critical Medication Supply, Need for a Medical Escort, Strain on Local Medical Resources, Increased Transmission of Communicable Diseases, Access to MedEvac, Health Infrastructure Strain)

    4. Client Profile & Exposure (e.g., Undergraduate/Graduate/Staff, Supervision/Organizational Support, Program Type, Group Size, Field Site or Urban Environment, How Far Must Commute to Necessities, Housing/Shelter Security, When Travelers Intend to Leave, Airport Type, Access to Intelligence or Info Sharing, Safe Havens or Alternatives)

    5. Geo-Political & Intelligence Assessment (e.g., Severity of Crisis24 Alerts, Preexisting Crisis24 Location Intelligence Rating, Dynamic Risk Library Assessment, US State Department Travel Advisory, FCDO Travel Warning, Australia Smarttraveller Warning, Relative Concern of Crisis24 Personnel, Crisis24 Life Safety Assessment, CAT [Crisis Advisory Team] Activation, Organizational Risk Appetite, Existing Mitigations/Security Protocols)

    6. Infrastructure & Resource Stability (e.g., Environmental and Weather Risk, Changes in Local Climate, Disruptions to Communication, Internet Infrastructure, Power Grid Stability, Medical System Burden, Communications Breakdown, Relative Capabilities of Assistance Company)

    Use the following logic when assigning **Likelihood**:
    - 0 (Unlikely): speculative, rare, uncertain
    - 1 (Possible): monitoring, could/may/might
    - 2 (Likely): active, confirmed, ongoing

    Give a **Likelihood of 2** if actively occurring or confirmed.
    Avoid downgrading ongoing confirmed events.

    If resolved, set directionality=0 and adjust severity/likelihood.

    Group same root-cause impacts only if same mode, area, timeframe.
    However, **preserve separate risk entries** when effects differ in nature, geography, or timing.

    Return only a valid JSON array:
    [
      {{
        "name": "...",
        "category": "...",
        "severity": 0,
        "directionality": 0,
        "likelihood": 0,
        "relevance": 0
      }}
    ]

    Do not include any text before/after the JSON.

    Scenario:
    {scenario_text}
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        st.error("Failed to parse GPT response.")
        st.code(content, language="json")
        return []
    return [RiskInput(**e) for e in parsed if isinstance(e, dict)]


def calculate_risk_summary(inputs):
    rows, total = [], 0
    for r in inputs:
        score = r.weighted_score()
        total += score
        rows.append({
            "Scenario": r.name,
            "Risk Category": r.category,
            "Severity": r.severity,
            "Directionality": r.directionality,
            "Likelihood": r.likelihood,
            "Relevance": r.relevance,
            "Weighted Score": score
        })
    df = pd.DataFrame(rows)
    max_score = len(inputs) * 10
    norm_score = int(round((total / max_score) * 10)) if max_score else 0
    uniq = len(set(r.category for r in inputs))
    bonus = min(round(((len(inputs) - uniq) / len(inputs)) * 2), 2) if inputs else 0
    final = min(norm_score + bonus, 10)
    return df, total, final


def advice_matrix(score: int, tolerance: str):
    if score <= 3:
        risk_level = "Low"
    elif score <= 6:
        risk_level = "Moderate"
    elif score <= 8:
        risk_level = "High"
    else:
        risk_level = "Extreme"
    if tolerance == "Low":
        advice = (
            "Crisis24 Proactive Engagement" if risk_level in ["High", "Extreme"] else
            "Heightened Vigilance" if risk_level == "Moderate" else
            "Normal Precautions"
        )
    elif tolerance == "Moderate":
        advice = (
            "Consider Crisis24 Consultation" if risk_level == "Extreme" else
            "Heightened Vigilance" if risk_level == "High" else
            "Normal Precautions"
        )
    else:
        advice = "Heightened Vigilance" if risk_level == "Extreme" else "Normal Precautions"
    return risk_level, advice


st.set_page_config(layout="wide")
st.title("AI-Assisted Risk Model & Advice Matrix")

scenario = st.text_area("Enter Threat Scenario")
tolerance = st.selectbox("Select Client Risk Tolerance", ["Low", "Moderate", "High"], index=1)

if st.button("Analyze Scenario"):
    with st.spinner("Analyzing..."):
        st.session_state.risks = gpt_extract_risks(scenario)
        st.session_state.show_editor = True

if "risks" not in st.session_state:
    st.session_state.risks = []

if st.session_state.get("show_editor") and st.session_state.risks:
    risks = st.session_state.risks
    categories = [
        "Threat Environment",
        "Operational Disruption",
        "Health & Medical Risk",
        "Client Profile & Exposure",
        "Geo-Political & Intelligence Assessment",
        "Infrastructure & Resource Stability"
    ]
    st.subheader("Mapped Risks and Scores")
    for i, r in enumerate(risks):
        cols = st.columns([2, 2, 1, 1, 1, 1, 0.5])
        name = cols[0].text_input("Scenario", value=r.name, key=f"name_{i}")
        category = cols[1].selectbox("Risk Category", categories, index=categories.index(r.category), key=f"cat_{i}")
        severity = cols[2].selectbox("Severity", [0, 1, 2], index=r.severity, key=f"sev_{i}")
        directionality = cols[3].selectbox("Directionality", [0, 1, 2], index=r.directionality, key=f"dir_{i}")
        likelihood = cols[4].selectbox("Likelihood", [0, 1, 2], index=r.likelihood, key=f"like_{i}")
        relevance = cols[5].selectbox("Relevance", [0, 1, 2], index=r.relevance, key=f"rel_{i}")
        if cols[6].button("🗑️", key=f"del_{i}"):
            risks.pop(i)
            break
        else:
            risks[i] = RiskInput(name, severity, relevance, directionality, likelihood, category)

    if "new_count" not in st.session_state:
        st.session_state.new_count = 0

    st.markdown("---")
    for j in range(st.session_state.new_count):
        cols = st.columns([2, 2, 1, 1, 1, 1, 0.5])
        name = cols[0].text_input("Scenario", key=f"name_new_{j}")
        category = cols[1].selectbox("Risk Category", categories, key=f"cat_new_{j}")
        severity = cols[2].selectbox("Severity", [0, 1, 2], key=f"sev_new_{j}")
        directionality = cols[3].selectbox("Directionality", [0, 1, 2], key=f"dir_new_{j}")
        likelihood = cols[4].selectbox("Likelihood", [0, 1, 2], key=f"like_new_{j}")
        relevance = cols[5].selectbox("Relevance", [0, 1, 2], key=f"rel_new_{j}")
        delete_new = cols[6].button("🗑️", key=f"del_new_{j}")
        if not delete_new and name:
            risks.append(RiskInput(name, severity, relevance, directionality, likelihood, category))

        # Button to add a new blank row
    col_add, _ = st.columns([1, 5])
    with col_add:
        if st.button("➕ Add row", key="add_row_new"):
            st.session_state.new_count = st.session_state.get("new_count", 0) + 1
            st.experimental_rerun()

    # Summary and advice output
    df_summary, total_score, final_score = calculate_risk_summary(risks)
    risk_level, guidance = advice_matrix(final_score, tolerance)

    st.markdown(f"**Aggregated Risk Score:** {total_score}")
    st.markdown(f"**Assessed Risk Score (1–10):** {final_score}")
    st.markdown(f"**Risk Level:** {risk_level}")
    st.markdown(f"**Advice for {tolerance} Tolerance:** {guidance}")
