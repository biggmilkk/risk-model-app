import streamlit as st
from dataclasses import dataclass
import pandas as pd
import openai
from collections import Counter
import json
from uuid import uuid4

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@dataclass
class RiskInput:
    name: str
    severity: int
    relevance: int
    likelihood: int
    category: str

    def weighted_score(self) -> int:
        return (self.severity * 1) + (self.relevance * 2) + (self.likelihood * 1)

def gpt_extract_risks(scenario_text):
    with open("gpt_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(scenario_text=scenario_text)

    with st.spinner("Analyzing..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)
    except Exception as e:
        st.error("Failed to parse GPT response.")
        st.code(content, language="json")
        return []

    risks = []
    for entry in parsed:
        try:
            entry.pop("directionality", None)
            risks.append(RiskInput(**entry))
        except Exception as e:
            st.warning(f"Failed to convert entry: {entry}")
            st.exception(e)

    if not risks:
        st.warning("Parsed successfully, but no valid risks were returned.")

    return risks

def calculate_risk_summary(inputs, alert_severity_level=None):
    rows = []
    total_score = 0
    for risk in inputs:
        score = risk.weighted_score()
        total_score += score
        rows.append({
            "Scenario": risk.name,
            "Risk Category": risk.category,
            "Severity": risk.severity,
            "Likelihood": risk.likelihood,
            "Relevance": risk.relevance,
            "Weighted Score": score
        })

    df = pd.DataFrame(rows)
    max_possible_score = len(inputs) * 10
    normalized_score = int(round((total_score / max_possible_score) * 10)) if max_possible_score > 0 else 0

    high_risks = [r for r in inputs if r.weighted_score() >= 6]
    low_risks = [r for r in inputs if 3 <= r.weighted_score() < 6]

    high_categories = [r.category for r in high_risks]
    low_category_counts = Counter([r.category for r in low_risks])
    low_clustered_categories = [cat for cat, count in low_category_counts.items() if count >= 3]

    cluster_categories = high_categories + low_clustered_categories
    total_cluster_risks = len(cluster_categories)
    unique_cluster_categories = len(set(cluster_categories))

    if total_cluster_risks > 0:
        clustering_ratio = (total_cluster_risks - unique_cluster_categories) / total_cluster_risks
        cluster_bonus = min(round(clustering_ratio * 2), 2)
    else:
        cluster_bonus = 0

    severity_bonus_map = {
        "Informational": 0,
        "Caution": 0,
        "Warning": 1,
        "Critical": 2
    }
    severity_bonus = severity_bonus_map.get(alert_severity_level, 0) if alert_severity_level else 0

    final_score = min(normalized_score + cluster_bonus + severity_bonus, 10)
    return df, total_score, final_score, severity_bonus

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
        if risk_level in ["High", "Extreme"]:
            advice = "Crisis24 Proactive Engagement"
        elif risk_level == "Moderate":
            advice = "Heightened Vigilance"
        else:
            advice = "Normal Precautions"
    elif tolerance == "Moderate":
        if risk_level == "Extreme":
            advice = "Consider Crisis24 Consultation"
        elif risk_level == "High":
            advice = "Heightened Vigilance"
        else:
            advice = "Normal Precautions"
    elif tolerance == "High":
        if risk_level == "Extreme":
            advice = "Heightened Vigilance"
        else:
            advice = "Normal Precautions"
    else:
        advice = "No Guidance Available"

    return risk_level, advice

st.set_page_config(layout="wide")

if "existing_risks" not in st.session_state:
    st.session_state.existing_risks = []
if "new_risks" not in st.session_state:
    st.session_state.new_risks = []
if "deleted_existing" not in st.session_state:
    st.session_state.deleted_existing = set()
if "alert_severity_used" not in st.session_state:
    st.session_state.alert_severity_used = ""

st.title("🧠 AI-Assisted Risk Model & Advice Matrix")
st.markdown("---")

st.subheader("Step 1: Enter Scenario")
scenario = st.text_area("Paste threat scenario or alert text here", height=200)

col1, col2 = st.columns([3, 1])
with col1:
    tolerance = st.selectbox("Select Client Risk Tolerance", ["Low", "Moderate", "High"], index=1)
with col2:
    alert_severity = st.selectbox("Alert Severity Level (if applicable)", ["", "Informational", "Caution", "Warning", "Critical"])

if st.button("Analyze Scenario") and scenario:
    st.session_state.existing_risks = gpt_extract_risks(scenario)
    st.session_state.new_risks = []
    st.session_state.deleted_existing = set()
    st.session_state.alert_severity_used = alert_severity

if st.session_state.existing_risks or st.session_state.new_risks:
    st.subheader("Mapped Risks and Scores")

    edited_risks = []
    st.markdown("#### GPT-Identified Risks")
    for i, ri in enumerate(st.session_state.existing_risks):
        if i in st.session_state.deleted_existing:
            continue
        cols = st.columns([3, 1, 1, 1, 2, 1])
        name = cols[0].text_input("Scenario", ri.name, key=f"existing_name_{i}")
        severity = cols[1].selectbox("Severity", [0, 1, 2], index=ri.severity, key=f"existing_sev_{i}")
        likelihood = cols[2].selectbox("Likelihood", [0, 1, 2], index=ri.likelihood, key=f"existing_lik_{i}")
        relevance = cols[3].selectbox("Relevance", [0, 1, 2], index=ri.relevance, key=f"existing_rel_{i}")
        category = cols[4].selectbox("Category", [
            "Threat Environment", "Operational Disruption", "Health & Medical Risk",
            "Client Profile & Exposure", "Geo-Political & Intelligence Assessment",
            "Infrastructure & Resource Stability"
        ], index=[
            "Threat Environment", "Operational Disruption", "Health & Medical Risk",
            "Client Profile & Exposure", "Geo-Political & Intelligence Assessment",
            "Infrastructure & Resource Stability"
        ].index(ri.category), key=f"existing_cat_{i}")
        if cols[5].button("🗑️", key=f"del_existing_{i}"):
            st.session_state.deleted_existing.add(i)
            st.experimental_rerun()
        else:
            edited_risks.append(RiskInput(name, severity, relevance, likelihood, category))

    st.markdown("---")
    st.markdown("#### Add Scenarios Manually")
    for i, uid in enumerate(st.session_state.new_risks):
        cols = st.columns([3, 1, 1, 1, 2, 1])
        name = cols[0].text_input("Scenario", key=f"new_name_{uid}")
        severity = cols[1].selectbox("Severity", [0, 1, 2], key=f"new_sev_{uid}")
        likelihood = cols[2].selectbox("Likelihood", [0, 1, 2], key=f"new_lik_{uid}")
        relevance = cols[3].selectbox("Relevance", [0, 1, 2], key=f"new_rel_{uid}")
        category = cols[4].selectbox("Category", [
            "Threat Environment", "Operational Disruption", "Health & Medical Risk",
            "Client Profile & Exposure", "Geo-Political & Intelligence Assessment",
            "Infrastructure & Resource Stability"
        ], key=f"new_cat_{uid}")
        if cols[5].button("🗑️", key=f"del_new_{uid}"):
            st.session_state.new_risks.pop(i)
            st.experimental_rerun()
        else:
            if name:
                edited_risks.append(RiskInput(name, severity, relevance, likelihood, category))

    add_col = st.columns([1, 12])[0]
    if add_col.button("➕ Add Scenario", key="add_button"):
        st.session_state.new_risks.append(str(uuid4()))

    df_summary, aggregated_score, final_score, severity_bonus = calculate_risk_summary(edited_risks, st.session_state.alert_severity_used)

    st.subheader("Scores")
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    st.markdown(f"**Aggregated Risk Score:** {aggregated_score}")
    st.markdown(f"**Alert Severity Bonus:** +{severity_bonus}")
    st.markdown(f"**Final Scenario Score (1-10):** {final_score}")

    risk_level, advice = advice_matrix(final_score, tolerance)
    st.success(f"**Risk Level:** {risk_level}\n\n**Recommended Action:** {advice}")
