import streamlit as st
from dataclasses import dataclass
import pandas as pd
import openai
from collections import Counter
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
        return (self.severity * 1) + (self.relevance * 2) + (self.directionality * 1) + (self.likelihood * 1)

def gpt_extract_risks(scenario_text):
    with open("gpt_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(scenario_text=scenario_text)

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
            risks.append(RiskInput(**entry))
        except Exception as e:
            st.warning(f"Failed to convert entry: {entry}")
            st.exception(e)

    if not risks:
        st.warning("Parsed successfully, but no valid risks were returned.")

    return risks

def calculate_risk_summary(inputs):
    rows = []
    total_score = 0
    for risk in inputs:
        score = risk.weighted_score()
        total_score += score
        rows.append({
            "Scenario": risk.name,
            "Risk Category": risk.category,
            "Severity": risk.severity,
            "Directionality": risk.directionality,
            "Likelihood": risk.likelihood,
            "Relevance": risk.relevance,
            "Weighted Score": score
        })

    df = pd.DataFrame(rows)
    max_possible_score = len(inputs) * 10
    normalized_score = int(round((total_score / max_possible_score) * 10)) if max_possible_score > 0 else 0

    categories = [r.category for r in inputs]
    total_categories = len(categories)
    unique_categories = len(set(categories))
    if total_categories > 0:
        clustering_ratio = (total_categories - unique_categories) / total_categories
        cluster_bonus = min(round(clustering_ratio * 2), 2)
    else:
        cluster_bonus = 0

    final_score = min(normalized_score + cluster_bonus, 10)
    return df, total_score, final_score

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
st.title("AI-Assisted Risk Model & Advice Matrix")

scenario = st.text_area("Enter Threat Scenario")
tolerance = st.selectbox("Select Client Risk Tolerance", ["Low", "Moderate", "High"], index=1)

if st.button("Analyze Scenario"):
    with st.spinner("Analyzing..."):
        st.session_state.risks = gpt_extract_risks(scenario)
        st.session_state.deleted_existing = set()
        st.session_state.new_entries = []
        st.session_state.show_editor = True

if st.session_state.get("show_editor") and st.session_state.get("risks") is not None:
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
    edited_risks = []

    for i, risk in enumerate(risks):
        if i in st.session_state.deleted_existing:
            continue
        cols = st.columns([2, 2, 1, 1, 1, 1, 0.5])
        name = cols[0].text_input("Scenario", value=risk.name, key=f"name_{i}")
        category = cols[1].selectbox("Risk Category", categories, index=categories.index(risk.category), key=f"cat_{i}")
        severity = cols[2].selectbox("Severity", [0, 1, 2], index=risk.severity, key=f"sev_{i}")
        directionality = cols[3].selectbox("Directionality", [0, 1, 2], index=risk.directionality, key=f"dir_{i}")
        likelihood = cols[4].selectbox("Likelihood", [0, 1, 2], index=risk.likelihood, key=f"like_{i}")
        relevance = cols[5].selectbox("Relevance", [0, 1, 2], index=risk.relevance, key=f"rel_{i}")
        if cols[6].button("🗑️", key=f"del_existing_{i}"):
            st.session_state.deleted_existing.add(i)
            st.experimental_rerun()
        else:
            edited_risks.append(RiskInput(name, severity, relevance, directionality, likelihood, category))

    st.markdown("---")
    for j, row in enumerate(st.session_state.get("new_entries", [])):
        cols = st.columns([2, 2, 1, 1, 1, 1, 0.5])
        name = cols[0].text_input("Scenario", value=row.name, key=f"name_new_{j}")
        category = cols[1].selectbox("Risk Category", categories, index=categories.index(row.category), key=f"cat_new_{j}")
        severity = cols[2].selectbox("Severity", [0, 1, 2], index=row.severity, key=f"sev_new_{j}")
        directionality = cols[3].selectbox("Directionality", [0, 1, 2], index=row.directionality, key=f"dir_new_{j}")
        likelihood = cols[4].selectbox("Likelihood", [0, 1, 2], index=row.likelihood, key=f"like_new_{j}")
        relevance = cols[5].selectbox("Relevance", [0, 1, 2], index=row.relevance, key=f"rel_new_{j}")
        if cols[6].button("🗑️", key=f"del_new_{j}"):
            st.session_state.new_entries.pop(j)
            st.experimental_rerun()
        else:
            st.session_state.new_entries[j] = RiskInput(name, severity, relevance, directionality, likelihood, category)

    col_add, _ = st.columns([1, 5])
    with col_add:
        if st.button("➕", key="add_row_btn_bottom_inline"):
            st.session_state.new_entries.append(
                RiskInput("", 0, 0, 0, 0, categories[0])
            )
            st.experimental_rerun()

    updated_inputs = edited_risks + st.session_state.new_entries

    df_summary, aggregated_score, final_score = calculate_risk_summary(updated_inputs)
    risk_level, guidance = advice_matrix(final_score, tolerance)

    df_summary.index = df_summary.index + 1

    st.markdown("**Scores:**")
    st.markdown(f"**Aggregated Risk Score:** {aggregated_score}")
    st.markdown(f"**Assessed Risk Score (1–10):** {final_score}")
    st.markdown(f"**Risk Level:** {risk_level}")
    st.markdown(f"**Advice for {tolerance} Tolerance:** {guidance}")
