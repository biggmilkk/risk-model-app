import streamlit as st
from dataclasses import dataclass
import pandas as pd
import openai
from collections import Counter

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
    prompt = f"""
    You are a risk analyst AI. Given the following scenario, return a list of risks. For each risk, map it to one of the following higher-level risk categories, and estimate its severity (0-2), relevance (0-2), and directionality (0-2). Use whole numbers only.

    Risk Categories:
    ... (rest of your prompt remains the same)
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    import json
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
    except Exception as e:
        st.error("Failed to parse GPT response.")
        st.code(content, language="json")
        return []

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
            "Relevance": risk.relevance,
            "Directionality": risk.directionality,
            "Likelihood": risk.likelihood,
            "Weighted Score": score
        })

    df = pd.DataFrame(rows)
    max_possible_score = len(inputs) * 10
    normalized_score = int(round((total_score / max_possible_score) * 10)) if max_possible_score > 0 else 0

    # Risk Clustering Bonus
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
        st.session_state.show_editor = True

if st.session_state.get("show_editor") and st.session_state.get("risks"):
    risks = st.session_state.risks
    categories = [
        "Threat Environment",
        "Operational Disruption",
        "Health & Medical Risk",
        "Client Profile & Exposure",
        "Geo-Political & Intelligence Assessment",
        "Infrastructure & Resource Stability"
    ]
    updated_inputs = []
    st.subheader("Mapped Risks and Scores")
    edited_risks = []
    for i, risk in enumerate(risks):
        cols = st.columns(6)
        name = cols[0].text_input("Scenario", value=risk.name, key=f"name_{i}")
        category = cols[1].selectbox("Risk Category", categories, index=categories.index(risk.category), key=f"cat_{i}")
        severity = cols[2].selectbox("Severity", [0, 1, 2], index=risk.severity if risk.severity in [0, 1, 2] else 1, key=f"sev_{i}")
        relevance = cols[3].selectbox("Relevance", [0, 1, 2], index=risk.relevance if risk.relevance in [0, 1, 2] else 1, key=f"rel_{i}")
        directionality = cols[4].selectbox("Directionality", [0, 1, 2], index=risk.directionality if risk.directionality in [0, 1, 2] else 1, key=f"dir_{i}")
        likelihood = cols[5].selectbox("Likelihood", [0, 1, 2], index=risk.likelihood if risk.likelihood in [0, 1, 2] else 1, key=f"like_{i}")
        edited_risks.append(RiskInput(name, severity, relevance, directionality, likelihood, category))

    updated_inputs = edited_risks

    df_summary, aggregated_score, final_score = calculate_risk_summary(updated_inputs)
    risk_level, guidance = advice_matrix(final_score, tolerance)

    df_summary.index = df_summary.index + 1  # Start numbering from 1

    st.markdown("**Scores:**")
    st.markdown(f"**Aggregated Risk Score:** {aggregated_score}")
    st.markdown(f"**Assessed Risk Score (1–10):** {final_score}")
    st.markdown(f"**Risk Level:** {risk_level}")
    st.markdown(f"**Advice for {tolerance} Tolerance:** {guidance}")
