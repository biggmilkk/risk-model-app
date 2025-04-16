import streamlit as st
from dataclasses import dataclass
import pandas as pd
import openai

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@dataclass
class RiskInput:
    name: str
    severity: float
    relevance: float
    directionality: float
    mapped_criteria: str

    def assigned_weighting(self) -> float:
        return self.relevance * self.directionality

    def disaggregated_score(self) -> float:
        return self.severity * self.assigned_weighting()

def gpt_extract_risks(scenario_text):
    prompt = f"""
    You are a risk analyst AI. Given the following scenario, return a list of risks. For each risk, map it to one of the known risk criteria below, and estimate its severity (0-2), relevance (0-2 in 0.5 steps), and directionality (0.5, 1, or 1.5).

    Risk Criteria:
    - Critical Incident (e.g., political volatility, protest activity)
    - Life Safety Concerns and Relative Concern of Crisis24 Personnel
    - Severity of Relevant Crisis24 Alerts
    - Impact Considerations (e.g., business/transport disruptions)
    - Location Considerations
    - Immediacy Considerations
    - Symbolic or Political Targeting
    - Airport Type or Access to Safe Exit
    - Housing/Shelter Security
    - Supervision or Organizational Support

    Scenario:
    """ + scenario_text + """

    Return the result in JSON format with this structure:
    [
      {
        "name": "Short risk name",
        "mapped_criteria": "Mapped criteria from list",
        "severity": value,
        "relevance": value,
        "directionality": value
      },
      ...
    ]
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    import json
    content = response.choices[0].message.content
    try:
        return [RiskInput(**entry) for entry in json.loads(content)]
    except Exception as e:
        st.error("Failed to parse GPT response.")
        st.text(content)
        return []

def calculate_risk_summary(inputs):
    rows = []
    for risk in inputs:
        rows.append({
            "Scenario": risk.name,
            "Mapped Criteria": risk.mapped_criteria,
            "Severity": risk.severity,
            "Relevance": risk.relevance,
            "Directionality": risk.directionality,
            "Assigned Weighting": risk.assigned_weighting(),
            "Disaggregated Score": risk.disaggregated_score()
        })

    df = pd.DataFrame(rows)
    aggregated = df["Disaggregated Score"].sum()
    dynamic_scale_total = sum(r.relevance for r in inputs) * 3
    assessed_score = round((aggregated / dynamic_scale_total) * 100, 2) if dynamic_scale_total > 0 else 0.0
    return df, aggregated, assessed_score

def advice_matrix(score: float, tolerance: str):
    if score < 25:
        risk_level = "Low"
    elif score < 50:
        risk_level = "Moderate"
    elif score < 75:
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
st.title("AI-Assisted Risk Model & Advice Generator")

scenario = st.text_area("Enter Threat Scenario")
tolerance = st.selectbox("Select Client Risk Tolerance", ["Low", "Moderate", "High"], index=1)

if st.button("Analyze Scenario"):
    with st.spinner("Analyzing with ChatGPT..."):
        risks = gpt_extract_risks(scenario)
        if risks:
            df_summary, aggregated_score, final_score = calculate_risk_summary(risks)
            risk_level, guidance = advice_matrix(final_score, tolerance)

            st.subheader("Mapped Risks and Scores")
            df_summary.index = df_summary.index + 1  # Start numbering from 1
            st.dataframe(df_summary, use_container_width=True)  # Full-width table

            st.markdown(f"**Aggregated Risk Score:** {aggregated_score}")
            st.markdown(f"**Assessed Risk Score (0-100):** {final_score}")
            st.markdown(f"**Risk Level:** {risk_level}")
            st.markdown(f"**Advice for {tolerance} Tolerance:** {guidance}")
