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
    category: str

    def assigned_weighting(self) -> float:
        return self.relevance * self.directionality

    def disaggregated_score(self) -> float:
        return self.severity * self.assigned_weighting()

def gpt_extract_risks(scenario_text):
    prompt = f"""
    You are a risk analyst AI. Given the following scenario, return a list of risks. For each risk, map it to one of the following higher-level risk categories, and estimate its severity (0-2), relevance (0-2 in 0.5 steps), and directionality (0.5, 1, or 1.5).

    Risk Categories:
1. Threat Environment (e.g., Critical Incident, Sustained Civil Unrest, Anti-American Sentiment, Status of Government, History of Resolution, Actions Taken by Local Government, Key Populations Being Targeted, Police/Military Presence, Observance of Lawlessness, Likelihood of Regional Conflict Spillover, Other Assistance Companies Issuing Warnings, Other Higher Ed Clients Discussing Evacuation, Closure of Educational Institutions)

2. Operational Disruption (e.g., Impact Considerations, Location Considerations, Immediacy Considerations, Event Lead Time, Road Closures, Curfews, Disruptions to Mobile Voice/SMS/Data Services, Observance of Power Outages, Access to Fuel, Access to Food and Clean Water, Transportation Infrastructure, Airlines Limiting or Canceling Flights)

3. Health & Medical Risk (e.g., Severity of Health Situation [Self or Official Report], Crisis24 Medical Assessment, Deviation from Baseline Medical History, Availability of Medical/Mental Health Treatment, Critical Medication Supply, Need for a Medical Escort, Strain on Local Medical Resources, Increased Transmission of Communicable Diseases, Access to MedEvac, Health Infrastructure Strain)

4. Client Profile & Exposure (e.g., Undergraduate/Graduate/Staff, Supervision/Organizational Support, Program Type, Group Size, Field Site or Urban Environment, How Far Must Commute to Necessities, Housing/Shelter Security, When Travelers Intend to Leave, Airport Type, Access to Intelligence or Info Sharing, Safe Havens or Alternatives)

5. Geo-Political & Intelligence Assessment (e.g., Severity of Crisis24 Alerts, Preexisting Crisis24 Location Intelligence Rating, Dynamic Risk Library Assessment, US State Department Travel Advisory, FCDO Travel Warning, Australia Smarttraveller Warning, Relative Concern of Crisis24 Personnel, Crisis24 Life Safety Assessment, CAT [Crisis Advisory Team] Activation, Organizational Risk Appetite, Existing Mitigations/Security Protocols)

6. Infrastructure & Resource Stability (e.g., Environmental and Weather Risk, Changes in Local Climate, Disruptions to Communication, Internet Infrastructure, Power Grid Stability, Medical System Burden, Communications Breakdown, Relative Capabilities of Assistance Company)

Scenario:
    """ + scenario_text + """

    Return the result in JSON format with this structure:
    [
      {
        "name": "Short risk name",
        "category": "Mapped category from list",
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
            "Risk Category": risk.category,
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
st.title("AI-Assisted Risk Model & Advice Matrix")

scenario = st.text_area("Enter Threat Scenario")
tolerance = st.selectbox("Select Client Risk Tolerance", ["Low", "Moderate", "High"], index=1)

if st.button("Analyze Scenario"):
    with st.spinner("Analyzing..."):
        risks = gpt_extract_risks(scenario)
        if risks:
            categories = [
                "Threat Environment",
                "Operational Disruption",
                "Health & Medical Risk",
                "Client Profile & Exposure",
                "Geo-Political & Intelligence Assessment",
                "Infrastructure & Resource Stability"
            ]
            updated_inputs = []
            st.subheader("Mapped Risks and Scores (Editable)")
            edited_risks = []
            for i, risk in enumerate(risks):
                cols = st.columns(5)
                name = cols[0].text_input("Scenario", value=risk.name, key=f"name_{i}")
                category = cols[1].selectbox("Risk Category", categories, index=categories.index(risk.category), key=f"cat_{i}")
                severity = cols[2].selectbox("Severity", [0, 0.5, 1, 1.5, 2], index=int(risk.severity * 2), key=f"sev_{i}")
                relevance = cols[3].selectbox("Relevance", [0, 0.5, 1, 1.5, 2], index=int(risk.relevance * 2), key=f"rel_{i}")
                directionality = cols[4].selectbox("Directionality", [0.5, 1, 1.5], index=int((risk.directionality - 0.5) * 2), key=f"dir_{i}")
                edited_risks.append(RiskInput(name, severity, relevance, directionality, category))

            updated_inputs = edited_risks

            df_summary, aggregated_score, final_score = calculate_risk_summary(updated_inputs)
            risk_level, guidance = advice_matrix(final_score, tolerance)
            df_summary.index = df_summary.index + 1  # Start numbering from 1
    st.dataframe(df_summary, use_container_width=True)

    if st.button("Recalculate"):
        old_aggregated_score, old_final_score = aggregated_score, final_score
        df_summary, aggregated_score, final_score = calculate_risk_summary(updated_inputs)
        risk_level, guidance = advice_matrix(final_score, tolerance)
        st.markdown("**Updated Results after Recalculation:**")

        delta_agg = aggregated_score - old_aggregated_score
        delta_final = final_score - old_final_score

        agg_note = f"{aggregated_score} (up from {old_aggregated_score})" if delta_agg > 0 else f"{aggregated_score} (down from {old_aggregated_score})" if delta_agg < 0 else f"{aggregated_score} (unchanged)"
        final_note = f"{final_score} (up from {old_final_score})" if delta_final > 0 else f"{final_score} (down from {old_final_score})" if delta_final < 0 else f"{final_score} (unchanged)"

        st.markdown(f"**Aggregated Risk Score:** {agg_note}")
        st.markdown(f"**Assessed Risk Score (0-100):** {final_note}")
        st.markdown(f"**Risk Level:** {risk_level}")
        st.markdown(f"**Advice for {tolerance} Tolerance:** {guidance}")  # Full-width table

            
