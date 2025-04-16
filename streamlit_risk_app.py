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
- Critical Incident? (e.g., political volatility [armed clashes, curfew, coup d'etat, border closure, close of travel hubs, violent civil unrest]; large-scale natural disaster; terrorism; large-scale industrial disaster; plane accident with traveler registered as on board)
- Crisis24 Assessment of Life Safety Concerns
- Relative Concern of Crisis24 Personnel When Viewing Event from Client’s Perspective
- Severity of Relevant Crisis24 Alerts Issued
- Preexisting Crisis24 Location Intelligence Overall Rating
- Dynamic Risk Library Assessment of Political Instability/Security & Natural Disasters Odds
- Location Considerations
- Immediacy Considerations
- Impact Considerations
- Crisis Advice Team (CAT) Assembled for Meetings?
- When Travelers Intend to Leave Travel Location
- Airport Type From Which Travelers Intend to Leave
- Supervision/Support
- Security of Housing (Assuming Travelers Can Get There)
- Undergraduate, Graduate, Staff, or Faculty Travelers
- Strain on Local Medical Resources
- Program Type
- Group Size
- Field Site or Urban Environment
- Increased Transmission of Communicable Diseases
- Changes in Local Climate (Environmental)
- How Far Students Must Commute to Necessities (More or Less Exposure to Risky Environments)
- Event Lead Time Before Impact
- US Department of State Travel Advisory Status
- US Department of State requests/allows departure of nonessential employees
- UK FCDO Travel Warnings Status in Area of Travel
- Australia Smarttraveller Warnings Status in Area of Travel
- Actions Taken by Local Government
- Other Assistance Companies Issuing Warning/Standby Notices
- Other Higher Ed Clients Discussing Evacuation
- Closure of Educational Institutions
- Key Populations Being Targeted
- Anti-American sentiment
- History of Resolution
- Status of Government
- Sustained Civil Unrest
- Police/Military Presence in Areas Where Students Live or Travelers are Staying
- Observance of Lawlessness
- Likelihood of Regional Conflict Spillover Into Traveler's Area of Operations
- Airlines Limiting or Canceling Flights Into or Out of Major Airports
- Curfews That Limit Movement to Airports or Access to Necessary Resources
- Road Closures that Limit Movement to Airports or Access to Necessary Resources
- Relative Capabilities of Assistance Company
- Disruptions to Mobile Voice and SMS Services
- Disruptions to Mobile Data and Internet Services
- Access to Food and Clean Water
- Access to Fuel
- Issues with Interpersonal Communication
- Observance of Power Outages
- Severity of Health Situation (Patient Self-Report)
- Severity of Health Situation (Official/Provider Report)
- Reported Magnitude of Deviation from Baseline Medical/Mental Health History
- Crisis24 Medical Assessment of Current Medical/Mental Health Condition (If Available)
- Availability of Relevant Medical/Mental Health Treatment
- Critical Medication Supply/Availability
- Need for a Medical Escort

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
    with st.spinner("Analyzing..."):
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
