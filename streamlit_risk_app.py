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
    directionality: int
    likelihood: int
    relevance: int
    category: str

    def weighted_score(self) -> int:
        return (self.severity * 1) + (self.relevance * 2) + (self.directionality * 1) + (self.likelihood * 1)

def gpt_extract_risks(scenario_text):
    prompt = f"""
You are a risk analyst AI. Given the following scenario, return a list of distinct, clearly defined risks. For each risk, classify it into one of the following high-level categories. Include a representative short name for each risk and assign the following scores using integers (0, 1, or 2):

Risk Categories with Examples:

1. Threat Environment
   - Examples: Critical Incident, Sustained Civil Unrest, Anti-American Sentiment, Status of Government, History of Resolution, Actions Taken by Local Government, Key Populations Being Targeted, Police/Military Presence, Observance of Lawlessness, Likelihood of Regional Conflict Spillover

2. Operational Disruption
   - Examples: Impact Considerations, Location Considerations, Immediacy Considerations, Event Lead Time, Road Closures, Curfews, Disruptions to Mobile Voice/SMS/Data Services, Observance of Power Outages, Access to Fuel, Access to Food and Clean Water, Transportation Infrastructure, Airlines Limiting or Canceling Flights

3. Health & Medical Risk
   - Examples: Severity of Health Situation (Self or Official Report), Crisis24 Medical Assessment, Deviation from Baseline Medical History, Availability of Medical/Mental Health Treatment, Critical Medication Supply, Need for a Medical Escort, Strain on Local Medical Resources, Increased Transmission of Communicable Diseases, Access to MedEvac, Health Infrastructure Strain

4. Client Profile & Exposure
   - Examples: Undergraduate/Graduate/Staff, Supervision/Organizational Support, Program Type, Group Size, Field Site or Urban Environment, How Far Must Commute to Necessities, Housing/Shelter Security, When Travelers Intend to Leave, Airport Type, Access to Intelligence or Info Sharing, Safe Havens or Alternatives

5. Geo-Political & Intelligence Assessment
   - Examples: Severity of Crisis24 Alerts, Preexisting Crisis24 Location Intelligence Rating, Dynamic Risk Library Assessment, US State Department Travel Advisory, FCDO Travel Warning, Australia Smarttraveller Warning, Relative Concern of Crisis24 Personnel, Crisis24 Life Safety Assessment, CAT (Crisis Advisory Team) Activation, Organizational Risk Appetite, Existing Mitigations/Security Protocols

6. Infrastructure & Resource Stability
   - Examples: Environmental and Weather Risk, Changes in Local Climate, Disruptions to Communication, Internet Infrastructure, Power Grid Stability, Medical System Burden, Communications Breakdown, Relative Capabilities of Assistance Company

Scoring Instructions:
- Severity: 0 (low), 1 (moderate), 2 (high)
- Directionality: 0 (improving), 1 (stable), 2 (worsening)
- Likelihood: 0 (unlikely), 1 (possible), 2 (likely or ongoing)
- Relevance: 0 (not relevant), 1 (somewhat relevant), 2 (very relevant)

Return only a valid JSON array using this exact format:

[
  {{
    "name": "Short description of the risk",
    "category": "One of: Threat Environment, Operational Disruption, Health & Medical Risk, Client Profile & Exposure, Geo-Political & Intelligence Assessment, Infrastructure & Resource Stability",
    "severity": 0,
    "directionality": 0,
    "likelihood": 0,
    "relevance": 0
  }}
]

Do not include explanations, markdown, or any text before or after the JSON.

Scenario:
{scenario_text}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that returns only valid JSON arrays."},
            {"role": "user", "content": prompt}
        ],
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

st.title("AI-Assisted Risk Model & Advice Matrix")

scenario = st.text_area("Paste or type your scenario here")

if "risks" not in st.session_state:
    st.session_state.risks = []

if st.button("Analyze Scenario"):
    with st.spinner("Analyzing..."):
        st.session_state.risks = gpt_extract_risks(scenario)

if st.session_state.risks:
    st.subheader("Mapped Risks and Scores")
    risk_data = []
    for i, risk in enumerate(st.session_state.risks):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            name = st.text_input("Risk", value=risk.name, key=f"name_{i}")
        with col2:
            severity = st.selectbox("Severity", [0, 1, 2], index=risk.severity, key=f"severity_{i}")
        with col3:
            directionality = st.selectbox("Directionality", [0, 1, 2], index=risk.directionality, key=f"directionality_{i}")
        with col4:
            likelihood = st.selectbox("Likelihood", [0, 1, 2], index=risk.likelihood, key=f"likelihood_{i}")
        with col5:
            relevance = st.selectbox("Relevance", [0, 1, 2], index=risk.relevance, key=f"relevance_{i}")
        with col6:
            category = st.selectbox("Category", [
                "Threat Environment",
                "Operational Disruption",
                "Health & Medical Risk",
                "Client Profile & Exposure",
                "Geo-Political & Intelligence Assessment",
                "Infrastructure & Resource Stability"
            ], index=[
                "Threat Environment",
                "Operational Disruption",
                "Health & Medical Risk",
                "Client Profile & Exposure",
                "Geo-Political & Intelligence Assessment",
                "Infrastructure & Resource Stability"
            ].index(risk.category), key=f"category_{i}")

        risk_data.append(RiskInput(name, severity, directionality, likelihood, relevance, category))

    total_score = sum(r.weighted_score() for r in risk_data)
    normalized_score = min(10, round(total_score / max(len(risk_data), 1)))
    st.markdown(f"**Aggregated Score:** {total_score} → **Final Risk Rating (1-10):** {normalized_score}")
