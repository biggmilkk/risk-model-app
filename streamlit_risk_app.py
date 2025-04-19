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
- **Severity**: 0 (low), 1 (moderate), 2 (high)
- **Directionality**: 0 (improving), 1 (stable), 2 (worsening)
- **Likelihood**: 0 (unlikely), 1 (possible), 2 (likely or ongoing)
- **Relevance**: 0 (not relevant), 1 (somewhat relevant), 2 (very relevant)

Return only a valid JSON array using this exact format:

[
  {{
    "name": "Short description of the risk",
    "category": "One of: Threat Environment, Operational Disruption, Health & Medical Risk, Client Profile & Exposure, Geo-Political & Intelligence Assessment, Infrastructure & Resource Stability",
    "severity": 0-2,
    "directionality": 0-2,
    "likelihood": 0-2,
    "relevance": 0-2
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
