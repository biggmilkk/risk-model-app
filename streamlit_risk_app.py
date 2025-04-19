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

    - 0 (Unlikely): Use for risks that are speculative, rare, or expressed with weak/moderate conditional language. Look for: “unlikely,” “improbable,” “uncertain.”
    - 1 (Possible): Use for risks that are mentioned or under monitoring but have unclear certainty. Keywords: “possible,” “could,” “might,” “may,” “potential.” Use this for most forecast-based risks unless the scenario clearly confirms the event will happen.
    - 2 (Likely): Use if the risk is highly probable, actively occurring, or has been confirmed by official sources or real-time developments. This includes active alerts, confirmed incidents, or ongoing events. Keywords: “likely,” “expected,” “currently happening,” “ongoing,” or evidence of active emergency response.

    Give a **Likelihood of 2** if the event is actively occurring, confirmed by reports or official alerts, or underway at the time of reporting. For example, active fires, confirmed transport disruptions, evacuations in progress, or emergency responses in motion should all be considered high likelihood.

    Avoid assigning Likelihood = 1 to ongoing or already confirmed disruptions just because they are phrased as possibilities in adjacent context.

    If authorities or credible reporting suggests a high probability of a development—such as heightened security after a deadly attack—this should also be scored as **Likelihood = 2**. For example, "heightened security will likely be maintained" reflects a strong expectation and should not be downgraded to 1.

    Use context to differentiate between speculative responses (e.g., “may deploy security forces”) and confident operational forecasts (e.g., “will likely maintain heightened security”). The latter implies a high-certainty action and should receive **Likelihood = 2** if stated clearly.

    If authorities or scenario text indicates that disruptions are “likely,” “expected,” or “probable,” especially in operational or logistical domains (e.g., transport, security), this should be scored as **Likelihood = 2**. This includes phrases like “localized transport disruptions are likely” or “clashes are expected.”

    Do not downscore these situations just because the event hasn't yet occurred — the language strongly indicates high confidence and should be reflected accordingly.

    Do **not** assign Likelihood = 2 when phrasing includes soft or tentative language such as “may,” “could,” “might,” or “are considering,” unless paired with official confirmation or additional strong indicators. These represent **possibility**, not certainty.

    Example:
    - “Officials may impose movement restrictions” → Likelihood = 1
    - “Retaliatory attacks are possible” → Likelihood = 1
    - “Clashes cannot be ruled out” → Likelihood = 1

    If a threat described in the scenario has already been resolved or is no longer active (e.g., a successful rescue, arrest, de-escalation), reflect this in the scoring:
    - Set **directionality** to 0 (improving) if the situation has been mitigated or resolved.
    - Lower the **severity** and **likelihood** accordingly.
    - Do not generate risks for events that have concluded unless there are relevant **ongoing** implications or residual risks.

    Group impacts stemming from the same root cause **only if they affect the same mode of operation, area, and timeframe**.

    However, **preserve separate risk entries** when the effects differ in nature (e.g., air vs. road), geography, or timing. Avoid excessive grouping that could overlook important distinctions in how risks impact clients.

    Return only a valid JSON array:
    [
      {
        "name": "...",
        "category": "...",
        "severity": 0,
        "directionality": 0,
        "likelihood": 0,
        "relevance": 0
      }
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
    try:
        parsed = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        st.error("Failed to parse GPT response.")
        st.code(response.choices[0].message.content, language="json")
        return []
    return [RiskInput(**e) for e in parsed if isinstance(e, dict)]


def calculate_risk_summary(inputs):
    rows, total = [], 0
    for r in inputs:
        s = r.weighted_score(); total += s
        rows.append({
            "Scenario":r.name,
            "Risk Category":r.category,
            "Severity":r.severity,
            "Directionality":r.directionality,
            "Likelihood":r.likelihood,
            "Relevance":r.relevance,
            "Weighted Score":s
        })
    df = pd.DataFrame(rows)
    max_s = len(inputs)*10
    norm = int(round((total/max_s)*10)) if max_s else 0
    uniq = len(set(r.category for r in inputs))
    bonus = min(round(((len(inputs)-uniq)/len(inputs))*2),2) if inputs else 0
    return df, total, min(norm+bonus,10)


def advice_matrix(score: int, tolerance: str):
    lvl = "Low" if score<=3 else "Moderate" if score<=6 else "High" if score<=8 else "Extreme"
    if tolerance=="Low": adv = "Crisis24 Proactive Engagement" if lvl in ["High","Extreme"] else "Heightened Vigilance" if lvl=="Moderate" else "Normal Precautions"
    elif tolerance=="Moderate": adv = "Consider Crisis24 Consultation" if lvl=="Extreme" else "Heightened Vigilance" if lvl=="High" else "Normal Precautions"
    else: adv = "Heightened Vigilance" if lvl=="Extreme" else "Normal Precautions"
    return lvl, adv

st.set_page_config(layout="wide")
st.title("AI-Assisted Risk Model & Advice Matrix")

scenario = st.text_area("Enter Threat Scenario")
tolerance = st.selectbox("Select Client Risk Tolerance",["Low","Moderate","High"],index=1)

if st.button("Analyze Scenario"):
    with st.spinner("Analyzing..."):
        st.session_state.risks = gpt_extract_risks(scenario)
        st.session_state.show_editor = True

if "risks" not in st.session_state: st.session_state.risks=[]

if st.session_state.get("show_editor") and st.session_state.risks:
    risks=st.session_state.risks
    cats=["Threat Environment","Operational Disruption","Health & Medical Risk","Client Profile & Exposure","Geo-Political & Intelligence Assessment","Infrastructure & Resource Stability"]
    st.subheader("Mapped Risks and Scores")
    for i,r in enumerate(risks):
        cols=st.columns([2,2,1,1,1,1,0.5])
        n=cols[0].text_input("Scenario",value=r.name,key=f"n{i}")
        c=cols[1].selectbox("Risk Category",cats,index=cats.index(r.category),key=f"c{i}")
        sev=cols[2].selectbox("Severity",[0,1,2],index=r.severity,key=f"s{i}")
        dir=cols[3].selectbox("Directionality",[0,1,2],index=r.directionality,key=f"d{i}")
        lik=cols[4].selectbox("Likelihood",[0,1,2],index=r.likelihood,key=f"l{i}")
        rel=cols[5].selectbox("Relevance",[0,1,2],index=r.relevance,key=f"e{i}")
        if cols[6].button("🗑️",key=f"del{i}"):
            risks.pop(i); break
        else: risks[i]=RiskInput(n,sev,rel,dir,lik,c)
    if "new_count" not in st.session_state: st.session_state.new_count=0
    st.markdown("---")
    for j in range(st.session_state.new_count):
        cols=st.columns([2,2,1,1,1,1,0.5])
        n=cols[0].text_input("Scenario",key=f"nn{j}")
        c=cols[1].selectbox("Risk Category",cats,key=f"cc{j}")
        sev=cols[2].selectbox("Severity",[0,1,2],key=f"ss{j}")
        dir=cols[3].selectbox("Directionality",[0,1,2],key=f"dd{j}")
        lik=cols[4].selectbox("Likelihood",[0,1,2],key=f"ll{j}")
        rel=cols[5].selectbox("Relevance",[0,1,2],key=f"ee{j}")
        if n: risks.append(RiskInput(n,sev,rel,dir,lik,c))
    col_add,_=st.columns([1,5])
    with col_add:
        if st.button("➕ Add row"): st.session_state.new_count+=1
    df,tot,fin=calculate_risk_summary(risks)
    lvl,adv=advice_matrix(fin,tolerance)
    st.markdown(f"**Aggregated Risk Score:** {tot}")
    st.markdown(f"**Assessed Risk Score (1–10):** {fin}")
    st.markdown(f"**Risk Level:** {lvl}")
    st.markdown(f"**Advice for {tolerance} Tolerance:** {adv}")
