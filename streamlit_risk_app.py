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
        return self.severity * 1 + self.relevance * 2 + self.directionality * 1 + self.likelihood * 1


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

Do **not** assign Likelihood = 2 when phrasing includes soft or tentative language such as “may,” “could,” “might,” or “are considering,” unless paired with official confirmation or additional strong indicators. These represent **possibility**, not certainty. For example:
- “Officials may impose movement restrictions” → Likelihood = 1
- “Retaliatory attacks are possible” → Likelihood = 1
- “Clashes cannot be ruled out” → Likelihood = 1
These are speculative unless accompanied by concrete alerts, patterns, or actionable intelligence indicating strong probability.

Examples:
- “Firefighters are responding to a blaze” → Likelihood = 2
- “Authorities have evacuated a university due to a fire” → Likelihood = 2
- “Power outages have occurred” → Likelihood = 2
- “Authorities **could** issue evacuation orders” → Likelihood = 1
- “Urban flooding **may** occur” → Likelihood = 1
- “Flooding is **ongoing**” or “Evacuation **underway**” → Likelihood = 2
- “The Bharatiya Janata Party (BJP) will march on April 17…” → Likelihood = 2
- “Organizers expect thousands to attend a planned protest” → Likelihood = 2

Assigning all risks a Likelihood of 2 is incorrect and inflates the model. Do not assign a 2 unless the specific line describing the risk includes words or evidence supporting high certainty.

If a threat described in the scenario has already been resolved or is no longer active (e.g., a successful rescue, arrest, de-escalation), reflect this in the scoring:
- Set **directionality** to 0 (improving) if the situation has been mitigated or resolved.
- Lower the **severity** and **likelihood** accordingly.
- Do not generate risks for events that have concluded unless there are relevant **ongoing** implications or residual risks (e.g., reputational risk, recurring threat).

Example:
- “A kidnapped individual was rescued and the suspects were killed” → Directionality = 0, Likelihood = 0 or 1, Severity = 0 or 1 depending on ongoing risk.

Group impacts stemming from the same root cause **only if they affect the same mode of operation, area, and timeframe**. For example, if both road closures and traffic delays are caused by snow in the same region and time window, combine them as “Winter storm-related road transport disruption.”

However, **preserve separate risk entries** when the effects differ in nature (e.g., air vs. road), geography, or timing. Avoid excessive grouping that could overlook important distinctions in how risks impact clients.

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
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        st.error("Failed to parse GPT response.")
        st.code(content, language="json")
        return []

    risks = []
    for entry in parsed:
        try:
            risks.append(RiskInput(**entry))
        except Exception:
            st.warning(f"Failed to convert entry: {entry}")
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
    clustering_ratio = (total_categories - unique_categories) / total_categories if total_categories > 0 else 0
    cluster_bonus = min(round(clustering_ratio * 2), 2)
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
        advice = (
            "Crisis24 Proactive Engagement"
            if risk_level in ["High", "Extreme"] else
            "Heightened Vigilance" if risk_level == "Moderate" else
            "Normal Precautions"
        )
    elif tolerance == "Moderate":
        advice = (
            "Consider Crisis24 Consultation"
            if risk_level == "Extreme" else
            "Heightened Vigilance" if risk_level == "High" else
            "Normal Precautions"
        )
    elif tolerance == "High":
        advice = (
            "Heightened Vigilance" if risk_level == "Extreme" else "Normal Precautions"
        )
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
        "Infrastructure & Resource Stability",
    ]
    st.subheader("Mapped Risks and Scores")
    for i, risk in enumerate(risks):
        cols = st.columns([2, 2, 1, 1, 1, 1, 0.5])
        name = cols[0].text_input("Scenario", value=risk.name, key=f"name_{i}")
        category = cols[1].selectbox("Risk Category", categories, index=categories.index(risk.category), key=f"cat_{i}")
        severity = cols[2].selectbox("Severity", [0, 1, 2], index=risk.severity, key=f"sev_{i}")
        directionality = cols[3].selectbox("Directionality", [0, 1, 2], index=risk.directionality, key=f"dir_{i}")
        likelihood = cols[4].selectbox("Likelihood", [0, 1, 2], index=risk.likelihood, key=f"like_{i}")
        relevance = cols[5].selectbox("Relevance", [0, 1, 2], index=risk.relevance, key=f"rel_{i}")
        if cols[6].button("🗑️", key=f"del_{i}"):
            st.session_state.risks.pop(i)
            st.experimental_rerun()
        else:
            st.session_state.risks[i] = RiskInput(name, severity, relevance, directionality, likelihood, category)
    if "new_count" not in st.session_state:
        st.session_state.new_count = 0
    st.markdown("---")
    for j in range(st.session_state.new_count):
        cols = st.columns([2, 2, 1, 1, 1, 1, 0.5])
        name = cols[0].text_input("Scenario", value="", key=f"name_new_{j}")
        category = cols[1].selectbox("Risk Category", categories, key=f"cat_new_{j}")
        severity = cols[2].selectbox("Severity", [0, 1, 2], key=f"sev_new_{j}")
        directionality = cols[3].selectbox("Directionality", [0, 1, 2], key=f"dir_new_{j}")
        likelihood = cols[4].selectbox("Likelihood", [0, 1, 2], key=f"like_new_{j}")
        relevance = cols[5].selectbox("Relevance", [0, 1, 2], key=f"rel_new_{j}")
        if cols[6].button("🗑️", key=f"del_new_{j}"):
            continue
        if name:
            st.session_state.risks.append(RiskInput(name, severity, relevance, directionality, likelihood, category))
    col_add, _ = st.columns([1, 5])
    with col_add:
        if st.button("➕", key="add_row_btn_bottom_inline"):
            st.session_state.new_count += 1
            st.experimental_rerun()
    df_summary, aggregated_score, final_score = calculate_risk_summary(st.session_state.risks)
    risk_level, guidance = advice_matrix(final_score, tolerance)
    df_summary.index = df_summary.index + 1
    st.markdown("**Scores:**")
    st.markdown(f"**Aggregated Risk Score:** {aggregated_score}")
    st.markdown(f"**Assessed Risk Score (1–10):** {final_score}")
    st.markdown(f"**Risk Level:** {risk_level}")
    st.markdown(f"**Advice for {tolerance} Tolerance:** {guidance}")
