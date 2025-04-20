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
    max_possible_score = len(inputs) * 8
    normalized_score = int(round((total_score / max_possible_score) * 10)) if max_possible_score > 0 else 0

    high_risks = [r for r in inputs if r.weighted_score() == 8]
    mid_risks = [r for r in inputs if 6 <= r.weighted_score() <= 7]
    low_risks = [r for r in inputs if r.weighted_score() <= 5]

    cluster_counts = Counter()

    for cat, count in Counter([r.category for r in high_risks]).items():
        if count >= 2:
            cluster_counts[cat] += 1
    for cat, count in Counter([r.category for r in mid_risks]).items():
        if count >= 3:
            cluster_counts[cat] += 1
    for cat, count in Counter([r.category for r in low_risks]).items():
        if count >= 5:
            cluster_counts[cat] += 1

    qualifying_categories = [cat for cat, count in cluster_counts.items() if count >= 1]

    if len(qualifying_categories) >= 3:
        cluster_bonus = 2
    elif len(qualifying_categories) >= 1:
        cluster_bonus = 1
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

def advice_matrix(score: int):
    if score == 0:
        return {
            "Low": "NA",
            "Moderate": "NA",
            "High": "NA"
        }
    elif score <= 3:
        return {
            "Low": "Normal Precautions",
            "Moderate": "Normal Precautions",
            "High": "Normal Precautions"
        }
    elif score == 4:
        return {
            "Low": "Heightened Vigilance",
            "Moderate": "Normal Precautions",
            "High": "Normal Precautions"
        }
    elif score == 5:
        return {
            "Low": "Heightened Vigilance",
            "Moderate": "Heightened Vigilance",
            "High": "Normal Precautions"
        }
    elif score == 6:
        return {
            "Low": "Heightened Vigilance",
            "Moderate": "Heightened Vigilance",
            "High": "Heightened Vigilance"
        }
    elif score == 7:
        return {
            "Low": "Crisis24 Consultation Recommended",
            "Moderate": "Heightened Vigilance",
            "High": "Heightened Vigilance"
        }
    elif score == 8:
        return {
            "Low": "Crisis24 Consultation Recommended",
            "Moderate": "Crisis24 Consultation Recommended",
            "High": "Heightened Vigilance"
        }
    elif score == 9:
        return {
            "Low": "Crisis24 Proactive Engagement",
            "Moderate": "Crisis24 Consultation Recommended",
            "High": "Crisis24 Consultation Recommended"
        }
    else:  # score 10
        return {
            "Low": "Crisis24 Proactive Engagement",
            "Moderate": "Crisis24 Proactive Engagement",
            "High": "Crisis24 Consultation Recommended"
        }

st.set_page_config(layout="wide")
st.title("AI-Assisted Risk Model & Advice Matrix")

if "scenario_text" not in st.session_state:
    st.session_state["scenario_text"] = ""

if "use_alert_severity" not in st.session_state:
    st.session_state["use_alert_severity"] = False

if "alert_severity_level" not in st.session_state:
    st.session_state["alert_severity_level"] = "Informational"

if "alert_severity_used" not in st.session_state:
    st.session_state["alert_severity_used"] = None

st.session_state["scenario_text"] = st.text_area("Enter Threat Scenario", value=st.session_state["scenario_text"])
st.session_state["use_alert_severity"] = st.checkbox("Include source alert severity rating", value=st.session_state["use_alert_severity"])
if st.session_state["use_alert_severity"]:
    st.session_state["alert_severity_level"] = st.selectbox("Select Alert Severity (if applicable)", ["Informational", "Caution", "Warning", "Critical"], index=["Informational", "Caution", "Warning", "Critical"].index(st.session_state["alert_severity_level"]))

if st.button("Analyze Scenario"):
    scenario = st.session_state["scenario_text"]
    alert_severity = st.session_state["alert_severity_level"] if st.session_state["use_alert_severity"] else None

    keys_to_keep = {"scenario_text", "use_alert_severity", "alert_severity_level"}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

    st.session_state.session_id = str(uuid4())
    risks = gpt_extract_risks(scenario)
    if risks:
        st.session_state.risks = risks
        st.session_state.deleted_existing = set()
        st.session_state.new_entries = []
        st.session_state.show_editor = True
        st.session_state.alert_severity_used = alert_severity
        st.rerun()
    else:
        st.error("No risks were identified. Please check your scenario.")

if st.session_state.get("show_editor") and st.session_state.get("risks") is not None:
    risks = st.session_state.risks
    categories = [
        "Threat Environment",
        "Operational Disruption",
        "Health & Medical Risk",
        "Life Safety Risk",
        "Strategic Risk Indicators",
        "Infrastructure & Resource Stability"
    ]

    key_prefix = st.session_state.get("session_id", "")

    st.subheader("Mapped Risks and Scores")
    edited_risks = []

    for i, risk in enumerate(risks):
        if i in st.session_state.deleted_existing:
            continue
        cols = st.columns([2, 2, 1, 1, 1, 0.5])
        name = cols[0].text_input("Scenario", value=risk.name, key=f"{key_prefix}_name_{i}")
        category = cols[1].selectbox("Risk Category", categories, index=categories.index(risk.category), key=f"{key_prefix}_cat_{i}")
        severity = cols[2].selectbox("Severity", [0, 1, 2], index=risk.severity, key=f"{key_prefix}_sev_{i}")
        likelihood = cols[3].selectbox("Likelihood", [0, 1, 2], index=risk.likelihood, key=f"{key_prefix}_like_{i}")
        relevance = cols[4].selectbox("Relevance", [0, 1, 2], index=risk.relevance, key=f"{key_prefix}_rel_{i}")
        if cols[5].button("🗑️", key=f"{key_prefix}_del_existing_{i}"):
            st.session_state.deleted_existing.add(i)
            st.rerun()
        else:
            edited_risks.append(RiskInput(name, severity, relevance, likelihood, category))

    st.markdown("---")
    for j, row in enumerate(st.session_state.get("new_entries", [])):
        cols = st.columns([2, 2, 1, 1, 1, 0.5])
        name = cols[0].text_input("Scenario", value=row.name, key=f"name_new_{j}")
        category = cols[1].selectbox("Risk Category", categories, index=categories.index(row.category), key=f"cat_new_{j}")
        severity = cols[2].selectbox("Severity", [0, 1, 2], index=row.severity, key=f"sev_new_{j}")
        likelihood = cols[3].selectbox("Likelihood", [0, 1, 2], index=row.likelihood, key=f"like_new_{j}")
        relevance = cols[4].selectbox("Relevance", [0, 1, 2], index=row.relevance, key=f"rel_new_{j}")
        if cols[5].button("🗑️", key=f"del_new_{j}"):
            st.session_state.new_entries.pop(j)
            st.rerun()
        else:
            st.session_state.new_entries[j] = RiskInput(name, severity, relevance, likelihood, category)

    col_add, _ = st.columns([1, 5])
    with col_add:
        if st.button("➕ Add Scenario", key="add_row_btn_bottom_inline"):
            st.session_state.new_entries.append(RiskInput("", 0, 0, 0, categories[0]))
            st.rerun()

    updated_inputs = edited_risks + st.session_state.new_entries
    df_summary, aggregated_score, final_score, severity_bonus = calculate_risk_summary(updated_inputs, st.session_state.alert_severity_used)
    advice_output = advice_matrix(final_score)

    df_summary.index = df_summary.index + 1

    st.markdown("**Scores:**")
    st.markdown(f"**Aggregated Risk Score:** {aggregated_score}")
    st.markdown(f"**Assessed Risk Score (1–10):** {final_score}")

    for tol, advice in advice_output.items():
        st.markdown(f"**Advice for {tol} Exposure:** {advice}")

    if severity_bonus:
        st.markdown(f"**Alert Severity Bonus Applied:** {st.session_state.alert_severity_used} (+{severity_bonus})")
