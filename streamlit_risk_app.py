import streamlit as st
from dataclasses import dataclass
import pandas as pd
import openai
from collections import Counter
import json
from uuid import uuid4
import math

# Initialize OpenAI client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@dataclass
class RiskInput:
    name: str
    severity: int
    likelihood: int
    category: str

    def weighted_score(self) -> float:
        return self.severity + self.likelihood


def gpt_extract_risks(scenario_text: str) -> list[RiskInput]:
    """
    Sends scenario_text to GPT prompt, parses JSON response into RiskInput objects.
    """
    # Load prompt template
    with open("gpt_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(scenario_text=scenario_text)

    # Call GPT
    with st.spinner("Analyzing..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=0.0
        )

    content = response.choices[0].message.content.strip()

    # Parse JSON
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        st.error("Failed to parse GPT response as JSON.")
        st.code(content, language="json")
        return []

    risks: list[RiskInput] = []
    for entry in parsed:
        # Ensure expected keys
        name = entry.get("name")
        severity = entry.get("severity")
        likelihood = entry.get("likelihood")
        category = entry.get("category")
        if name is None or severity is None or likelihood is None or category is None:
            st.warning(f"Skipping entry with missing fields: {entry}")
            continue
        # Validate types
        try:
            risks.append(
                RiskInput(
                    name=str(name).strip(),
                    severity=int(severity),
                    likelihood=int(likelihood),
                    category=str(category).strip()
                )
            )
        except (ValueError, TypeError):
            st.warning(f"Invalid entry data types: {entry}")
            continue

    if not risks:
        st.warning("No valid risks returned from GPT.")
    return risks


def calculate_risk_summary(inputs: list[RiskInput], critical_alert: bool = False):
    # Build rows and compute raw score
    rows = []
    total_score = 0.0
    for r in inputs:
        sc = r.weighted_score()
        total_score += sc
        rows.append({
            "Scenario": r.name,
            "Risk Category": r.category,
            "Severity": r.severity,
            "Likelihood": r.likelihood,
            "Weighted Score": sc
        })
    df = pd.DataFrame(rows)

    # Normalize (max per risk = 4)
    max_possible = len(inputs) * 4
    normalized = int(round((total_score / max_possible) * 10)) if max_possible > 0 else 0

    # Cluster bonuses
    high = [r for r in inputs if r.weighted_score() == 4]
    mid = [r for r in inputs if 3 <= r.weighted_score() < 4]
    counts = Counter()
    for cat, cnt in Counter([r.category for r in high]).items():
        if cnt >= 3:
            counts[cat] += 1
    for cat, cnt in Counter([r.category for r in mid]).items():
        if cnt >= 5:
            counts[cat] += 1
    qualifying = [c for c, cnt in counts.items() if cnt >= 1]
    if len(qualifying) >= 3:
        cluster_bonus = 2
    elif len(qualifying) >= 1:
        cluster_bonus = 1
    else:
        cluster_bonus = 0

    # Severity bonus for critical alert
    severity_bonus = 1 if critical_alert and total_score > 0 else 0

    # Debug info
    st.markdown("---")
    st.markdown("**Debug Info:**")
    st.markdown(f"Total Score: {total_score}")
    st.markdown(f"Max Possible Score: {max_possible}")
    st.markdown(f"Normalized Score: {normalized}")
    st.markdown(f"Cluster Bonus: {cluster_bonus}")
    st.markdown(f"Severity Bonus: {severity_bonus}")

    # Custom rounding: only round up fractions >= 0.6
    raw_final = normalized + cluster_bonus + severity_bonus
    capped = min(raw_final, 10)
    frac = capped - math.floor(capped)
    final_score = math.ceil(capped) if frac >= 0.6 else math.floor(capped)

    return df, total_score, final_score, severity_bonus


def advice_matrix(score: int) -> dict[str, str]:
    mapping = {
        0: {"Low": "NA", "Moderate": "NA", "High": "NA"},
        1: {"Low": "Normal Precautions", "Moderate": "Normal Precautions", "High": "Normal Precautions"},
        2: {"Low": "Normal Precautions", "Moderate": "Normal Precautions", "High": "Normal Precautions"},
        3: {"Low": "Normal Precautions", "Moderate": "Normal Precautions", "High": "Normal Precautions"},
        4: {"Low": "Heightened Vigilance", "Moderate": "Normal Precautions", "High": "Normal Precautions"},
        5: {"Low": "Heightened Vigilance", "Moderate": "Heightened Vigilance", "High": "Normal Precautions"},
        6: {"Low": "Heightened Vigilance", "Moderate": "Heightened Vigilance", "High": "Heightened Vigilance"},
        7: {"Low": "Crisis24 Consultation Recommended", "Moderate": "Heightened Vigilance", "High": "Heightened Vigilance"},
        8: {"Low": "Crisis24 Consultation Recommended", "Moderate": "Crisis24 Consultation Recommended", "High": "Heightened Vigilance"},
        9: {"Low": "Crisis24 Proactive Engagement", "Moderate": "Crisis24 Consultation Recommended", "High": "Crisis24 Consultation Recommended"},
        10: {"Low": "Crisis24 Proactive Engagement", "Moderate": "Crisis24 Proactive Engagement", "High": "Crisis24 Consultation Recommended"}
    }
    return mapping.get(score, mapping[0])

# Streamlit UI
st.set_page_config(layout="wide")
st.title("AI-Assisted Risk Model & Advice Matrix")

# Session state
if "scenario_text" not in st.session_state:
    st.session_state["scenario_text"] = ""
if "critical_alert" not in st.session_state:
    st.session_state["critical_alert"] = False
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "risks" not in st.session_state:
    st.session_state["risks"] = []
if "deleted" not in st.session_state:
    st.session_state["deleted"] = set()
if "new_entries" not in st.session_state:
    st.session_state["new_entries"] = []
if "show_editor" not in st.session_state:
    st.session_state["show_editor"] = False

# Input form
st.session_state["scenario_text"] = st.text_area("Enter Threat Scenario", value=st.session_state["scenario_text"])
st.session_state["critical_alert"] = st.checkbox("Source is a Critical Severity Crisis24 Alert", value=st.session_state["critical_alert"])

if st.button("Analyze Scenario"):
    # Reset state
    st.session_state["session_id"] = str(uuid4())
    st.session_state["deleted"] = set()
    st.session_state["new_entries"] = []
    st.session_state["show_editor"] = False
    risks = gpt_extract_risks(st.session_state["scenario_text"])
    if risks:
        st.session_state["risks"] = risks
        st.session_state["show_editor"] = True
    else:
        st.error("No risks identified. Please revise input.")

# Editor and summary
if st.session_state["show_editor"]:
    risks = st.session_state["risks"]
    categories = [
        "Threat Environment",
        "Operational Disruption",
        "Health & Medical Risk",
        "Life Safety Risk",
        "Strategic Risk Indicators",
        "Infrastructure & Resource Stability"
    ]
    st.subheader("Mapped Risks and Scores")
    edited = []
    for idx, r in enumerate(risks):
        if idx in st.session_state["deleted"]:
            continue
        cols = st.columns([2,2,1,1,0.5])
        name = cols[0].text_input("Scenario", value=r.name, key=f"name_{idx}")
        cat = cols[1].selectbox("Risk Category", categories, index=categories.index(r.category), key=f"cat_{idx}")
        sev = cols[2].selectbox("Severity", [0,1,2], index=r.severity, key=f"sev_{idx}")
        lik = cols[3].selectbox("Likelihood", [0,1,2], index=r.likelihood, key=f"lik_{idx}")
        if cols[4].button("🗑️", key=f"del_{idx}"):
            st.session_state["deleted"].add(idx)
            st.experimental_rerun()
        else:
            edited.append(RiskInput(name, sev, lik, cat))
    st.markdown("---")
    # New entries
    for j, ne in enumerate(st.session_state["new_entries"]):
        cols = st.columns([2,2,1,1,0.5])
        name = cols[0].text_input("Scenario", value=ne.name, key=f"new_name_{j}")
        cat = cols[1].selectbox("Risk Category", categories, index=categories.index(ne.category), key=f"new_cat_{j}")
        sev = cols[2].selectbox("Severity", [0,1,2], index=ne.severity, key=f"new_sev_{j}")
        lik = cols[3].selectbox("Likelihood", [0,1,2], index=ne.likelihood, key=f"new_lik_{j}")
        if cols[4].button("🗑️", key=f"new_del_{j}"):
            st.session_state["new_entries"].pop(j)
            st.experimental_rerun()
        else:
            st.session_state["new_entries"][j] = RiskInput(name, sev, lik, cat)
    # Add button
    if st.button("➕ Add Scenario"):
        st.session_state["new_entries"].append(RiskInput("",0,0,"Threat Environment"))
        st.experimental_rerun()

    # Calculate and display summary
    all_inputs = edited + st.session_state["new_entries"]
    df_summary, total_score, final_score, severity_bonus = calculate_risk_summary(all_inputs, st.session_state["critical_alert"])
    df_summary.index = df_summary.index + 1
    st.markdown("**Scores:**")
    st.markdown(f"**Aggregated Risk Score:** {total_score}")
    st.markdown(f"**Assessed Risk Score (1–10):** {final_score}")
    advice = advice_matrix(final_score)
    for lvl, adv in advice.items():
        st.markdown(f"**Advice for {lvl} Exposure:** {adv}")
    if severity_bonus:
        st.markdown(f"**Critical Alert Bonus Applied:** +{severity_bonus}")
