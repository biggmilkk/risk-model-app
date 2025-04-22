import streamlit as st
from dataclasses import dataclass
import pandas as pd
import openai
from collections import Counter
import json
from uuid import uuid4
import math
from datetime import datetime

# Initialize OpenAI client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@dataclass
class RiskInput:
    name: str
    severity: int
    likelihood: int
    immediacy: int
    category: str

    def weighted_score(self) -> float:
        return self.severity + self.likelihood + self.immediacy

def gpt_extract_risks(scenario_text: str) -> list[RiskInput]:
    with open("gpt_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    now_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    prompt = prompt_template.format(scenario_text=scenario_text, current_datetime=now_utc)

    with st.spinner("Analyzing..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=0.0
        )

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.strip("`")
    lines = content.splitlines()
    if lines and lines[0].strip().lower() == "json":
        content = "\n".join(lines[1:])

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        st.error("Failed to parse GPT response as JSON.")
        st.code(content, language="json")
        return []

    risks: list[RiskInput] = []
    for entry in parsed:
        if not all(k in entry for k in ("name", "severity", "likelihood", "immediacy", "category")):
            st.warning(f"Skipping invalid entry: {entry}")
            continue
        try:
            risks.append(
                RiskInput(
                    name=str(entry["name"]).strip(),
                    severity=int(entry["severity"]),
                    likelihood=int(entry["likelihood"]),
                    immediacy=int(entry["immediacy"]),
                    category=str(entry["category"]).strip()
                )
            )
        except Exception as e:
            st.warning(f"Error parsing entry {entry}: {e}")
    return risks

def calculate_risk_summary(inputs: list[RiskInput], critical_alert: bool=False):
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
            "Immediacy": r.immediacy,
            "Weighted Score": sc
        })
    df = pd.DataFrame(rows)

    max_possible = len(inputs) * 6
    normalized = int(round((total_score / max_possible) * 10)) if max_possible else 0

    cluster_counts = Counter()
    for r in inputs:
        if r.weighted_score() == 6:
            cluster_counts[r.category] += 1
    high_clusters = [cat for cat, cnt in cluster_counts.items() if cnt >= 2]

    mid_counts = Counter()
    for r in inputs:
        if r.weighted_score() == 5:
            mid_counts[r.category] += 1
    mid_clusters = [cat for cat, cnt in mid_counts.items() if cnt >= 5]

    qualifying = set(high_clusters + mid_clusters)
    cluster_bonus = 2 if len(qualifying) >= 3 else 1 if len(qualifying) >= 1 else 0

    severity_bonus = 1 if critical_alert and total_score > 0 else 0

    st.markdown("---")
    st.markdown("**Debug Info:**")
    st.markdown(f"Total Score: {total_score}")
    st.markdown(f"Normalized Score: {normalized}")
    st.markdown(f"Cluster Bonus: {cluster_bonus}")
    st.markdown(f"Critical Alert Bonus: {severity_bonus}")

    raw_final = normalized + cluster_bonus + severity_bonus
    capped = min(raw_final, 10)
    final_score = math.ceil(capped) if (capped - math.floor(capped)) >= 0.6 else math.floor(capped)
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

# --- App UI ---
st.set_page_config(layout="wide")
st.title("AI-Assisted Risk Model & Advice Matrix")

if "scenario_text" not in st.session_state:
    st.session_state["scenario_text"] = ""
if "critical_alert" not in st.session_state:
    st.session_state["critical_alert"] = False
if "risks" not in st.session_state:
    st.session_state["risks"] = []
if "deleted" not in st.session_state:
    st.session_state["deleted"] = set()
if "new_entries" not in st.session_state:
    st.session_state["new_entries"] = []
if "show_editor" not in st.session_state:
    st.session_state["show_editor"] = False
if "gpt_run_id" not in st.session_state:
    st.session_state["gpt_run_id"] = str(uuid4())

# Input Controls
st.session_state["scenario_text"] = st.text_area("Enter Threat Scenario", st.session_state["scenario_text"])
st.session_state["critical_alert"] = st.checkbox("Source is a Critical Severity Crisis24 Alert", value=st.session_state["critical_alert"])

if st.button("Analyze Scenario"):
    if not st.session_state["scenario_text"].strip():
        st.warning("Please enter a threat scenario before analyzing.")
    else:
        st.session_state.update({
            "deleted": set(),
            "new_entries": [],
            "show_editor": False,
            "gpt_run_id": str(uuid4())  # Force refresh keys
        })
        risks = gpt_extract_risks(st.session_state["scenario_text"])
        if risks:
            st.session_state["risks"] = risks
            st.session_state["show_editor"] = True
        else:
            st.error("No risks identified. Please revise input.")

# Risk Editor and Summary
if st.session_state["show_editor"]:
    categories = [
        "Threat Environment",
        "Operational Disruption",
        "Health & Medical Risk",
        "Life Safety Risk",
        "Strategic Risk Indicators",
        "Infrastructure & Resource Stability"
    ]

    # ✅ Tooltip text
    severity_tooltip = (
        "Severity = How impactful is the risk?\n"
        "• 2 = Major disruption or fatalities\n"
        "• 1 = Localized or moderate impact (default)\n"
        "• 0 = Minimal or no impact"
    )
    likelihood_tooltip = (
        "Likelihood = How likely is the risk to occur?\n"
        "• 2 = Ongoing or confirmed\n"
        "• 1 = Possible or forecasted (default)\n"
        "• 0 = Resolved or speculative"
    )
    immediacy_tooltip = (
        "Immediacy = How soon is the risk expected?\n"
        "• 2 = Happening now or <24h\n"
        "• 1 = Coming days or uncertain (default)\n"
        "• 0 = Past or long-term"
    )

    st.subheader("Mapped Risks and Scores")
    edited = []

    for idx, r in enumerate(st.session_state["risks"]):
        if idx in st.session_state["deleted"]:
            continue
        uid = st.session_state["gpt_run_id"]
        cols = st.columns([2, 2, 1, 1, 1, 0.5])
        name = cols[0].text_input("Scenario", value=r.name, key=f"name_{idx}_{uid}")
        cat = cols[1].selectbox("Risk Category", categories, index=categories.index(r.category), key=f"cat_{idx}_{uid}")
        sev = cols[2].selectbox("Severity", [0, 1, 2], index=r.severity, key=f"sev_{idx}_{uid}", help=severity_tooltip)
        lik = cols[3].selectbox("Likelihood", [0, 1, 2], index=r.likelihood, key=f"lik_{idx}_{uid}", help=likelihood_tooltip)
        imm = cols[4].selectbox("Immediacy", [0, 1, 2], index=r.immediacy, key=f"imm_{idx}_{uid}", help=immediacy_tooltip)
        if cols[5].button("🗑️", key=f"del_{idx}_{uid}"):
            st.session_state["deleted"].add(idx)
        else:
            edited.append(RiskInput(name, sev, lik, imm, cat))

    for j, ne in enumerate(st.session_state["new_entries"]):
        cols = st.columns([2, 2, 1, 1, 1, 0.5])
        name = cols[0].text_input("Scenario", value=ne.name, key=f"new_name_{j}")
        cat = cols[1].selectbox("Risk Category", categories, index=categories.index(ne.category), key=f"new_cat_{j}")
        sev = cols[2].selectbox("Severity", [0, 1, 2], index=ne.severity, key=f"new_sev_{j}", help=severity_tooltip)
        lik = cols[3].selectbox("Likelihood", [0, 1, 2], index=ne.likelihood, key=f"new_lik_{j}", help=likelihood_tooltip)
        imm = cols[4].selectbox("Immediacy", [0, 1, 2], index=ne.immediacy, key=f"new_imm_{j}", help=immediacy_tooltip)
        if cols[5].button("🗑️", key=f"new_del_{j}"):
            st.session_state["new_entries"].pop(j)
        else:
            st.session_state["new_entries"][j] = RiskInput(name, sev, lik, imm, cat)

    if st.button("➕ Add Scenario"):
        st.session_state["new_entries"].append(RiskInput("", 0, 0, 0, categories[0]))

    # Final calculations and advice
    inputs = edited + st.session_state["new_entries"]
    df_summary, total_score, final_score, severity_bonus = calculate_risk_summary(inputs, st.session_state["critical_alert"])
    st.markdown("**Scores:**")
    st.markdown(f"**Aggregated Risk Score:** {total_score}")
    st.markdown(f"**Assessed Risk Score (1–10):** {final_score}")
    advice = advice_matrix(final_score)
    for lvl, adv in advice.items():
        st.markdown(f"**Advice for {lvl} Exposure:** {adv}")
    if severity_bonus:
        st.markdown(f"**Critical Alert Bonus Applied:** +{severity_bonus}")

