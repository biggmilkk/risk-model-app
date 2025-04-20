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
            temperature=0,
            top_p=0.2
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

    st.markdown("---")
    st.markdown("**Debug Info:**")
    st.markdown(f"Total Score: {total_score}")
    st.markdown(f"Max Possible Score: {max_possible_score}")
    st.markdown(f"Normalized Score: {normalized_score}")
    st.markdown(f"Cluster Bonus: {cluster_bonus}")
    st.markdown(f"Severity Bonus: {severity_bonus}")

    final_score = min(normalized_score + cluster_bonus + severity_bonus, 10)
    return df, total_score, final_score, severity_bonus
