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
    immediacy: int
    category: str

    def weighted_score(self) -> float:
        return self.severity + self.likelihood


def gpt_extract_risks(scenario_text: str) -> list[RiskInput]:
    with open("gpt_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt = prompt_template.format(scenario_text=scenario_text)

    with st.spinner("Analyzing..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=0.0
        )

    content = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        st.error("Failed to parse GPT response as JSON.")
        st.code(content, language="json")
        return []

    risks = []
    for entry in parsed:
        name = entry.get("name")
        severity = entry.get("severity")
        likelihood = entry.get("likelihood")
        immediacy = entry.get("immediacy")
        category = entry.get("category")
        if None in (name, severity, likelihood, immediacy, category):
            st.warning(f"Skipping entry with missing fields: {entry}")
            continue
        try:
            risks.append(RiskInput(
                name=str(name),
                severity=int(severity),
                likelihood=int(likelihood),
                immediacy=int(immediacy),
                category=str(category)
            ))
        except Exception:
            st.warning(f"Invalid entry types: {entry}")
    if not risks:
        st.warning("No valid risks returned from GPT.")
    return risks


def calculate_risk_summary(inputs: list[RiskInput], critical_alert: bool=False):
    rows = []
    total = 0
    for r in inputs:
        score = r.weighted_score()
        total += score
        rows.append({
            "Scenario": r.name,
            "Risk Category": r.category,
            "Severity": r.severity,
            "Likelihood": r.likelihood,
            "Immediacy": r.immediacy,
            "Weighted Score": score
        })
    df = pd.DataFrame(rows)
    max_score = len(inputs)*4
    norm = int(round((total/max_score)*10)) if max_score>0 else 0

    high = [r for r in inputs if r.weighted_score()==4]
    mid = [r for r in inputs if 3<=r.weighted_score()<4]
    counts = Counter()
    for cat,cnt in Counter([r.category for r in high]).items():
        if cnt>=3: counts[cat]+=1
    for cat,cnt in Counter([r.category for r in mid]).items():
        if cnt>=5: counts[cat]+=1
    quals = [c for c,cnt in counts.items() if cnt>=1]
    cluster = 2 if len(quals)>=3 else 1 if quals else 0

    bonus = 1 if critical_alert and total>0 else 0
    st.markdown("---")
    st.markdown("**Debug Info:**")
    st.markdown(f"Total Score: {total}")
    st.markdown(f"Normalized Score: {norm}")
    st.markdown(f"Cluster Bonus: {cluster}")
    st.markdown(f"Critical Alert Bonus: {bonus}")

    raw = norm+cluster+bonus
    capped = min(raw,10)
    frac = capped-math.floor(capped)
    final = math.ceil(capped) if frac>=0.6 else math.floor(capped)
    return df,total,final,bonus


def advice_matrix(score:int):
    mapping={
        0:{"Low":"NA","Moderate":"NA","High":"NA"},
        1:{"Low":"Normal Precautions","Moderate":"Normal Precautions","High":"Normal Precautions"},
        2:{"Low":"Normal Precautions","Moderate":"Normal Precautions","High":"Normal Precautions"},
        3:{"Low":"Normal Precautions","Moderate":"Normal Precautions","High":"Normal Precautions"},
        4:{"Low":"Heightened Vigilance","Moderate":"Normal Precautions","High":"Normal Precautions"},
        5:{"Low":"Heightened Vigilance","Moderate":"Heightened Vigilance","High":"Normal Precautions"},
        6:{"Low":"Heightened Vigilance","Moderate":"Heightened Vigilance","High":"Heightened Vigilance"},
        7:{"Low":"Consultation Recommended","Moderate":"Heightened Vigilance","High":"Heightened Vigilance"},
        8:{"Low":"Consultation Recommended","Moderate":"Consultation Recommended","High":"Heightened Vigilance"},
        9:{"Low":"Proactive Engagement","Moderate":"Consultation Recommended","High":"Consultation Recommended"},
        10:{"Low":"Proactive Engagement","Moderate":"Proactive Engagement","High":"Consultation Recommended"}
    }
    return mapping.get(score,mapping[0])

# Streamlit UI
st.set_page_config(layout="wide")
st.title("AI-Assisted Risk Model & Advice Matrix")

if "scenario_text" not in st.session_state:
    st.session_state.update({
        "scenario_text":"",
        "critical_alert":False,
        "session_id":None,
        "risks":[],
        "deleted":set(),
        "new_entries":[],
        "show_editor":False
    })

st.session_state["scenario_text"] = st.text_area("Enter Threat Scenario",st.session_state["scenario_text"])
st.session_state["critical_alert"] = st.checkbox("Source is a Critical Severity Crisis24 Alert",value=st.session_state["critical_alert"])

if st.button("Analyze Scenario"):
    st.session_state.update({
        "session_id":str(uuid4()),
        "deleted":set(),
        "new_entries":[],
        "show_editor":False
    })
    risks = gpt_extract_risks(st.session_state["scenario_text"])
    if risks:
        st.session_state["risks"]=risks
        st.session_state["show_editor"]=True
    else:
        st.error("No risks identified. Please revise input.")

if st.session_state["show_editor"]:
    cats=["Threat Environment","Operational Disruption","Health & Medical Risk","Life Safety Risk","Strategic Risk Indicators","Infrastructure & Resource Stability"]
    st.subheader("Mapped Risks and Scores")
    edited=[]
    for i,r in enumerate(st.session_state["risks"]):
        if i in st.session_state["deleted"]: continue
        c=st.columns([2,2,1,1,1,0.5])
        name=c[0].text_input("Scenario",value=r.name,key=f"name_{i}")
        cat=c[1].selectbox("Risk Category",cats,index=cats.index(r.category),key=f"cat_{i}")
        sev=c[2].selectbox("Severity",[0,1,2],index=r.severity,key=f"sev_{i}")
        lik=c[3].selectbox("Likelihood",[0,1,2],index=r.likelihood,key=f"lik_{i}")
        imm=c[4].selectbox("Immediacy",[0,1,2],index=r.immediacy,key=f"imm_{i}")
        if c[5].button("🗑️",key=f"del_{i}"):
            st.session_state["deleted"].add(i)
            st.experimental_rerun()
        else:
            edited.append(RiskInput(name,sev,lik,imm,cat))
    st.markdown("---")
    for j,ne in enumerate(st.session_state["new_entries"]):
        c=st.columns([2,2,1,1,1,0.5])
        name=c[0].text_input("Scenario",value=ne.name,key=f"new_name_{j}")
        cat=c[1].selectbox("Risk Category",cats,index=cats.index(ne.category),key=f"new_cat_{j}")
        sev=c[2].selectbox("Severity",[0,1,2],index=ne.severity,key=f"new_sev_{j}")
        lik=c[3].selectbox("Likelihood",[0,1,2],index=ne.likelihood,key=f"new_lik_{j}")
        imm=c[4].selectbox("Immediacy",[0,1,2],index=ne.immediacy,key=f"new_imm_{j}")
        if c[5].button("🗑️",key=f"new_del_{j}"):
            st.session_state["new_entries"].pop(j)
            st.experimental_rerun()
        else:
            st.session_state["new_entries"][j]=RiskInput(name,sev,lik,imm,cat)
    if st.button("➕ Add Scenario"):
        st.session_state["new_entries"].append(RiskInput("",0,0,0,cats[0]))
        st.experimental_rerun()

    inputs=edited+st.session_state["new_entries"]
    df,total,final,bonus=calculate_risk_summary(inputs,st.session_state["critical_alert"])
    st.markdown("**Scores:**")
    st.markdown(f"**Aggregated Risk Score:** {total}")
    st.markdown(f"**Assessed Risk Score (1–10):** {final}")
    advice=advice_matrix(final)
    for lvl,adv in advice.items():
        st.markdown(f"**Advice for {lvl} Exposure:** {adv}")
    if bonus:
        st.markdown(f"**Critical Alert Bonus Applied:** +{bonus}")
