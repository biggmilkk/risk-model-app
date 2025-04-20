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
    """
    Sends scenario_text to GPT prompt, parses JSON response into RiskInput objects.
    Strips markdown fences if present.
    """
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
    # Remove markdown code fences
    if content.startswith("```"):
        content = content.strip('`')
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        st.error("Failed to parse GPT response as JSON.")
        st.code(content, language="json")
        return []

    risks: list[RiskInput] = []
    for entry in parsed:
        if not all(k in entry for k in ("name","severity","likelihood","immediacy","category")):
            st.warning(f"Skipping invalid entry: {entry}")
            continue
        try:
            risks.append(RiskInput(
                entry["name"],
                int(entry["severity"]),
                int(entry["likelihood"]),
                int(entry["immediacy"]),
                entry["category"]
            ))
        except Exception as e:
            st.warning(f"Error parsing entry {entry}: {e}")
    return risks


def calculate_risk_summary(inputs: list[RiskInput], critical_alert: bool=False):
    rows=[]
    total=0
    for r in inputs:
        sc=r.weighted_score()
        total+=sc
        rows.append({
            "Scenario":r.name,
            "Risk Category":r.category,
            "Severity":r.severity,
            "Likelihood":r.likelihood,
            "Immediacy":r.immediacy,
            "Weighted Score":sc
        })
    df=pd.DataFrame(rows)
    # Normalization
    max_score=len(inputs)*4
    normalized=int(round((total/max_score)*10)) if max_score else 0
    # Clustering
    high=[r.category for r in inputs if r.weighted_score()==4]
    mid=[r.category for r in inputs if 3<=r.weighted_score()<4]
    counts=Counter(high)
    counts.update({cat:cnt for cat,cnt in Counter(mid).items() if cnt>=5})
    qualifies=[cat for cat,cnt in counts.items() if cnt>=1]
    cluster_bonus=2 if len(qualifies)>=3 else 1 if qualifies else 0
    severity_bonus=1 if critical_alert and total>0 else 0
    # Debug
    st.markdown("---")
    st.markdown("**Debug Info:**")
    st.markdown(f"Total Score: {total}")
    st.markdown(f"Normalized Score: {normalized}")
    st.markdown(f"Cluster Bonus: {cluster_bonus}")
    st.markdown(f"Critical Alert Bonus: {severity_bonus}")
    # Final score with custom rounding
    raw=normalized+cluster_bonus+severity_bonus
    capped=min(raw,10)
    final=math.ceil(capped) if capped-math.floor(capped)>=0.6 else math.floor(capped)
    return df,total,final,severity_bonus


def advice_matrix(score:int):
    table={
        0:{"Low":"NA","Moderate":"NA","High":"NA"},
        1:{"Low":"Normal Precautions","Moderate":"Normal Precautions","High":"Normal Precautions"},
        2:{"Low":"Normal Precautions","Moderate":"Normal Precautions","High":"Normal Precautions"},
        3:{"Low":"Normal Precautions","Moderate":"Normal Precautions","High":"Normal Precautions"},
        4:{"Low":"Heightened Vigilance","Moderate":"Normal Precautions","High":"Normal Precautions"},
        5:{"Low":"Heightened Vigilance","Moderate":"Heightened Vigilance","High":"Normal Precautions"},
        6:{"Low":"Heightened Vigilance","Moderate":"Heightened Vigilance","High":"Heightened Vigilance"},
        7:{"Low":"Crisis24 Consultation Recommended","Moderate":"Heightened Vigilance","High":"Heightened Vigilance"},
        8:{"Low":"Crisis24 Consultation Recommended","Moderate":"Crisis24 Consultation Recommended","High":"Heightened Vigilance"},
        9:{"Low":"Crisis24 Proactive Engagement","Moderate":"Crisis24 Consultation Recommended","High":"Crisis24 Consultation Recommended"},
        10:{"Low":"Crisis24 Proactive Engagement","Moderate":"Crisis24 Proactive Engagement","High":"Crisis24 Consultation Recommended"}
    }
    return table.get(score,table[0])

# UI
st.set_page_config(layout="wide")
st.title("AI-Assisted Risk Model & Advice Matrix")
# State
if not st.session_state.get("init"):
    st.session_state["scenario_text"]=""
    st.session_state["critical_alert"]=False
    st.session_state["risks"]=[]
    st.session_state["deleted"]=set()
    st.session_state["new_entries"]=[]
    st.session_state["show"]=False
    st.session_state["init"]=True
# Inputs
st.session_state["scenario_text"]=st.text_area("Enter Threat Scenario",st.session_state["scenario_text"])
st.session_state["critical_alert"]=st.checkbox("Source is a Critical Severity Crisis24 Alert",value=st.session_state["critical_alert"])
if st.button("Analyze Scenario"):
    st.session_state["risks"]=[]
    st.session_state["deleted"]=set()
    st.session_state["new_entries"]=[]
    st.session_state["show"]=False
    r=gpt_extract_risks(st.session_state["scenario_text"])
    if r:
        st.session_state["risks"]=r
        st.session_state["show"]=True
    else:
        st.error("No risks identified. Please revise input.")
# Editor
if st.session_state["show"]:
    cats=["Threat Environment","Operational Disruption","Health & Medical Risk","Life Safety Risk","Strategic Risk Indicators","Infrastructure & Resource Stability"]
    st.subheader("Mapped Risks")
    ed=[]
    for i,r in enumerate(st.session_state["risks"]):
        if i in st.session_state["deleted"]: continue
        cols=st.columns([2,2,1,1,1,0.5])
        name=cols[0].text_input("Scenario",value=r.name,key=f"name_{i}")
        cat=cols[1].selectbox("Category",cats,index=cats.index(r.category),key=f"cat_{i}")
        sev=cols[2].selectbox("Severity",[0,1,2],index=r.severity,key=f"sev_{i}")
        lik=cols[3].selectbox("Likelihood",[0,1,2],index=r.likelihood,key=f"lik_{i}")
        imm=cols[4].selectbox("Immediacy",[0,1,2],index=r.immediacy,key=f"imm_{i}")
        if cols[5].button("🗑️",key=f"del_{i}"):
            st.session_state["deleted"].add(i)
            st.experimental_rerun()
        else:
            ed.append(RiskInput(name,sev,lik,imm,cat))
    st.markdown("---")
    for j,ne in enumerate(st.session_state["new_entries"]):
        cols=st.columns([2,2,1,1,1,0.5])
        name=cols[0].text_input("Scenario",value=ne.name,key=f"new_name_{j}")
        cat=cols[1].selectbox("Category",cats,index=cats.index(ne.category),key=f"new_cat_{j}")
        sev=cols[2].selectbox("Severity",[0,1,2],index=ne.severity,key=f"new_sev_{j}")
        lik=cols[3].selectbox("Likelihood",[0,1,2],index=ne.likelihood,key=f"new_lik_{j}")
        imm=cols[4].selectbox("Immediacy",[0,1,2],index=ne.immediacy,key=f"new_imm_{j}")
        if cols[5].button("🗑️",key=f"new_del_{j}"):
            st.session_state["new_entries"].pop(j)
            st.experimental_rerun()
        else:
            st.session_state["new_entries"][j]=RiskInput(name,sev,lik,imm,cat)
    if st.button("➕ Add Scenario"):
        st.session_state["new_entries"].append(RiskInput("",0,0,0,cats[0]))
        st.experimental_rerun()
    # Summary
    df,total,final,bonus=calculate_risk_summary(ed+st.session_state["new_entries"],st.session_state["critical_alert"])
    st.markdown("**Scores:**")
    st.markdown(f"**Aggregated Risk Score:** {total}")
    st.markdown(f"**Assessed Risk Score (1–10):** {final}")
    adv=advice_matrix(final)
    for lvl,a in adv.items(): st.markdown(f"**Advice for {lvl} Exposure:** {a}")
    if bonus: st.markdown(f"**Critical Alert Bonus Applied:** +{bonus}")
