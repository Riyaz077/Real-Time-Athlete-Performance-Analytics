"""Streamlit dashboard for athlete sensor analytics demo/results."""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Athlete Performance Analytics",page_icon="🏃",layout="wide")
st.title("Real-Time Athlete Performance Analytics")
st.caption("Sensor monitoring • Random Forest performance estimation • Isolation Forest anomaly detection")

path=Path("results/athlete_analytics_output.csv")
uploaded=st.sidebar.file_uploader("Upload processed sensor analytics CSV",type=["csv"])
if uploaded is not None:
    df=pd.read_csv(uploaded)
elif path.exists():
    df=pd.read_csv(path,parse_dates=["timestamp"])
else:
    st.info("No processed data yet. Run `python -m src.pipeline --generate-demo` from the project root.")
    st.stop()

required={"athlete_id","heart_rate","speed","acceleration_magnitude","timestamp"}
if not required.issubset(df.columns):
    st.error(f"Missing columns: {required-set(df.columns)}")
    st.stop()
if "timestamp" in df:
    df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
athletes=["All"]+sorted(df.athlete_id.dropna().unique().tolist())
athlete=st.sidebar.selectbox("Athlete",athletes)
view=df if athlete=="All" else df[df.athlete_id==athlete]
c1,c2,c3,c4=st.columns(4)
c1.metric("Athletes",view.athlete_id.nunique())
c2.metric("Sensor records",f"{len(view):,}")
c3.metric("Anomalies",int(view.get("is_anomaly",pd.Series(False,index=view.index)).astype(bool).sum()))
c4.metric("Alerts",int((view.get("alert_severity",pd.Series("None",index=view.index))!="None").sum()))
tab1,tab2,tab3,tab4=st.tabs(["Overview","Performance","Anomalies","Alerts"])
with tab1:
    st.subheader("Sensor trends")
    metric=st.selectbox("Sensor metric",["heart_rate","speed","acceleration_magnitude"])
    chart=view.sort_values("timestamp")
    st.plotly_chart(px.line(chart,x="timestamp",y=metric,color="athlete_id",title=f"{metric} over time"),use_container_width=True)
with tab2:
    st.subheader("Performance score estimation")
    if "performance_score" in view and "predicted_performance_score" in view:
        st.plotly_chart(px.scatter(view,x="performance_score",y="predicted_performance_score",color="athlete_id",
                                   title="Actual/proxy vs predicted score"),use_container_width=True)
    else:
        st.warning("Performance target or predictions are not available in this file.")
with tab3:
    st.subheader("Isolation Forest anomaly events")
    if "is_anomaly" in view:
        anomalies=view[view["is_anomaly"].astype(bool)]
        st.dataframe(anomalies.head(500),use_container_width=True)
        st.plotly_chart(px.scatter(view,x="timestamp",y="heart_rate",color=view["is_anomaly"].astype(str),
                                   title="Anomaly flags over heart-rate timeline"),use_container_width=True)
    else:
        st.info("No anomaly flag column found.")
with tab4:
    st.subheader("Rule-based alerts")
    if "alert_types" in view:
        alerts=view[view["alert_types"].fillna("")!=""]
        st.dataframe(alerts.head(500),use_container_width=True)
        st.download_button("Download alerts CSV",alerts.to_csv(index=False).encode(),file_name="athlete_alerts.csv")
    else:
        st.info("No alert records found.")
st.caption("Educational prototype only. Statistical anomalies are not diagnoses or proof of injury.")
