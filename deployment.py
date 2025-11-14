# dashboard.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

CSV_FILE = "aggregated_data.csv"

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame(columns=[
            "id","platform","url","title_or_text","source","published",
            "vader_sentiment","scores","llm_sentiment","llm_confidence",
            "llm_summary","saved_at"
        ])
    df = pd.read_csv(path)
    # types
    if "scores" in df.columns:
        df["scores"] = pd.to_numeric(df["scores"], errors="coerce")
    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
    # derived
    df["date"] = df["published"].dt.date
    df["keyword_hint"] = (
        df["title_or_text"].fillna("").str.lower()
    )
    return df.dropna(subset=["published","scores"])

def sentiment_label(x: float) -> str:
    if x >= 0.05: return "Positive"
    if x <= -0.05: return "Negative"
    return "Neutral"

# ---------- Sidebar (Filters) ----------
st.sidebar.title("Filters")
df = load_data(CSV_FILE)

if df.empty:
    st.info("No data yet. Run your collector to create `aggregated_data.csv`.")
    st.stop()

min_date, max_date = df["published"].min().date(), df["published"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
keyword = st.sidebar.text_input("Keyword contains (optional)").strip().lower()
sources = st.sidebar.multiselect(
    "Sources", options=sorted(df["source"].dropna().unique().tolist()),
    default=sorted(df["source"].dropna().unique().tolist())
)

show_only_alerts = st.sidebar.checkbox("Only show Strong Alerts (≤ -0.5 or ≥ 0.5)", value=False)
st.sidebar.write("---")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()

# ---------- Apply filters ----------
mask = (
    (df["date"] >= (date_range[0] if isinstance(date_range, (list, tuple)) else min_date)) &
    (df["date"] <= (date_range[-1] if isinstance(date_range, (list, tuple)) else max_date)) &
    (df["source"].isin(sources))
)
if keyword:
    mask &= df["keyword_hint"].str.contains(keyword)

if show_only_alerts:
    mask &= ((df["scores"] <= -0.5) | (df["scores"] >= 0.5))

fdf = df.loc[mask].copy().sort_values("published")

# ---------- Header ----------
st.title("Market Sentiment")
st.caption("From aggregated_data.csv — VADER scores, optional LLM summary, clean UI.")

# ---------- KPIs ----------
col1, col2, col3, col4 = st.columns(4)
avg_score = fdf["scores"].mean() if not fdf.empty else np.nan
pos_ratio = (fdf["scores"] >= 0.05).mean() if not fdf.empty else np.nan
neg_alerts = int((fdf["scores"] <= -0.5).sum())
pos_alerts = int((fdf["scores"] >= 0.5).sum())

col1.metric("Overall Avg Sentiment", f"{avg_score:.2f}" if not np.isnan(avg_score) else "—",
            sentiment_label(avg_score) if not np.isnan(avg_score) else "")
col2.metric("Positive Articles (≥ 0.05)", f"{int((fdf['scores'] >= 0.05).sum())}")
col3.metric("Strong Positive Alerts (≥ 0.5)", f"{pos_alerts}")
col4.metric("Strong Negative Alerts (≤ -0.5)", f"{neg_alerts}")

st.write("")

# ---------- Sentiment over time (daily average) ----------
st.subheader("Sentiment Over Time")
daily = fdf.set_index("published").resample("D")["scores"].mean().dropna()

if not daily.empty:
    fig = plt.figure()
    plt.plot(daily.index, daily.values)
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment (VADER)")
    plt.title("Daily Average Sentiment")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
else:
    st.info("No points in the selected range to plot.")

st.write("")

# ---------- Sentiment by Source ----------
st.subheader("Sentiment by Source")
src = (
    fdf.groupby("source", dropna=True)["scores"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
if not src.empty:
    fig2 = plt.figure()
    plt.barh(src.index, src.values)
    plt.xlabel("Avg Sentiment")
    plt.ylabel("Source")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)
else:
    st.info("No source data to display.")

st.write("")

# ---------- Alerts (table) ----------
st.subheader("Strong Alerts (±0.5)")
alerts = fdf[(fdf["scores"] <= -0.5) | (fdf["scores"] >= 0.5)].copy()
alerts = alerts.sort_values("published", ascending=False)[
    ["published","source","title_or_text","scores","url"]
]
if alerts.empty:
    st.info("No strong alerts in current filters.")
else:
    st.dataframe(alerts, use_container_width=True, hide_index=True)

# ---------- News Feed ----------
st.subheader("Latest Articles")
feed = fdf.sort_values("published", ascending=False)[
    ["published","source","title_or_text","scores","vader_sentiment","url"]
].head(30)

def color_score(v):
    try:
        v = float(v)
        if v >= 0.05: return f"✅ {v:.2f}"
        if v <= -0.05: return f"❌ {v:.2f}"
        return f"• {v:.2f}"
    except Exception:
        return v

for _, r in feed.iterrows():
    st.markdown(
        f"**{r['title_or_text']}**  \n"
        f"{r['source']} — *{r['published']}*  \n"
        f"Score: {color_score(r['scores'])}  |  "
        f"[Open link]({r['url']})",
        unsafe_allow_html=False
    )

st.write("---")
st.download_button(
    "Download filtered CSV",
    data=fdf.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_filtered.csv",
    mime="text/csv"
)
