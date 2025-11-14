import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go

# Load
df = pd.read_csv("aggregated_data.csv")

# Convert columns
df["saved_at"] = pd.to_datetime(df["saved_at"], errors="coerce")
df["scores"] = pd.to_numeric(df["scores"], errors="coerce")

# Drop invalid rows
df = df.dropna(subset=["saved_at", "scores"])

# Group by date to get average sentiment per day
daily = df.groupby(df["saved_at"].dt.date)["scores"].mean().reset_index()
daily.columns = ["ds", "y"]

print(f"✅ Loaded {len(daily)} days of data:")
print(daily)

# Fit Prophet model
m = Prophet()
m.fit(daily)

# Forecast next 7 days
future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)

# Plot
fig = go.Figure()

# Actual Sentiment
fig.add_trace(go.Scatter(
    x=daily["ds"], y=daily["y"],
    mode="lines+markers",
    name="Actual Sentiment",
    line=dict(color="black")
))

# Forecast Line
fig.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat"],
    mode="lines",
    name="Forecast",
    line=dict(color="green")
))

# Confidence interval shading
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_upper"],
    mode="lines",
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_lower"],
    mode="lines",
    fill='tonexty',
    fillcolor='rgba(0,255,0,0.15)',
    line=dict(width=0),
    name="Confidence Range"
))

fig.update_layout(
    title="Sentiment Forecast Using Prophet (Next 7 Days)",
    xaxis_title="Date",
    yaxis_title="Average Sentiment Score",
    template="plotly_white"
)

fig.show()
