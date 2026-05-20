"""Charon dashboard — orbit visualization and mission overview.

Run with:
    python dashboard/app.py

Then open http://127.0.0.1:8050 in your browser.
"""

import sys
sys.path.insert(0, ".")

from datetime import datetime, timezone, timedelta
import numpy as np

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

from core.spacecraft import Spacecraft
from mission.target import target_from_tle
from mission.sequence import Sequence


# ------------------------------------------------------------------
# Sample data — replace with your own TLEs
# ------------------------------------------------------------------

ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00001234  00000+0  12345-4 0  9990
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815764429440"""

STARLINK_TLE = """STARLINK-1234
1 45012U 20001A   24001.50000000  .00001000  00000+0  10000-4 0  9991
2 45012  53.0000 100.0000 0001000  90.0000 270.0000 15.06000000000001"""

STARLINK_TLE_2 = """STARLINK-5678
1 45013U 20001B   24001.50000000  .00001100  00000+0  11000-4 0  9992
2 45013  53.0000 120.0000 0001200  80.0000 280.0000 15.07000000000001"""

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
TARGETS = [
    target_from_tle(ISS_TLE,       fuel_needed=120.0, priority=1),
    target_from_tle(STARLINK_TLE,  fuel_needed=80.0,  priority=2),
    target_from_tle(STARLINK_TLE_2,fuel_needed=60.0,  priority=3),
]
SPACECRAFT = Spacecraft(dry_mass=500.0, fuel_mass=3000.0, isp=310.0, name="Charon-1")
SEQUENCE   = Sequence(targets=TARGETS, spacecraft=SPACECRAFT, t_start=T0)
RECORDS    = SEQUENCE.evaluate()

COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD", "#BA7517"]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def earth_sphere():
    """Return a Plotly Surface trace for Earth."""
    R = 6371.0
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "#1a3a5c"], [1, "#2a6090"]],
        showscale=False,
        opacity=0.85,
        hoverinfo="skip",
        name="Earth",
    )


def orbit_trace(target, t0, name, color, n_points=120):
    """Sample one full orbital period and return a Scatter3d trace."""
    state0 = target.state_at(t0)
    r = np.linalg.norm(state0.r)
    mu = 398600.4418
    period = 2 * np.pi * np.sqrt(r**3 / mu)  # seconds

    times = [t0 + timedelta(seconds=period * i / n_points) for i in range(n_points + 1)]
    states = target.propagator.propagate_many(times)

    xs = [s.r[0] for s in states]
    ys = [s.r[1] for s in states]
    zs = [s.r[2] for s in states]

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        name=name,
        line=dict(color=color, width=2),
        hovertemplate=f"<b>{name}</b><br>x=%{{x:.0f}} km<br>y=%{{y:.0f}} km<br>z=%{{z:.0f}} km<extra></extra>",
    )


def position_marker(target, t, name, color):
    """Return a Scatter3d marker for the current position of a target."""
    r = target.position_at(t)
    return go.Scatter3d(
        x=[r[0]], y=[r[1]], z=[r[2]],
        mode="markers+text",
        name=name,
        marker=dict(size=6, color=color, symbol="circle"),
        text=[name],
        textposition="top center",
        hovertemplate=f"<b>{name}</b><br>alt={target.altitude_at(t):.0f} km<extra></extra>",
    )


# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------

app = dash.Dash(__name__, title="Charon — Mission Planner")

app.layout = html.Div([

    html.Div([
        html.H1("Charon", style={"margin": "0", "fontSize": "22px", "fontWeight": "500"}),
        html.P("On-Orbit Servicing Mission Planner",
               style={"margin": "0", "fontSize": "13px", "opacity": "0.6"}),
    ], style={"padding": "16px 24px", "borderBottom": "0.5px solid #e0e0e0"}),

    html.Div([

        # Left — 3D orbit view
        html.Div([
            dcc.Graph(id="orbit-plot", style={"height": "520px"},
                      config={"displayModeBar": False}),
        ], style={"flex": "1.6", "minWidth": "0"}),

        # Right — mission metrics
        html.Div([
            html.H3("Mission summary", style={"fontSize": "14px", "fontWeight": "500",
                                               "marginBottom": "12px"}),
            html.Div(id="metrics-panel"),

            html.H3("Visit order", style={"fontSize": "14px", "fontWeight": "500",
                                           "marginTop": "24px", "marginBottom": "12px"}),
            html.Div(id="visit-list"),

            html.H3("Delta-v per leg", style={"fontSize": "14px", "fontWeight": "500",
                                               "marginTop": "24px", "marginBottom": "12px"}),
            dcc.Graph(id="dv-bar", style={"height": "200px"},
                      config={"displayModeBar": False}),

        ], style={"flex": "1", "minWidth": "260px", "padding": "24px",
                  "borderLeft": "0.5px solid #e0e0e0", "overflowY": "auto"}),

    ], style={"display": "flex", "height": "calc(100vh - 65px)"}),

], style={"fontFamily": "system-ui, sans-serif", "height": "100vh", "overflow": "hidden"})


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------

@app.callback(
    Output("orbit-plot",   "figure"),
    Output("metrics-panel","children"),
    Output("visit-list",   "children"),
    Output("dv-bar",       "figure"),
    Input("orbit-plot",    "id"),   # fires once on load
)
def update(_):
    # --- 3D orbit figure ---
    traces = [earth_sphere()]
    for i, target in enumerate(TARGETS):
        color = COLORS[i % len(COLORS)]
        traces.append(orbit_trace(target, T0, target.name, color))
        traces.append(position_marker(target, T0, target.name, color))

    orbit_fig = go.Figure(data=traces)
    orbit_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="#0d1b2a",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            aspectmode="cube",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            font=dict(color="white", size=11),
            x=0.01, y=0.99,
        ),
        showlegend=True,
    )

    # --- Metrics panel ---
    def metric_card(label, value):
        return html.Div([
            html.Div(label, style={"fontSize": "11px", "color": "#888", "marginBottom": "2px"}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "500"}),
        ], style={"background": "#f5f5f5", "borderRadius": "8px",
                  "padding": "10px 14px", "marginBottom": "8px"})

    feasible = SEQUENCE.is_feasible()
    metrics = html.Div([
        metric_card("Total Δv", f"{SEQUENCE.total_dv():.3f} km/s"),
        metric_card("Total fuel", f"{SEQUENCE.total_fuel():.1f} kg"),
        metric_card("Duration", f"{SEQUENCE.total_duration() / 3600:.1f} h"),
        metric_card("Feasible", "✓ Yes" if feasible else "✗ No"),
    ])

    # --- Visit list ---
    visits = html.Div([
        html.Div([
            html.Span(f"{i+1}. ", style={"fontWeight": "500", "color": COLORS[i % len(COLORS)]}),
            html.Span(r.target.name, style={"fontSize": "13px"}),
            html.Span(f"  {r.maneuver.dv_total:.3f} km/s",
                      style={"fontSize": "12px", "color": "#888", "float": "right"}),
        ], style={"padding": "6px 0", "borderBottom": "0.5px solid #eee"})
        for i, r in enumerate(RECORDS)
    ])

    # --- Delta-v bar chart ---
    dv_fig = go.Figure(go.Bar(
        x=[r.target.name for r in RECORDS],
        y=[r.maneuver.dv_total for r in RECORDS],
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(RECORDS))],
        hovertemplate="<b>%{x}</b><br>Δv = %{y:.4f} km/s<extra></extra>",
    ))
    dv_fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="km/s", gridcolor="#eee", tickfont=dict(size=11)),
        xaxis=dict(tickfont=dict(size=11)),
        showlegend=False,
    )

    return orbit_fig, metrics, visits, dv_fig


if __name__ == "__main__":
    app.run(debug=True)