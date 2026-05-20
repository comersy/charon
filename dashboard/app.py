"""Charon — Mission Planner Dashboard.

4-tab interactive dashboard:
    Tab 1 — Orbits & Mission overview
    Tab 2 — Genetic optimizer convergence
    Tab 3 — Rendezvous simulation
    Tab 4 — Mission timeline & fuel budget

Run with:
    python dashboard/app.py

Then open http://127.0.0.1:8050
"""

import sys
sys.path.insert(0, ".")

from datetime import datetime, timezone, timedelta
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px

from core.spacecraft import Spacecraft
from mission.target import target_from_tle
from simulation.timeline import MissionPlanner, MissionTimeline


# ------------------------------------------------------------------
# Sample TLEs — replace with real ones from Celestrak
# ------------------------------------------------------------------

TLES = [
    ("ISS (ZARYA)",
     "1 25544U 98067A   24001.50000000  .00001234  00000+0  12345-4 0  9990",
     "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815764429440",
     120.0, 1),
    ("STARLINK-1234",
     "1 45012U 20001A   24001.50000000  .00001000  00000+0  10000-4 0  9991",
     "2 45012  53.0000 100.0000 0001000  90.0000 270.0000 15.06000000000001",
     80.0, 2),
    ("STARLINK-5678",
     "1 45013U 20001B   24001.50000000  .00001100  00000+0  11000-4 0  9992",
     "2 45013  53.0000 120.0000 0001200  80.0000 280.0000 15.07000000000001",
     60.0, 3),
    ("STARLINK-9999",
     "1 45014U 20001C   24001.50000000  .00001050  00000+0  10500-4 0  9993",
     "2 45014  53.0000 140.0000 0001100  85.0000 275.0000 15.05000000000001",
     50.0, 4),
]

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD", "#BA7517"]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_targets():
    targets = []
    for name, l1, l2, fuel, priority in TLES:
        tle_block = f"{name}\n{l1}\n{l2}"
        targets.append(target_from_tle(tle_block, fuel_needed=fuel, priority=priority))
    return targets


def make_spacecraft():
    return Spacecraft(dry_mass=500.0, fuel_mass=5000.0, isp=310.0, name="Charon-1")


def earth_sphere():
    R = 6371.0
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "#0d2137"], [1, "#1a4a7a"]],
        showscale=False, opacity=0.9,
        hoverinfo="skip", name="Earth",
    )


def orbit_trace(target, t0, name, color, n_points=120):
    state0 = target.state_at(t0)
    r = np.linalg.norm(state0.r)
    period = 2 * np.pi * np.sqrt(r**3 / 398600.4418)
    times = [t0 + timedelta(seconds=period * i / n_points) for i in range(n_points + 1)]
    states = target.propagator.propagate_many(times)
    return go.Scatter3d(
        x=[s.r[0] for s in states],
        y=[s.r[1] for s in states],
        z=[s.r[2] for s in states],
        mode="lines", name=name,
        line=dict(color=color, width=2),
        hovertemplate=f"<b>{name}</b><extra></extra>",
    )


def position_marker(target, t, name, color):
    r = target.position_at(t)
    alt = target.altitude_at(t)
    return go.Scatter3d(
        x=[r[0]], y=[r[1]], z=[r[2]],
        mode="markers+text", name=name,
        marker=dict(size=7, color=color),
        text=[name], textposition="top center",
        textfont=dict(color="white", size=11),
        hovertemplate=f"<b>{name}</b><br>Alt: {alt:.0f} km<extra></extra>",
    )


# ------------------------------------------------------------------
# UI components
# ------------------------------------------------------------------

def metric_card(label, value, color="#378ADD"):
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": "#888", "marginBottom": "4px"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "500", "color": color}),
    ], style={
        "background": "#f7f7f7", "borderRadius": "8px",
        "padding": "12px 16px", "flex": "1", "minWidth": "120px",
    })


def section_title(text):
    return html.H3(text, style={
        "fontSize": "13px", "fontWeight": "500",
        "color": "#444", "marginBottom": "12px", "marginTop": "0",
        "textTransform": "uppercase", "letterSpacing": "0.05em",
    })


# ------------------------------------------------------------------
# App layout
# ------------------------------------------------------------------

app = dash.Dash(__name__, title="Charon — Mission Planner")
app.config.suppress_callback_exceptions = True

HEADER = html.Div([
    html.Div([
        html.Span("CHARON", style={"fontWeight": "600", "fontSize": "16px", "letterSpacing": "0.1em"}),
        html.Span(" — On-Orbit Servicing Mission Planner",
                  style={"fontSize": "13px", "color": "#888", "marginLeft": "8px"}),
    ], style={"display": "flex", "alignItems": "center"}),
    html.Button("▶  Run Mission", id="run-btn",
                style={
                    "background": "#185FA5", "color": "white",
                    "border": "none", "borderRadius": "6px",
                    "padding": "8px 20px", "fontSize": "13px",
                    "cursor": "pointer", "fontWeight": "500",
                }),
], style={
    "display": "flex", "justifyContent": "space-between", "alignItems": "center",
    "padding": "12px 24px", "borderBottom": "0.5px solid #e0e0e0",
    "background": "white", "position": "sticky", "top": "0", "zIndex": "100",
})

TABS = dcc.Tabs(id="tabs", value="tab-orbits", style={"borderBottom": "0.5px solid #e0e0e0"},
    children=[
        dcc.Tab(label="Orbits & Mission", value="tab-orbits",
                style={"fontSize": "13px"}, selected_style={"fontSize": "13px", "fontWeight": "500"}),
        dcc.Tab(label="Optimizer",        value="tab-optimizer",
                style={"fontSize": "13px"}, selected_style={"fontSize": "13px", "fontWeight": "500"}),
        dcc.Tab(label="Rendezvous",       value="tab-rendezvous",
                style={"fontSize": "13px"}, selected_style={"fontSize": "13px", "fontWeight": "500"}),
        dcc.Tab(label="Timeline",         value="tab-timeline",
                style={"fontSize": "13px"}, selected_style={"fontSize": "13px", "fontWeight": "500"}),
    ]
)

app.layout = html.Div([
    HEADER,
    TABS,
    html.Div(id="tab-content", style={"height": "calc(100vh - 96px)", "overflow": "auto"}),
    dcc.Store(id="mission-store"),
    dcc.Loading(id="loading", type="circle", children=html.Div(id="loading-output")),
], style={"fontFamily": "system-ui, -apple-system, sans-serif", "background": "#fafafa"})


# ------------------------------------------------------------------
# Tab 1 — Orbits & Mission
# ------------------------------------------------------------------

def render_orbits(timeline=None):
    targets = make_targets()
    traces = [earth_sphere()]

    for i, target in enumerate(targets):
        color = COLORS[i % len(COLORS)]
        traces.append(orbit_trace(target, T0, target.name, color))
        traces.append(position_marker(target, T0, target.name, color))

    # Draw transfer path if mission ran
    if timeline:
        visit_order = [r.target for r in timeline.visit_records]
        positions = [t.position_at(T0) for t in visit_order]
        depot = np.array([6771.0, 0.0, 0.0])
        path = [depot] + positions
        traces.append(go.Scatter3d(
            x=[p[0] for p in path],
            y=[p[1] for p in path],
            z=[p[2] for p in path],
            mode="lines+markers",
            name="Mission path",
            line=dict(color="white", width=3, dash="dot"),
            marker=dict(size=4, color="white"),
            hoverinfo="skip",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="#0a1628",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            aspectmode="cube",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white", size=11),
                    x=0.01, y=0.99),
        showlegend=True,
    )

    # Metrics
    if timeline:
        metrics = html.Div([
            metric_card("Total Δv", f"{timeline.total_dv:.3f} km/s", "#378ADD"),
            metric_card("Fuel used", f"{timeline.total_fuel:.0f} kg", "#D85A30"),
            metric_card("Duration", f"{timeline.duration_hours:.1f} h", "#1D9E75"),
            metric_card("Docked", f"{timeline.n_docked}/{len(timeline.visit_records)}", "#7F77DD"),
            metric_card("Feasible", "✓ Yes" if timeline.optimization.feasible else "✗ No",
                        "#1D9E75" if timeline.optimization.feasible else "#D85A30"),
        ], style={"display": "flex", "gap": "8px", "padding": "16px 24px",
                  "background": "white", "borderBottom": "0.5px solid #eee"})
    else:
        metrics = html.Div([
            html.P("Press ▶ Run Mission to compute the optimal servicing plan.",
                   style={"color": "#888", "fontSize": "13px", "margin": "0"}),
        ], style={"padding": "16px 24px", "background": "white",
                  "borderBottom": "0.5px solid #eee"})

    return html.Div([
        metrics,
        dcc.Graph(figure=fig, style={"height": "calc(100vh - 200px)"},
                  config={"displayModeBar": False}),
    ])


# ------------------------------------------------------------------
# Tab 2 — Optimizer
# ------------------------------------------------------------------

def render_optimizer(timeline=None):
    if not timeline:
        return html.Div([
            html.P("Run the mission first to see optimizer results.",
                   style={"color": "#888", "fontSize": "13px", "padding": "40px"})
        ])

    opt = timeline.optimization
    history = opt.history

    # Fitness convergence curve
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=list(range(len(history))), y=history,
        mode="lines", name="Best fitness",
        line=dict(color="#378ADD", width=2),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.08)",
    ))
    fig_conv.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Generation", gridcolor="#eee", title_font_size=12),
        yaxis=dict(title="Best Δv (km/s)", gridcolor="#eee", title_font_size=12),
        showlegend=False,
    )

    # Visit order bar
    records = timeline.visit_records
    fig_order = go.Figure(go.Bar(
        x=[r.target.name for r in records],
        y=[r.maneuver.dv_total for r in records],
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(records))],
        text=[f"{r.maneuver.dv_total:.3f}" for r in records],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Δv = %{y:.4f} km/s<extra></extra>",
    ))
    fig_order.update_layout(
        margin=dict(l=40, r=20, t=20, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Δv (km/s)", gridcolor="#eee"),
        xaxis=dict(gridcolor="#eee"),
        showlegend=False,
    )

    # Summary cards
    cards = html.Div([
        metric_card("Generations", str(opt.generations)),
        metric_card("Best Δv", f"{opt.best_dv:.4f} km/s", "#378ADD"),
        metric_card("Initial Δv", f"{history[0]:.4f} km/s", "#888"),
        metric_card("Improvement", f"{(history[0]-history[-1]):.4f} km/s", "#1D9E75"),
    ], style={"display": "flex", "gap": "8px", "marginBottom": "24px"})

    # Optimal order list
    order_list = html.Div([
        html.Div([
            html.Span(f"{i+1}", style={
                "background": COLORS[i % len(COLORS)], "color": "white",
                "borderRadius": "50%", "width": "24px", "height": "24px",
                "display": "inline-flex", "alignItems": "center",
                "justifyContent": "center", "fontSize": "12px",
                "marginRight": "10px", "flexShrink": "0",
            }),
            html.Span(r.target.name, style={"fontSize": "13px", "flex": "1"}),
            html.Span(f"{r.maneuver.dv_total:.3f} km/s",
                      style={"fontSize": "12px", "color": "#888"}),
        ], style={"display": "flex", "alignItems": "center",
                  "padding": "8px 0", "borderBottom": "0.5px solid #eee"})
        for i, r in enumerate(records)
    ])

    return html.Div([
        html.Div([
            html.Div([
                section_title("Convergence"),
                cards,
                dcc.Graph(figure=fig_conv, style={"height": "280px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1.5", "padding": "24px",
                      "background": "white", "borderRadius": "8px",
                      "border": "0.5px solid #eee"}),

            html.Div([
                section_title("Optimal visit order"),
                order_list,
                html.Div(style={"height": "24px"}),
                section_title("Δv per leg"),
                dcc.Graph(figure=fig_order, style={"height": "220px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "padding": "24px",
                      "background": "white", "borderRadius": "8px",
                      "border": "0.5px solid #eee"}),
        ], style={"display": "flex", "gap": "16px"}),
    ], style={"padding": "24px"})


# ------------------------------------------------------------------
# Tab 3 — Rendezvous
# ------------------------------------------------------------------

def render_rendezvous(timeline=None, selected_target=None):
    if not timeline:
        return html.Div([
            html.P("Run the mission first to see rendezvous results.",
                   style={"color": "#888", "fontSize": "13px", "padding": "40px"})
        ])

    target_names = list(timeline.rendezvous.keys())
    selected = selected_target or target_names[0]
    rdv = timeline.rendezvous[selected]
    states = rdv.states

    times_h = [(s.t - states[0].t).total_seconds() / 3600 for s in states]
    ranges  = [s.range for s in states]
    x_vals  = [s.r[0] for s in states]
    z_vals  = [s.r[2] for s in states]

    # Distance over time
    fig_range = go.Figure()
    fig_range.add_trace(go.Scatter(
        x=times_h, y=ranges,
        mode="lines", line=dict(color="#378ADD", width=2),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.08)",
        hovertemplate="t=%{x:.3f} h<br>range=%{y:.4f} km<extra></extra>",
    ))
    fig_range.add_hline(
        y=0.01, line_dash="dash", line_color="#1D9E75",
        annotation_text="Docking threshold (10 m)",
        annotation_font_size=11,
    )
    fig_range.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Time (h)", gridcolor="#eee"),
        yaxis=dict(title="Range (km)", gridcolor="#eee"),
        showlegend=False,
    )

    # LVLH trajectory (x-z plane)
    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=x_vals, y=z_vals,
        mode="lines",
        line=dict(color="#D85A30", width=2),
        name="Trajectory",
        hovertemplate="x=%{x:.3f} km<br>z=%{y:.3f} km<extra></extra>",
    ))
    fig_traj.add_trace(go.Scatter(
        x=[x_vals[0]], y=[z_vals[0]],
        mode="markers", marker=dict(size=10, color="#378ADD"),
        name="Start",
    ))
    fig_traj.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text", marker=dict(size=12, color="#1D9E75", symbol="star"),
        text=["Target"], textposition="top right",
        name="Target",
    ))
    fig_traj.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Along-track x (km)", gridcolor="#eee"),
        yaxis=dict(title="Radial z (km)", gridcolor="#eee"),
        showlegend=True,
        legend=dict(font_size=11),
    )

    # Dropdown
    dropdown = dcc.Dropdown(
        id="rdv-target-dropdown",
        options=[{"label": n, "value": n} for n in target_names],
        value=selected,
        clearable=False,
        style={"fontSize": "13px", "marginBottom": "16px", "width": "280px"},
    )

    # Stats
    success_color = "#1D9E75" if rdv.success else "#D85A30"
    cards = html.Div([
        metric_card("Result", "✓ Docked" if rdv.success else "✗ Failed", success_color),
        metric_card("Final range", f"{rdv.final_range*1000:.1f} m"),
        metric_card("Corrections", str(len(rdv.dv_corrections))),
        metric_card("RDV Δv", f"{rdv.total_dv:.4f} km/s", "#D85A30"),
        metric_card("Steps", str(len(rdv.states))),
    ], style={"display": "flex", "gap": "8px", "marginBottom": "20px"})

    return html.Div([
        html.Div([dropdown, cards], style={"padding": "24px 24px 0"}),
        html.Div([
            html.Div([
                section_title("Range over time"),
                dcc.Graph(figure=fig_range, style={"height": "300px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "padding": "24px", "background": "white",
                      "borderRadius": "8px", "border": "0.5px solid #eee"}),
            html.Div([
                section_title("LVLH trajectory (along-track vs radial)"),
                dcc.Graph(figure=fig_traj, style={"height": "300px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "padding": "24px", "background": "white",
                      "borderRadius": "8px", "border": "0.5px solid #eee"}),
        ], style={"display": "flex", "gap": "16px", "padding": "0 24px 24px"}),
    ])


# ------------------------------------------------------------------
# Tab 4 — Timeline
# ------------------------------------------------------------------

def render_timeline(timeline=None):
    if not timeline:
        return html.Div([
            html.P("Run the mission first to see the timeline.",
                   style={"color": "#888", "fontSize": "13px", "padding": "40px"})
        ])

    events = timeline.events
    records = timeline.visit_records

    # Gantt chart
    gantt_data = []
    t_ref = timeline.t_start
    for i, r in enumerate(records):
        tof_h = r.maneuver.tof / 3600
        t_dep = r.t_arrival - timedelta(seconds=r.maneuver.tof)
        gantt_data.append(dict(
            Task=r.target.name,
            Start=(t_dep - t_ref).total_seconds() / 3600,
            Finish=(r.t_arrival - t_ref).total_seconds() / 3600,
            Phase="Transfer",
            Color=COLORS[i % len(COLORS)],
        ))
        rdv = timeline.rendezvous.get(r.target.name)
        if rdv:
            rdv_dur = len(rdv.states) * 10 / 3600
            gantt_data.append(dict(
                Task=r.target.name,
                Start=(r.t_arrival - t_ref).total_seconds() / 3600,
                Finish=(r.t_arrival - t_ref).total_seconds() / 3600 + rdv_dur,
                Phase="Rendezvous",
                Color=COLORS[i % len(COLORS)],
            ))

    fig_gantt = go.Figure()
    phases = {"Transfer": 0.3, "Rendezvous": 0.7}
    for item in gantt_data:
        opacity = phases.get(item["Phase"], 0.5)
        fig_gantt.add_trace(go.Bar(
            x=[item["Finish"] - item["Start"]],
            y=[item["Task"]],
            base=[item["Start"]],
            orientation="h",
            marker=dict(color=item["Color"], opacity=opacity),
            name=item["Phase"],
            showlegend=False,
            hovertemplate=(
                f"<b>{item['Task']}</b><br>"
                f"Phase: {item['Phase']}<br>"
                f"Start: {item['Start']:.2f} h<br>"
                f"End: {item['Finish']:.2f} h<extra></extra>"
            ),
        ))
    fig_gantt.update_layout(
        barmode="overlay",
        margin=dict(l=120, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Mission elapsed time (h)", gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee"),
        height=220,
    )

    # Fuel budget waterfall
    fuel_labels = ["Initial"]
    fuel_values = [timeline.spacecraft._initial_fuel]
    fuel_colors = ["#378ADD"]

    remaining = timeline.spacecraft._initial_fuel
    for r in records:
        fuel_labels.append(r.target.name)
        fuel_values.append(-r.fuel_used)
        fuel_colors.append("#D85A30")
        remaining -= r.fuel_used

    fuel_labels.append("Remaining")
    fuel_values.append(remaining)
    fuel_colors.append("#1D9E75")

    fig_fuel = go.Figure(go.Waterfall(
        x=fuel_labels,
        y=fuel_values,
        measure=["absolute"] + ["relative"] * len(records) + ["total"],
        connector=dict(line=dict(color="#ccc", width=1)),
        decreasing=dict(marker_color="#D85A30"),
        increasing=dict(marker_color="#1D9E75"),
        totals=dict(marker_color="#378ADD"),
        hovertemplate="%{x}<br>%{y:.1f} kg<extra></extra>",
    ))
    fig_fuel.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Fuel (kg)", gridcolor="#eee"),
        xaxis=dict(gridcolor="#eee"),
        showlegend=False,
        height=260,
    )

    # Event log
    kind_colors = {
        "depart": "#7F77DD", "transfer": "#378ADD",
        "rendezvous": "#D85A30", "dock": "#1D9E75",
    }
    event_log = html.Div([
        html.Div([
            html.Span(e.kind.upper(), style={
                "fontSize": "10px", "fontWeight": "600",
                "color": kind_colors.get(e.kind, "#888"),
                "width": "80px", "display": "inline-block",
            }),
            html.Span(e.t.strftime("%H:%M"), style={
                "fontSize": "12px", "color": "#888",
                "width": "50px", "display": "inline-block",
            }),
            html.Span(e.target_name, style={
                "fontSize": "12px", "fontWeight": "500",
                "width": "160px", "display": "inline-block",
            }),
            html.Span(e.description, style={"fontSize": "12px", "color": "#666"}),
            html.Span(
                f"  Δv={e.dv:.3f}" if e.dv > 0 else "",
                style={"fontSize": "11px", "color": "#888", "marginLeft": "8px"},
            ),
        ], style={
            "padding": "6px 0", "borderBottom": "0.5px solid #f0f0f0",
            "display": "flex", "alignItems": "center",
        })
        for e in sorted(events, key=lambda x: x.t)
    ], style={"maxHeight": "200px", "overflowY": "auto"})

    return html.Div([
        html.Div([
            html.Div([
                section_title("Mission Gantt"),
                dcc.Graph(figure=fig_gantt, style={"height": "220px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "padding": "24px", "background": "white",
                      "borderRadius": "8px", "border": "0.5px solid #eee"}),
            html.Div([
                section_title("Fuel budget"),
                dcc.Graph(figure=fig_fuel, style={"height": "260px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "padding": "24px", "background": "white",
                      "borderRadius": "8px", "border": "0.5px solid #eee"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        html.Div([
            section_title("Event log"),
            event_log,
        ], style={"padding": "24px", "background": "white",
                  "borderRadius": "8px", "border": "0.5px solid #eee"}),
    ], style={"padding": "24px"})


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------

@app.callback(
    Output("mission-store", "data"),
    Output("loading-output", "children"),
    Input("run-btn", "n_clicks"),
    prevent_initial_call=True,
)
def run_mission(n_clicks):
    if not n_clicks:
        return None, ""

    targets = make_targets()
    sc = make_spacecraft()

    planner = MissionPlanner(
        targets=targets,
        spacecraft=sc,
        t_start=T0,
        pop_size=40,
        n_generations=60,
        seed=42,
    )
    timeline = planner.run()

    # Serialize to JSON-safe dict
    data = {
        "total_dv": timeline.total_dv,
        "total_fuel": timeline.total_fuel,
        "duration_hours": timeline.duration_hours,
        "n_docked": timeline.n_docked,
        "n_targets": len(timeline.visit_records),
        "feasible": timeline.optimization.feasible,
        "opt_history": timeline.optimization.history,
        "opt_generations": timeline.optimization.generations,
        "opt_best_dv": timeline.optimization.best_dv,
        "visit_records": [
            {
                "name": r.target.name,
                "dv": r.maneuver.dv_total,
                "fuel_used": r.fuel_used,
                "tof": r.maneuver.tof,
                "t_arrival": r.t_arrival.isoformat(),
            }
            for r in timeline.visit_records
        ],
        "rendezvous": {
            name: {
                "success": rdv.success,
                "final_range": rdv.final_range,
                "total_dv": rdv.total_dv,
                "n_corrections": len(rdv.dv_corrections),
                "n_states": len(rdv.states),
                "times_h": [(s.t - rdv.states[0].t).total_seconds() / 3600
                            for s in rdv.states],
                "ranges": [s.range for s in rdv.states],
                "x_vals": [float(s.r[0]) for s in rdv.states],
                "z_vals": [float(s.r[2]) for s in rdv.states],
            }
            for name, rdv in timeline.rendezvous.items()
        },
        "events": [
            {
                "t": e.t.isoformat(),
                "kind": e.kind,
                "target_name": e.target_name,
                "description": e.description,
                "dv": e.dv,
                "fuel_used": e.fuel_used,
            }
            for e in timeline.events
        ],
        "fuel_initial": sc._initial_fuel,
        "fuel_remaining": sc.fuel_remaining,
    }
    return data, ""


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("mission-store", "data"),
)
def render_tab(tab, data):
    if tab == "tab-orbits":
        return render_orbits_from_data(data)
    elif tab == "tab-optimizer":
        return render_optimizer_from_data(data)
    elif tab == "tab-rendezvous":
        return render_rendezvous_from_data(data, None)
    elif tab == "tab-timeline":
        return render_timeline_from_data(data)
    return html.Div()

@app.callback(
    Output("tab-content", "children", allow_duplicate=True),
    Input("rdv-target-dropdown", "value"),
    State("mission-store", "data"),
    prevent_initial_call=True,
)
def update_rdv_target(selected, data):
    return render_rendezvous_from_data(data, selected)


# ------------------------------------------------------------------
# Render functions from serialized data
# ------------------------------------------------------------------

def render_orbits_from_data(data):
    targets = make_targets()
    traces = [earth_sphere()]
    for i, target in enumerate(targets):
        color = COLORS[i % len(COLORS)]
        traces.append(orbit_trace(target, T0, target.name, color))
        traces.append(position_marker(target, T0, target.name, color))

    if data:
        records = data["visit_records"]
        positions = [make_targets()[next(j for j, t in enumerate(make_targets())
                     if t.name == r["name"])].position_at(T0)
                     for r in records]
        depot = np.array([6771.0, 0.0, 0.0])
        path = [depot] + positions
        traces.append(go.Scatter3d(
            x=[p[0] for p in path], y=[p[1] for p in path], z=[p[2] for p in path],
            mode="lines+markers", name="Mission path",
            line=dict(color="white", width=3, dash="dot"),
            marker=dict(size=4, color="white"),
            hoverinfo="skip",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="#0a1628",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            aspectmode="cube",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white", size=11),
                    x=0.01, y=0.99),
    )

    if data:
        metrics = html.Div([
            metric_card("Total Δv", f"{data['total_dv']:.3f} km/s", "#378ADD"),
            metric_card("Fuel used", f"{data['total_fuel']:.0f} kg", "#D85A30"),
            metric_card("Duration", f"{data['duration_hours']:.1f} h", "#1D9E75"),
            metric_card("Docked", f"{data['n_docked']}/{data['n_targets']}", "#7F77DD"),
            metric_card("Feasible", "✓ Yes" if data["feasible"] else "✗ No",
                        "#1D9E75" if data["feasible"] else "#D85A30"),
        ], style={"display": "flex", "gap": "8px", "padding": "16px 24px",
                  "background": "white", "borderBottom": "0.5px solid #eee"})
    else:
        metrics = html.Div(
            html.P("Press ▶ Run Mission to compute the optimal servicing plan.",
                   style={"color": "#888", "fontSize": "13px", "margin": "0"}),
            style={"padding": "16px 24px", "background": "white",
                   "borderBottom": "0.5px solid #eee"},
        )

    return html.Div([
        metrics,
        dcc.Graph(figure=fig, style={"height": "calc(100vh - 200px)"},
                  config={"displayModeBar": False}),
    ])


def render_optimizer_from_data(data):
    if not data:
        return html.Div(html.P("Run the mission first.",
                               style={"color": "#888", "padding": "40px", "fontSize": "13px"}))

    history = data["opt_history"]
    records = data["visit_records"]

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=list(range(len(history))), y=history,
        mode="lines", line=dict(color="#378ADD", width=2),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.08)",
    ))
    fig_conv.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Generation", gridcolor="#eee"),
        yaxis=dict(title="Best Δv (km/s)", gridcolor="#eee"),
        showlegend=False,
    )

    fig_order = go.Figure(go.Bar(
        x=[r["name"] for r in records],
        y=[r["dv"] for r in records],
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(records))],
        text=[f"{r['dv']:.3f}" for r in records],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Δv = %{y:.4f} km/s<extra></extra>",
    ))
    fig_order.update_layout(
        margin=dict(l=40, r=20, t=20, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Δv (km/s)", gridcolor="#eee"),
        xaxis=dict(gridcolor="#eee"),
        showlegend=False,
    )

    cards = html.Div([
        metric_card("Generations", str(data["opt_generations"])),
        metric_card("Best Δv", f"{data['opt_best_dv']:.4f} km/s", "#378ADD"),
        metric_card("Initial Δv", f"{history[0]:.4f} km/s", "#888"),
        metric_card("Improvement", f"{history[0]-history[-1]:.4f} km/s", "#1D9E75"),
    ], style={"display": "flex", "gap": "8px", "marginBottom": "20px"})

    order_list = html.Div([
        html.Div([
            html.Span(f"{i+1}", style={
                "background": COLORS[i % len(COLORS)], "color": "white",
                "borderRadius": "50%", "width": "24px", "height": "24px",
                "display": "inline-flex", "alignItems": "center",
                "justifyContent": "center", "fontSize": "12px",
                "marginRight": "10px", "flexShrink": "0",
            }),
            html.Span(r["name"], style={"fontSize": "13px", "flex": "1"}),
            html.Span(f"{r['dv']:.3f} km/s", style={"fontSize": "12px", "color": "#888"}),
        ], style={"display": "flex", "alignItems": "center",
                  "padding": "8px 0", "borderBottom": "0.5px solid #eee"})
        for i, r in enumerate(records)
    ])

    return html.Div([
        html.Div([
            html.Div([
                section_title("Fitness convergence"),
                cards,
                dcc.Graph(figure=fig_conv, style={"height": "280px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1.5", "padding": "24px", "background": "white",
                      "borderRadius": "8px", "border": "0.5px solid #eee"}),
            html.Div([
                section_title("Optimal visit order"),
                order_list,
                html.Div(style={"height": "16px"}),
                section_title("Δv per leg"),
                dcc.Graph(figure=fig_order, style={"height": "220px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "padding": "24px", "background": "white",
                      "borderRadius": "8px", "border": "0.5px solid #eee"}),
        ], style={"display": "flex", "gap": "16px"}),
    ], style={"padding": "24px"})


def render_rendezvous_from_data(data, selected_target=None):
    if not data:
        return html.Div(html.P("Run the mission first.",
                               style={"color": "#888", "padding": "40px", "fontSize": "13px"}))

    rdv_data = data["rendezvous"]
    target_names = list(rdv_data.keys())
    selected = selected_target or target_names[0]
    rdv = rdv_data[selected]

    fig_range = go.Figure()
    fig_range.add_trace(go.Scatter(
        x=rdv["times_h"], y=rdv["ranges"],
        mode="lines", line=dict(color="#378ADD", width=2),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.08)",
    ))
    fig_range.add_hline(y=0.01, line_dash="dash", line_color="#1D9E75",
                        annotation_text="Docking threshold", annotation_font_size=11)
    fig_range.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Time (h)", gridcolor="#eee"),
        yaxis=dict(title="Range (km)", gridcolor="#eee"),
        showlegend=False,
    )

    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=rdv["x_vals"], y=rdv["z_vals"],
        mode="lines", line=dict(color="#D85A30", width=2), name="Trajectory",
    ))
    fig_traj.add_trace(go.Scatter(
        x=[rdv["x_vals"][0]], y=[rdv["z_vals"][0]],
        mode="markers", marker=dict(size=10, color="#378ADD"), name="Start",
    ))
    fig_traj.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text", marker=dict(size=12, color="#1D9E75", symbol="star"),
        text=["Target"], textposition="top right", name="Target",
    ))
    fig_traj.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Along-track x (km)", gridcolor="#eee"),
        yaxis=dict(title="Radial z (km)", gridcolor="#eee"),
    )

    success_color = "#1D9E75" if rdv["success"] else "#D85A30"
    cards = html.Div([
        metric_card("Result", "✓ Docked" if rdv["success"] else "✗ Failed", success_color),
        metric_card("Final range", f"{rdv['final_range']*1000:.1f} m"),
        metric_card("Corrections", str(rdv["n_corrections"])),
        metric_card("RDV Δv", f"{rdv['total_dv']:.4f} km/s", "#D85A30"),
        metric_card("Steps", str(rdv["n_states"])),
    ], style={"display": "flex", "gap": "8px", "marginBottom": "20px"})

    dropdown = dcc.Dropdown(
        id="rdv-target-dropdown",
        options=[{"label": n, "value": n} for n in target_names],
        value=selected,
        clearable=False,
        style={"fontSize": "13px", "marginBottom": "16px", "width": "280px"},
    )

    return html.Div([
        html.Div([dropdown, cards], style={"padding": "24px 24px 0"}),
        html.Div([
            html.Div([
                section_title("Range over time"),
                dcc.Graph(figure=fig_range, style={"height": "300px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "padding": "24px", "background": "white",
                      "borderRadius": "8px", "border": "0.5px solid #eee"}),
            html.Div([
                section_title("LVLH trajectory"),
                dcc.Graph(figure=fig_traj, style={"height": "300px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "padding": "24px", "background": "white",
                      "borderRadius": "8px", "border": "0.5px solid #eee"}),
        ], style={"display": "flex", "gap": "16px", "padding": "0 24px 24px"}),
    ])


def render_timeline_from_data(data):
    if not data:
        return html.Div(html.P("Run the mission first.",
                               style={"color": "#888", "padding": "40px", "fontSize": "13px"}))

    records = data["visit_records"]
    events  = data["events"]
    t_start = datetime.fromisoformat(events[0]["t"])

    # Gantt
    fig_gantt = go.Figure()
    for i, r in enumerate(records):
        t_arr = datetime.fromisoformat(r["t_arrival"])
        t_dep = t_arr - timedelta(seconds=r["tof"])
        start_h = (t_dep - t_start).total_seconds() / 3600
        end_h   = (t_arr - t_start).total_seconds() / 3600
        rdv_dur = data["rendezvous"][r["name"]]["n_states"] * 10 / 3600

        fig_gantt.add_trace(go.Bar(
            x=[end_h - start_h], y=[r["name"]], base=[start_h],
            orientation="h",
            marker=dict(color=COLORS[i % len(COLORS)], opacity=0.4),
            name="Transfer", showlegend=(i == 0),
            hovertemplate=f"<b>{r['name']}</b><br>Transfer<br>{start_h:.2f}h → {end_h:.2f}h<extra></extra>",
        ))
        fig_gantt.add_trace(go.Bar(
            x=[rdv_dur], y=[r["name"]], base=[end_h],
            orientation="h",
            marker=dict(color=COLORS[i % len(COLORS)], opacity=0.9),
            name="Rendezvous", showlegend=(i == 0),
            hovertemplate=f"<b>{r['name']}</b><br>Rendezvous<br>{end_h:.2f}h → {end_h+rdv_dur:.2f}h<extra></extra>",
        ))

    fig_gantt.update_layout(
        barmode="overlay",
        margin=dict(l=120, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Elapsed time (h)", gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee"),
        height=220, showlegend=True,
        legend=dict(font_size=11),
    )

    # Fuel waterfall
    fuel_initial = data["fuel_initial"]
    fig_fuel = go.Figure(go.Waterfall(
        x=["Initial"] + [r["name"] for r in records] + ["Remaining"],
        y=[fuel_initial] + [-r["fuel_used"] for r in records] + [data["fuel_remaining"]],
        measure=["absolute"] + ["relative"] * len(records) + ["total"],
        connector=dict(line=dict(color="#ccc", width=1)),
        decreasing=dict(marker_color="#D85A30"),
        increasing=dict(marker_color="#1D9E75"),
        totals=dict(marker_color="#378ADD"),
        hovertemplate="%{x}<br>%{y:.1f} kg<extra></extra>",
    ))
    fig_fuel.update_layout(
        margin=dict(l=40, r=20, t=20, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Fuel (kg)", gridcolor="#eee"),
        xaxis=dict(gridcolor="#eee"),
        showlegend=False, height=260,
    )

    # Event log
    kind_colors = {
        "depart": "#7F77DD", "transfer": "#378ADD",
        "rendezvous": "#D85A30", "dock": "#1D9E75",
    }
    event_log = html.Div([
        html.Div([
            html.Span(e["kind"].upper(), style={
                "fontSize": "10px", "fontWeight": "600",
                "color": kind_colors.get(e["kind"], "#888"),
                "width": "80px", "display": "inline-block",
            }),
            html.Span(datetime.fromisoformat(e["t"]).strftime("%H:%M"), style={
                "fontSize": "12px", "color": "#888",
                "width": "50px", "display": "inline-block",
            }),
            html.Span(e["target_name"], style={
                "fontSize": "12px", "fontWeight": "500",
                "width": "160px", "display": "inline-block",
            }),
            html.Span(e["description"], style={"fontSize": "12px", "color": "#666"}),
            html.Span(f"  Δv={e['dv']:.3f}" if e["dv"] > 0 else "",
                      style={"fontSize": "11px", "color": "#888", "marginLeft": "8px"}),
        ], style={"padding": "6px 0", "borderBottom": "0.5px solid #f0f0f0",
                  "display": "flex", "alignItems": "center"})
        for e in sorted(events, key=lambda x: x["t"])
    ], style={"maxHeight": "200px", "overflowY": "auto"})

    return html.Div([
        html.Div([
            html.Div([
                section_title("Mission Gantt"),
                dcc.Graph(figure=fig_gantt, style={"height": "220px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "padding": "24px", "background": "white",
                      "borderRadius": "8px", "border": "0.5px solid #eee"}),
            html.Div([
                section_title("Fuel budget"),
                dcc.Graph(figure=fig_fuel, style={"height": "260px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "padding": "24px", "background": "white",
                      "borderRadius": "8px", "border": "0.5px solid #eee"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        html.Div([
            section_title("Event log"),
            event_log,
        ], style={"padding": "24px", "background": "white",
                  "borderRadius": "8px", "border": "0.5px solid #eee"}),
    ], style={"padding": "24px"})


if __name__ == "__main__":
    app.run(debug=True)