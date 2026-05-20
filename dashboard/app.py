"""Charon — Live Mission Planner Dashboard.

Fetches real TLEs from Celestrak, displays live satellite positions,
and simulates an on-orbit servicing mission end-to-end.

Run with:
    python dashboard/app.py
"""

import sys
sys.path.insert(0, ".")

from datetime import datetime, timezone, timedelta
import numpy as np
import requests

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

from core.sgp4_propagator import SGP4Propagator
from mission.target import target_from_tle
from core.spacecraft import Spacecraft
from simulation.timeline import MissionPlanner


# ------------------------------------------------------------------
# Celestrak live fetch
# ------------------------------------------------------------------

CELESTRAK_URLS = {
    "Starlink":  "https://celestrak.org/SOCRATES/query.php?CODE=25544&FORMAT=TLE",
    "Starlink25": "https://celestrak.org/satcat/tle.php?GROUP=starlink&FORMAT=TLE",
    "Stations":  "https://celestrak.org/satcat/tle.php?GROUP=stations&FORMAT=TLE",
}

_tle_cache: dict = {}


def fetch_tles(url: str, max_sats: int = 25) -> list[dict]:
    """Fetch and parse TLEs from Celestrak."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Charon/1.0)"}
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
        sats = []
        i = 0
        while i + 2 < len(lines) and len(sats) < max_sats:
            name  = lines[i]
            line1 = lines[i+1]
            line2 = lines[i+2]
            if line1.startswith("1 ") and line2.startswith("2 "):
                sats.append({"name": name, "line1": line1, "line2": line2})
                i += 3
            else:
                i += 1
        return sats
    except Exception as e:
        print(f"[Celestrak] Fetch failed: {e}")
        return []


def fetch_tles_json(max_sats: int = 25) -> list[dict]:
    """Fetch Starlink TLEs via public TLE API."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Charon/1.0)"}
        # Public mirror, no restrictions
        url = "https://tle.ivanstanojevic.me/api/tle/?search=starlink&page-size=25"
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        sats = []
        for sat in data.get("member", [])[:max_sats]:
            name  = sat.get("name", "UNKNOWN")
            line1 = sat.get("line1", "")
            line2 = sat.get("line2", "")
            if line1 and line2:
                sats.append({"name": name, "line1": line1, "line2": line2})
        return sats
    except Exception as e:
        print(f"[TLE API] Fetch failed: {e}")
        return []


def load_catalog() -> list[dict]:
    """Load ISS + 25 Starlink satellites."""
    global _tle_cache
    if _tle_cache:
        return _tle_cache.get("sats", [])

    sats = []

    iss = fetch_tles(
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=TLE",
        max_sats=1,
    )
    sats.extend(iss)

    starlink = fetch_tles_json(max_sats=25)
    sats.extend(starlink)

    _tle_cache["sats"] = sats
    print(f"[Celestrak] Loaded {len(sats)} satellites.")
    return sats


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

COLORS = [
    "#378ADD", "#1D9E75", "#D85A30", "#7F77DD",
    "#BA7517", "#D4537E", "#639922", "#5DCAA5",
]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def earth_sphere() -> go.Surface:
    R = 6371.0
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 80)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "#0d2137"], [1, "#1a4a7a"]],
        showscale=False, opacity=0.9,
        hoverinfo="skip", name="Earth",
    )


def orbit_trace(prop: SGP4Propagator, t: datetime, name: str, color: str, n: int = 90) -> go.Scatter3d:
    """Compute one full orbital period and return a trace."""
    try:
        state0 = prop.propagate(t)
        r = np.linalg.norm(state0.r)
        period = 2 * np.pi * np.sqrt(r**3 / 398600.4418)
        times = [t + timedelta(seconds=period * i / n) for i in range(n + 1)]
        states = prop.propagate_many(times)
        return go.Scatter3d(
            x=[s.r[0] for s in states],
            y=[s.r[1] for s in states],
            z=[s.r[2] for s in states],
            mode="lines", name=name,
            line=dict(color=color, width=1.5),
            hovertemplate=f"<b>{name}</b><extra></extra>",
            showlegend=False,
        )
    except Exception:
        return None


def sat_marker(prop: SGP4Propagator, t: datetime, name: str, color: str) -> go.Scatter3d:
    """Current position marker for a satellite."""
    try:
        state = prop.propagate(t)
        r = state.r
        alt = state.altitude
        return go.Scatter3d(
            x=[r[0]], y=[r[1]], z=[r[2]],
            mode="markers+text", name=name,
            marker=dict(size=5, color=color, symbol="circle"),
            text=[name.split("(")[0].strip()[:12]],
            textposition="top center",
            textfont=dict(color="white", size=9),
            hovertemplate=f"<b>{name}</b><br>Alt: {alt:.0f} km<extra></extra>",
        )
    except Exception:
        return None


def servicer_marker(r: np.ndarray, name: str = "Charon-1") -> go.Scatter3d:
    return go.Scatter3d(
        x=[r[0]], y=[r[1]], z=[r[2]],
        mode="markers+text", name=name,
        marker=dict(size=9, color="#FFD700", symbol="diamond"),
        text=["🛸 " + name],
        textposition="top center",
        textfont=dict(color="#FFD700", size=11),
        hovertemplate=f"<b>{name}</b><br>Servicer<extra></extra>",
    )


def metric_card(label: str, value: str, color: str = "#378ADD") -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": "#888", "marginBottom": "4px"}),
        html.Div(value, style={"fontSize": "18px", "fontWeight": "500", "color": color}),
    ], style={
        "background": "#f7f7f7", "borderRadius": "8px",
        "padding": "10px 14px", "flex": "1", "minWidth": "100px",
    })


# ------------------------------------------------------------------
# App layout
# ------------------------------------------------------------------

app = dash.Dash(__name__, title="Charon — Live Mission Planner")
app.config.suppress_callback_exceptions = True

app.layout = html.Div([

    # Header
    html.Div([
        html.Div([
            html.Span("CHARON", style={"fontWeight": "700", "fontSize": "15px",
                                        "letterSpacing": "0.12em"}),
            html.Span(" — Live On-Orbit Servicing Planner",
                      style={"fontSize": "12px", "color": "#888", "marginLeft": "10px"}),
        ]),
        html.Div([
            html.Span(id="live-clock",
                      style={"fontSize": "12px", "color": "#888", "marginRight": "16px",
                             "fontFamily": "monospace"}),
            html.Button("▶  Run Mission", id="run-btn", style={
                "background": "#185FA5", "color": "white", "border": "none",
                "borderRadius": "6px", "padding": "8px 18px",
                "fontSize": "13px", "cursor": "pointer", "fontWeight": "500",
            }),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "10px 24px", "borderBottom": "0.5px solid #e0e0e0",
        "background": "white", "position": "sticky", "top": "0", "zIndex": "100",
    }),

    # Main layout
    html.Div([

        # Left sidebar — satellite list
        html.Div([
            html.Div([
                html.Div("Satellites", style={"fontSize": "11px", "fontWeight": "600",
                                            "color": "#444", "letterSpacing": "0.08em",
                                            "marginBottom": "10px", "textTransform": "uppercase"}),
                dcc.Checklist(
                    id="sat-checklist",
                    options=[],
                    value=[],
                    labelStyle={"display": "flex", "alignItems": "center",
                                "fontSize": "12px", "padding": "3px 0", "cursor": "pointer"},
                    inputStyle={"marginRight": "8px"},
                ),
                html.Div(id="sat-checklist-container"),
                html.Div([
                    html.Button("Select all", id="select-all-btn", style={
                        "fontSize": "11px", "padding": "4px 10px", "marginRight": "6px",
                        "border": "0.5px solid #ccc", "borderRadius": "4px",
                        "background": "transparent", "cursor": "pointer",
                    }),
                    html.Button("Clear", id="clear-btn", style={
                        "fontSize": "11px", "padding": "4px 10px",
                        "border": "0.5px solid #ccc", "borderRadius": "4px",
                        "background": "transparent", "cursor": "pointer",
                    }),
                ], style={"marginTop": "10px", "display": "flex"}),
            ], style={"padding": "16px 12px"}),
        ], style={
            "width": "220px", "flexShrink": "0",
            "borderRight": "0.5px solid #e0e0e0",
            "background": "white", "overflowY": "auto",
            "height": "calc(100vh - 45px)",
        }),

        # Center — globe
        html.Div([
            dcc.Graph(
                id="globe",
                style={"height": "calc(100vh - 45px)"},
                config={"displayModeBar": False},
            ),
        ], style={"flex": "1"}),

        # Right panel — mission info
        html.Div([
            dcc.Tabs(id="info-tabs", value="tab-mission", style={"fontSize": "12px"},
                children=[
                    dcc.Tab(label="Mission",    value="tab-mission",
                            style={"fontSize": "12px", "padding": "6px"},
                            selected_style={"fontSize": "12px", "fontWeight": "600", "padding": "6px"}),
                    dcc.Tab(label="Optimizer",  value="tab-optimizer",
                            style={"fontSize": "12px", "padding": "6px"},
                            selected_style={"fontSize": "12px", "fontWeight": "600", "padding": "6px"}),
                    dcc.Tab(label="Rendezvous", value="tab-rendezvous",
                            style={"fontSize": "12px", "padding": "6px"},
                            selected_style={"fontSize": "12px", "fontWeight": "600", "padding": "6px"}),
                    dcc.Tab(label="Timeline",   value="tab-timeline",
                            style={"fontSize": "12px", "padding": "6px"},
                            selected_style={"fontSize": "12px", "fontWeight": "600", "padding": "6px"}),
                ]
            ),
            html.Div(id="info-content", style={
                "overflowY": "auto",
                "height": "calc(100vh - 90px)",
                "padding": "0",
            }),
        ], style={
            "width": "360px", "flexShrink": "0",
            "borderLeft": "0.5px solid #e0e0e0",
            "background": "white",
        }),

    ], style={"display": "flex", "height": "calc(100vh - 45px)"}),

    # Stores & intervals
    dcc.Store(id="catalog-store"),
    dcc.Store(id="mission-store"),
    dcc.Store(id="selected-sats-store"),
    dcc.Interval(id="live-interval", interval=5000, n_intervals=0),  # 5s refresh
    dcc.Interval(id="clock-interval", interval=1000, n_intervals=0), # 1s clock

], style={"fontFamily": "system-ui, -apple-system, sans-serif", "overflow": "hidden"})


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------

@app.callback(
    Output("live-clock", "children"),
    Input("clock-interval", "n_intervals"),
)
def update_clock(_):
    return now_utc().strftime("UTC  %Y-%m-%d  %H:%M:%S")


@app.callback(
    Output("catalog-store", "data"),
    Output("sat-checklist", "options"),
    Output("sat-checklist", "value"),
    Input("live-interval", "n_intervals"),
    State("catalog-store", "data"),
)
def load_satellites(n, existing):
    if existing:
        sats = existing["sats"]
        options = [{"label": s["name"][:24], "value": s["name"]} for s in sats]
        return existing, options, dash.no_update

    sats = load_catalog()
    if not sats:
        return None, [], []

    options = [{"label": s["name"][:24], "value": s["name"]} for s in sats]
    default = [sats[0]["name"]] if sats else []
    return {"sats": sats}, options, default


@app.callback(
    Output("globe", "figure"),
    Input("live-interval", "n_intervals"),
    Input("mission-store", "data"),
    State("catalog-store", "data"),
    State("sat-checklist", "value"),
)
def update_globe(_, mission_data, catalog, selected_names):
    t = now_utc()
    traces = [earth_sphere()]

    if not catalog or not selected_names:
        fig = go.Figure(data=traces)
        _style_globe(fig)
        return fig

    sats = {s["name"]: s for s in catalog["sats"]}

    for i, name in enumerate(selected_names):
        if name not in sats:
            continue
        s = sats[name]
        color = COLORS[i % len(COLORS)]
        try:
            prop = SGP4Propagator(s["line1"], s["line2"], name=name)
            orb = orbit_trace(prop, t, name, color)
            mkr = sat_marker(prop, t, name, color)
            if orb:
                traces.append(orb)
            if mkr:
                traces.append(mkr)
        except Exception as e:
            print(f"[Globe] Error for {name}: {e}")

    # Draw servicer if mission ran
    if mission_data and "servicer_positions" in mission_data:
        positions = mission_data["servicer_positions"]
        if positions:
            # Full transfer path
            traces.append(go.Scatter3d(
                x=[p[0] for p in positions],
                y=[p[1] for p in positions],
                z=[p[2] for p in positions],
                mode="lines",
                name="Transfer path",
                line=dict(color="#FFD700", width=2, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            ))
            # Current servicer position (last point)
            traces.append(servicer_marker(np.array(positions[-1])))

    fig = go.Figure(data=traces)
    _style_globe(fig)
    return fig


def _style_globe(fig):
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="#060e1a",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            aspectmode="cube",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=10)),
        uirevision="globe",  # keeps camera angle on refresh
    )


@app.callback(
    Output("mission-store", "data"),
    Input("run-btn", "n_clicks"),
    State("catalog-store", "data"),
    State("sat-checklist", "value"),
    prevent_initial_call=True,
)
def run_mission(n_clicks, catalog, selected_names):
    if not catalog or not selected_names or len(selected_names) < 2:
        return None

    sats = {s["name"]: s for s in catalog["sats"]}
    t0 = now_utc()

    targets = []
    for i, name in enumerate(selected_names):
        if name not in sats:
            continue
        s = sats[name]
        tle_block = f"{name}\n{s['line1']}\n{s['line2']}"
        try:
            # Validate TLE before adding
            prop = SGP4Propagator(s["line1"], s["line2"], name=name)
            prop.propagate(t0)  # test propagation
            t = target_from_tle(tle_block, fuel_needed=80.0 + i * 10, priority=i + 1)
            targets.append(t)
            print(f"[Mission] Added target: {name}")
        except Exception as e:
            print(f"[Mission] Skipping {name}: {e}")

    if len(targets) < 2:
        return None

    sc = Spacecraft(dry_mass=500.0, fuel_mass=8000.0, isp=310.0, name="Charon-1")

    planner = MissionPlanner(
        targets=targets,
        spacecraft=sc,
        t_start=t0,
        pop_size=30,
        n_generations=50,
        seed=None,
    )
    timeline = planner.run()

    # Build servicer trajectory points for the globe
    R_EARTH = 6371.0
    servicer_positions = [[R_EARTH + 400.0, 0.0, 0.0]]
    for r in timeline.visit_records:
        pos = r.target.position_at(r.t_arrival)
        servicer_positions.append(pos.tolist())

    return {
        "total_dv":    timeline.total_dv,
        "total_fuel":  timeline.total_fuel,
        "duration_h":  timeline.duration_hours,
        "n_docked":    timeline.n_docked,
        "n_targets":   len(timeline.visit_records),
        "feasible":    timeline.optimization.feasible,
        "opt_history": timeline.optimization.history,
        "opt_gens":    timeline.optimization.generations,
        "opt_best_dv": timeline.optimization.best_dv,
        "servicer_positions": servicer_positions,
        "visit_records": [
            {
                "name":      r.target.name,
                "dv":        r.maneuver.dv_total,
                "fuel_used": r.fuel_used,
                "tof_h":     r.maneuver.tof / 3600,
                "t_arrival": r.t_arrival.isoformat(),
            }
            for r in timeline.visit_records
        ],
        "rendezvous": {
            name: {
                "success":       rdv.success,
                "final_range_m": rdv.final_range * 1000,
                "total_dv":      rdv.total_dv,
                "n_corrections": len(rdv.dv_corrections),
                "times_h":  [(s.t - rdv.states[0].t).total_seconds() / 3600 for s in rdv.states],
                "ranges":   [s.range for s in rdv.states],
                "x_vals":   [float(s.r[0]) for s in rdv.states],
                "z_vals":   [float(s.r[2]) for s in rdv.states],
            }
            for name, rdv in timeline.rendezvous.items()
        },
        "events": [
            {
                "t":           e.t.isoformat(),
                "kind":        e.kind,
                "target_name": e.target_name,
                "description": e.description,
                "dv":          e.dv,
            }
            for e in timeline.events
        ],
        "fuel_initial":   8000.0,
        "fuel_remaining": sc.fuel_remaining,
    }


@app.callback(
    Output("info-content", "children"),
    Input("info-tabs", "value"),
    Input("mission-store", "data"),
)
def render_info(tab, data):
    if tab == "tab-mission":
        return _render_mission(data)
    elif tab == "tab-optimizer":
        return _render_optimizer(data)
    elif tab == "tab-rendezvous":
        return _render_rendezvous(data)
    elif tab == "tab-timeline":
        return _render_timeline(data)
    return html.Div()


# ------------------------------------------------------------------
# Info panel renderers
# ------------------------------------------------------------------

def _no_data():
    return html.P("Select satellites and press ▶ Run Mission.",
                  style={"color": "#888", "fontSize": "12px", "padding": "20px"})


def _render_mission(data):
    if not data:
        return _no_data()

    cards = html.Div([
        metric_card("Total Δv",  f"{data['total_dv']:.3f} km/s",  "#378ADD"),
        metric_card("Fuel used", f"{data['total_fuel']:.0f} kg",  "#D85A30"),
    ], style={"display": "flex", "gap": "8px", "padding": "12px"})

    cards2 = html.Div([
        metric_card("Duration", f"{data['duration_h']:.1f} h",              "#1D9E75"),
        metric_card("Docked",   f"{data['n_docked']}/{data['n_targets']}",  "#7F77DD"),
        metric_card("Feasible", "✓" if data["feasible"] else "✗",
                    "#1D9E75" if data["feasible"] else "#D85A30"),
    ], style={"display": "flex", "gap": "8px", "padding": "0 12px 12px"})

    visit_rows = html.Div([
        html.Div([
            html.Span(f"{i+1}.", style={"color": COLORS[i % len(COLORS)],
                                         "fontWeight": "600", "marginRight": "8px",
                                         "fontSize": "12px"}),
            html.Span(r["name"][:22], style={"fontSize": "12px", "flex": "1"}),
            html.Span(f"{r['dv']:.3f} km/s",
                      style={"fontSize": "11px", "color": "#888"}),
        ], style={"display": "flex", "alignItems": "center",
                  "padding": "5px 0", "borderBottom": "0.5px solid #f0f0f0"})
        for i, r in enumerate(data["visit_records"])
    ], style={"padding": "0 12px 12px"})

    return html.Div([
        html.P("Mission summary", style={"fontSize": "11px", "fontWeight": "600",
                                          "color": "#444", "textTransform": "uppercase",
                                          "letterSpacing": "0.08em", "padding": "12px 12px 4px",
                                          "margin": "0"}),
        cards, cards2,
        html.P("Visit order", style={"fontSize": "11px", "fontWeight": "600",
                                      "color": "#444", "textTransform": "uppercase",
                                      "letterSpacing": "0.08em", "padding": "4px 12px",
                                      "margin": "0"}),
        visit_rows,
    ])


def _render_optimizer(data):
    if not data:
        return _no_data()

    history = data["opt_history"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(history))), y=history,
        mode="lines", line=dict(color="#378ADD", width=2),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.08)",
    ))
    fig.update_layout(
        margin=dict(l=40, r=10, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Generation", gridcolor="#eee", title_font_size=11),
        yaxis=dict(title="Best Δv (km/s)", gridcolor="#eee", title_font_size=11),
        height=220, showlegend=False,
    )

    improvement = history[0] - history[-1]
    return html.Div([
        html.Div([
            metric_card("Generations", str(data["opt_gens"])),
            metric_card("Best Δv", f"{data['opt_best_dv']:.3f} km/s", "#378ADD"),
            metric_card("Saved", f"{improvement:.3f} km/s", "#1D9E75"),
        ], style={"display": "flex", "gap": "8px", "padding": "12px"}),
        html.P("Convergence", style={"fontSize": "11px", "fontWeight": "600",
                                      "color": "#444", "textTransform": "uppercase",
                                      "letterSpacing": "0.08em", "padding": "0 12px 4px",
                                      "margin": "0"}),
        dcc.Graph(figure=fig, style={"height": "220px"},
                  config={"displayModeBar": False}),
    ])


def _render_rendezvous(data):
    if not data:
        return _no_data()

    rdv_data = data["rendezvous"]
    target_names = list(rdv_data.keys())

    dropdown = dcc.Dropdown(
        id="rdv-dropdown",
        options=[{"label": n[:28], "value": n} for n in target_names],
        value=target_names[0] if target_names else None,
        clearable=False,
        style={"fontSize": "12px", "margin": "12px"},
    )

    return html.Div([
        dropdown,
        html.Div(id="rdv-graphs"),
    ])


@app.callback(
    Output("rdv-graphs", "children"),
    Input("rdv-dropdown", "value"),
    State("mission-store", "data"),
    prevent_initial_call=True,
)
def update_rdv_graphs(selected, data):
    if not data or not selected:
        return html.Div()

    rdv = data["rendezvous"].get(selected)
    if not rdv:
        return html.Div()

    fig_range = go.Figure()
    fig_range.add_trace(go.Scatter(
        x=rdv["times_h"], y=rdv["ranges"],
        mode="lines", line=dict(color="#378ADD", width=2),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.08)",
    ))
    fig_range.add_hline(y=0.01, line_dash="dash", line_color="#1D9E75",
                        annotation_text="Dock threshold", annotation_font_size=10)
    fig_range.update_layout(
        margin=dict(l=40, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Time (h)", gridcolor="#eee", title_font_size=10),
        yaxis=dict(title="Range (km)", gridcolor="#eee", title_font_size=10),
        height=180, showlegend=False,
    )

    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=rdv["x_vals"], y=rdv["z_vals"],
        mode="lines", line=dict(color="#D85A30", width=2), name="Path",
    ))
    fig_traj.add_trace(go.Scatter(
        x=[rdv["x_vals"][0]], y=[rdv["z_vals"][0]],
        mode="markers", marker=dict(size=8, color="#378ADD"), name="Start",
    ))
    fig_traj.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=10, color="#1D9E75", symbol="star"),
        text=["Target"], textposition="top right", name="Target",
    ))
    fig_traj.update_layout(
        margin=dict(l=40, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Along-track (km)", gridcolor="#eee", title_font_size=10),
        yaxis=dict(title="Radial (km)", gridcolor="#eee", title_font_size=10),
        height=180, showlegend=False,
    )

    success_color = "#1D9E75" if rdv["success"] else "#D85A30"
    return html.Div([
        html.Div([
            metric_card("Result", "✓ Docked" if rdv["success"] else "✗ Failed", success_color),
            metric_card("Final range", f"{rdv['final_range_m']:.1f} m"),
            metric_card("RDV Δv", f"{rdv['total_dv']:.4f} km/s", "#D85A30"),
        ], style={"display": "flex", "gap": "6px", "padding": "0 12px 8px"}),
        html.P("Range", style={"fontSize": "10px", "color": "#888",
                                "textTransform": "uppercase", "padding": "0 12px", "margin": "0"}),
        dcc.Graph(figure=fig_range, config={"displayModeBar": False}),
        html.P("LVLH trajectory", style={"fontSize": "10px", "color": "#888",
                                          "textTransform": "uppercase", "padding": "0 12px", "margin": "0"}),
        dcc.Graph(figure=fig_traj, config={"displayModeBar": False}),
    ])


def _render_timeline(data):
    if not data:
        return _no_data()

    records = data["visit_records"]
    events  = data["events"]

    # Fuel waterfall
    fig_fuel = go.Figure(go.Waterfall(
        x=["Initial"] + [r["name"][:14] for r in records] + ["Remaining"],
        y=[data["fuel_initial"]] + [-r["fuel_used"] for r in records] + [data["fuel_remaining"]],
        measure=["absolute"] + ["relative"] * len(records) + ["total"],
        connector=dict(line=dict(color="#ccc", width=1)),
        decreasing=dict(marker_color="#D85A30"),
        increasing=dict(marker_color="#1D9E75"),
        totals=dict(marker_color="#378ADD"),
        hovertemplate="%{x}<br>%{y:.1f} kg<extra></extra>",
    ))
    fig_fuel.update_layout(
        margin=dict(l=40, r=10, t=10, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Fuel (kg)", gridcolor="#eee", title_font_size=10),
        xaxis=dict(gridcolor="#eee", tickangle=-30, tickfont_size=9),
        height=220, showlegend=False,
    )

    kind_colors = {
        "depart": "#7F77DD", "transfer": "#378ADD",
        "rendezvous": "#D85A30", "dock": "#1D9E75",
    }

    event_log = html.Div([
        html.Div([
            html.Span(e["kind"].upper()[:8], style={
                "fontSize": "9px", "fontWeight": "700",
                "color": kind_colors.get(e["kind"], "#888"),
                "width": "64px", "display": "inline-block", "flexShrink": "0",
            }),
            html.Span(datetime.fromisoformat(e["t"]).strftime("%H:%M"), style={
                "fontSize": "11px", "color": "#aaa",
                "width": "40px", "display": "inline-block", "flexShrink": "0",
            }),
            html.Span(e["target_name"][:16], style={
                "fontSize": "11px", "fontWeight": "500",
                "width": "110px", "display": "inline-block", "flexShrink": "0",
            }),
            html.Span(f"Δv={e['dv']:.3f}" if e["dv"] > 0 else "",
                      style={"fontSize": "10px", "color": "#888"}),
        ], style={"padding": "4px 0", "borderBottom": "0.5px solid #f5f5f5",
                  "display": "flex", "alignItems": "center"})
        for e in sorted(events, key=lambda x: x["t"])
    ], style={"maxHeight": "220px", "overflowY": "auto"})

    return html.Div([
        html.P("Fuel budget", style={"fontSize": "11px", "fontWeight": "600",
                                      "color": "#444", "textTransform": "uppercase",
                                      "letterSpacing": "0.08em", "padding": "12px 12px 4px",
                                      "margin": "0"}),
        dcc.Graph(figure=fig_fuel, config={"displayModeBar": False}),
        html.P("Event log", style={"fontSize": "11px", "fontWeight": "600",
                                    "color": "#444", "textTransform": "uppercase",
                                    "letterSpacing": "0.08em", "padding": "8px 12px 4px",
                                    "margin": "0"}),
        html.Div(event_log, style={"padding": "0 12px 12px"}),
    ])


if __name__ == "__main__":
    app.run(debug=True)