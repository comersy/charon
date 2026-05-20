# Charon 🛸

**Open-source Python framework for planning and simulating multi-target on-orbit servicing missions.**

Charon covers the full pipeline — from real TLE ingestion to genetic sequence optimization, rendezvous simulation, and live 3D visualization.

> No equivalent open-source tool exists. Commercial alternatives (STK, GMAT) are either proprietary or limited to single-mission scenarios.

---

## Features

- **Live TLE ingestion** — fetches real satellite positions from Celestrak in real time
- **SGP4 propagation** — accurate orbital state computation for any satellite
- **Hohmann & Lambert transfers** — delta-v calculation between any two orbits
- **Genetic optimizer** — finds the minimum delta-v visit order across N targets
- **Clohessy-Wiltshire rendezvous** — physics-based final approach simulation
- **End-to-end mission timeline** — chains optimizer → transfers → rendezvous into a single pipeline
- **Live dashboard** — interactive 3D globe, optimizer convergence, rendezvous trajectory, fuel budget

---

## Architecture

```
charon/
├── core/
│   ├── propagator/        # SGP4 propagator + abstract BasePropagator
│   ├── maneuver.py        # Hohmann & Lambert transfer solvers
│   └── spacecraft.py      # Servicer model, Tsiolkovsky fuel tracking
├── mission/
│   ├── target.py          # Target satellite (TLE + fuel need + deadline)
│   └── sequence.py        # Ordered visit sequence, feasibility checks
├── optimizer/
│   └── genetic.py         # Genetic algorithm (OX1 crossover, elitism)
├── simulation/
│   ├── rendezvous.py      # CW equations, proportional guidance
│   └── timeline.py        # End-to-end MissionPlanner
├── dashboard/
│   └── app.py             # Live Plotly/Dash dashboard
└── tests/                 # Unit tests for all modules
```

---

## Quickstart

```bash
git clone https://github.com/yourname/charon.git
cd charon
pip install -r requirements.txt
python dashboard/app.py
```

Open `http://127.0.0.1:8050` in your browser.

---

## Requirements

```
sgp4
numpy
dash
plotly
requests
```

Install all at once:

```bash
pip install sgp4 numpy dash plotly requests
```

---

## How it works

### 1. Select targets
Pick any satellites from the live Celestrak catalog (ISS, Starlink, etc.) using the sidebar checkboxes. Their real orbits appear on the 3D globe instantly.

### 2. Run Mission
Click **▶ Run Mission**. Charon:
1. Fetches current TLEs and computes live positions via SGP4
2. Runs the genetic algorithm to find the optimal visit order (minimum Δv)
3. Evaluates Hohmann transfers between consecutive targets
4. Simulates the final rendezvous approach for each target using Clohessy-Wiltshire equations
5. Builds a complete mission timeline with fuel tracking

### 3. Explore results
Four info panels on the right:
- **Mission** — total Δv, fuel used, duration, docking results
- **Optimizer** — fitness convergence curve, optimal visit order
- **Rendezvous** — range over time, LVLH trajectory per target
- **Timeline** — Gantt chart, fuel waterfall, full event log

---

## Physics

| Module | Method | Reference |
|--------|--------|-----------|
| Propagation | SGP4/SDP4 | Hoots & Roehrich (1980) |
| Coplanar transfer | Hohmann | Curtis (2013), Ch. 6 |
| General transfer | Lambert / universal variable | Izzo (2015) |
| Fuel consumption | Tsiolkovsky rocket equation | — |
| Final approach | Clohessy-Wiltshire (Hill) equations | Clohessy & Wiltshire (1960) |
| Sequence optimization | Genetic algorithm (OX1 + elitism) | — |

---

## Running tests

```bash
python -m tests.test_propagator
python -m tests.test_maneuver
python -m tests.test_spacecraft
python -m tests.test_target
python -m tests.test_sequence
python -m tests.test_genetic
python -m tests.test_rendezvous
python -m tests.test_timeline
```

---

## Roadmap

- [ ] RK4 / J2 propagator (plug-in via `BasePropagator`)
- [ ] Lambert-based optimizer (higher fidelity transfers)
- [ ] Animated servicer trajectory on the globe
- [ ] Real-time mission clock with animated satellite positions
- [ ] Multi-servicer mission planning
- [ ] Debris removal scenario support
- [ ] Export mission report to PDF

---

## License

MIT

---

## Contributing

PRs welcome. See the architecture above — each module is independently testable and documented.
