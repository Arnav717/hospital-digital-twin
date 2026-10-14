# MediTwin — AI-Powered Hospital Digital Twin

**Predictive healthcare operations and decision intelligence**
Shresth Kumar Gupta (23BCE1578) · Arnav Kamra (23BCE1706) · SCOPE, VIT Chennai

MediTwin is a virtual replica of a 360-bed multispecialty hospital in Chennai. It **forecasts** demand and
occupancy, **simulates** patient flow through ED, ICU, wards, theatres and the discharge lounge,
**optimises** nurse rosters, theatre lists and patient transfers, and **recommends** actions. Every
recommendation is verified in the simulator before it is shown. Most existing work covers only one of these
four layers; this project combines all of them in one platform.

```
Hospital data → Pre-processing → ML forecasting → Digital-twin state → Simulation engine
                                        │                                   │
                                        └──────────► Optimisation ◄─────────┘
                                                          │
                                         AI recommendation engine ──► Command centre / REST API
```

## Quick start

```bash
uv sync                          # Python 3.12 environment
uv run hospital-twin build       # ~75 s: 2 years of history, forecasts, optimisation, recommendations
uv run hospital-twin evaluate    # ~20 s: writes reports/evaluation.md
uv run hospital-twin app         # command centre at http://localhost:8501
uv run hospital-twin api         # REST API at http://localhost:8000/docs
uv run pytest                    # test suite
```

Optional: set `ANTHROPIC_API_KEY` and Claude will write the executive briefing. Without a key, a templated
briefing is used.

## What each layer does

| Layer | Implementation | Key file |
|---|---|---|
| **Hospital data** | Two years of synthetic hospital history (~165k encounters). Demand reflects Chennai patterns: weekly cycle, north-east-monsoon fever season, summer heat, winter respiratory wave, Indian public holidays, Diwali trauma spike, AR(1) noise and random mass-casualty events. The history is produced by running this demand through the simulator, so occupancy, boarding and cancellations respect real capacity limits. | `data/generator.py` |
| **Forecasting** | Seasonal-naive baseline, SARIMAX (holiday exogenous variables, with a divergence-safe fallback chain), recursive LightGBM (calendar, lag and rolling features) and an ensemble. Rolling-origin back-test (10 origins × 14 days) picks a champion model per target. 80% intervals are calibrated on back-test residuals. SHAP explains every forecast day. | `forecasting/` |
| **Digital twin** | Live snapshot of every bed, queue and patient (unit, elapsed stay, acuity). Forward simulations warm-start from it, sampling remaining length of stay from truncated log-normal distributions. | `twin/` |
| **Simulation** | SimPy discrete-event model. ESI triage; ED beds with left-without-being-seen patience; doctors on shifts (dynamic capacity); boarding and outlier placement; emergency and elective theatres; ICU exit block and step-down; discharge window and lounge. Monte-Carlo runs use common random numbers, and demand paths are sampled from the forecast interval. | `simulation/` |
| **Optimisation** | **CP-SAT nurse roster**: ratio-based demand from P90 census forecasts, ≤1 shift per day, no night→day turnaround, contracted hours, night cap, days off, float pool and agency cover. **CP-SAT elective scheduler**: theatre hours plus post-op ICU/ward bed constraints alongside the forecast emergency load, priority- and wait-weighted. **LP patient redistribution** (HiGHS) to outlier beds or partner hospitals. | `optimization/` |
| **AI recommendations** | Detects risk in forecasts and twin projections and proposes playbook interventions: surge ICU beds, discharge drive, extra physicians, ambulance diversion, elective re-planning, deferral, nurse roster. Each one is **re-simulated** to give its KPI change with a 90% CI, then accepted or rejected on a weighted benefit score. Accepted actions are bundled and re-simulated as an integrated plan. | `recommender/` |
| **Interfaces** | Streamlit + Plotly command centre (7 pages) and a FastAPI REST API. | `app/`, `api/` |

## Command centre

| Page | Highlights |
|---|---|
| 🛰️ Command Center | Live status, KPI cards with sparklines, **animated hospital floor plan** (every bed, boarders, ICU exit-block, waiting room) playing the next 72 h, 7-day risk outlook, top verified actions |
| 📈 Forecasts | Fan charts with calibrated bands, model comparison, SHAP global drivers and per-day waterfall, hour-of-week demand heatmap |
| 🧪 Scenario Lab | Presets (monsoon surge, mass casualty, ICU closure, staff shortage, festival) plus sliders. Animated twin, trajectory bands, outcome violins, Sankey of patient flow, one-click **AI response plan** for the scenario |
| ⚙️ Optimizer | Staffing-gap heatmaps (template vs optimised), roster grid, elective Gantt with bed-capacity checks, transfer map of partner hospitals |
| 🧠 AI Recommendations | Executive briefing, severity-ranked action cards with impact charts, and actions that were considered but rejected |
| 🔀 Patient Flow | Patient journey Sankey, acuity mix, LOS violins, occupancy history, the exit-block relationship |
| 🔬 Model Lab | Architecture diagram, back-test detail, twin validation, optimisation benchmarks, ablation |

## REST API

`GET /state` · `GET /forecast/{target}` · `POST /simulate` · `POST /recommendations` · `GET /roster` ·
`GET /electives` — interactive docs at `/docs`.

```bash
curl -X POST localhost:8000/simulate -H 'content-type: application/json' \
     -d '{"name":"ICU closure","bed_delta":{"ICU":-6},"reps":8}'
```

## Evaluation

`reports/evaluation.md` is regenerated by `hospital-twin evaluate`. It covers:

1. **Forecasting**: champion MAE, RMSE, MAPE, skill versus seasonal-naive, and coverage of the 80% interval.
2. **Twin validation**: the last 8 weeks of observed demand replayed through the simulator, comparing mean
   census and MAPE. Day-to-day correlation is moderate by design, because admissions are stochastic.
3. **Rostering**: CP-SAT versus a fixed template roster (uncovered shifts, over-staffing, cost).
4. **Elective scheduling**: CP-SAT bed-aware plan versus first-come-first-served (P1 coverage, bed-overflow
   days, late cases).
5. **Ablation**: reactive operations versus the simulation-verified AI plan under baseline, monsoon surge,
   mass-casualty and ICU-closure scenarios.

## Project layout

```
src/hospital_twin/
  config.py            hospital capacity, case-mix, LOS & service-time distributions
  data/generator.py    synthetic demand + history generation
  forecasting/         models.py, pipeline.py (back-test, intervals, SHAP)
  simulation/          engine.py (SimPy), runner.py (Monte-Carlo), scenario.py
  twin/                state.py (live state), context.py (loads built twin, runs projections)
  optimization/        roster.py, electives.py, redistribution.py
  recommender/         engine.py (detect → propose → verify), brief.py (executive briefing)
  pipeline.py          end-to-end build
  evaluation.py        report generation
  api/main.py          FastAPI
app/                   Streamlit command centre (Home.py + views/)
tests/                 pytest suite
```

## Limitations

- Patient data is synthetic. It is calibrated to plausible Indian tertiary-hospital volumes, but it is not a
  real hospital record. Real feeds (HL7/FHIR ADT events) would replace `data/generator.py` and the snapshot
  builder.
- Nurse staffing is optimised and monitored but does not slow care inside the simulator. ED physicians do.
- Partner-hospital capacity is static and fictional.
