# Hospital Digital Twin — Evaluation Report

Snapshot: 14 Oct 2026 08:00 · history: 730 days · 164,594 encounters

## 1. Forecasting (rolling-origin back-test, 10 origins × 14-day horizon)

| target      | model    |   MAE |   RMSE |   MAPE |   Skill vs naive % |   80% interval coverage |
|:------------|:---------|------:|-------:|-------:|-------------------:|------------------------:|
| ed_arrivals | LightGBM | 18.09 |  21.99 |   8.76 |              22.80 |                   78.57 |
| ed_census   | LightGBM |  2.38 |   2.92 |  10.59 |              28.17 |                   81.43 |
| icu_census  | Ensemble |  2.49 |   2.93 |  13.20 |              17.79 |                   78.57 |
| med_census  | Ensemble | 10.04 |  12.48 |   6.52 |              19.42 |                   82.14 |
| surg_census | LightGBM |  8.34 |  10.77 |   9.88 |              23.74 |                   81.43 |

Skill = MAE reduction vs seasonal-naive. Coverage = share of back-test actuals inside the calibrated 80% band.

## 2. Twin validation (replay of the last 8 weeks of observed demand, 10 replications)

| series               |   observed_mean |   twin_mean |   MAPE_% |   correlation |
|:---------------------|----------------:|------------:|---------:|--------------:|
| ED arrivals / h      |            8.92 |        8.95 |     6.09 |          0.72 |
| ED census            |           22.92 |       23.23 |     8.22 |          0.65 |
| ICU census           |           19.54 |       20.27 |    14.12 |          0.21 |
| Medical ward census  |          153.31 |      155.96 |     5.01 |          0.39 |
| Surgical ward census |           89.08 |       94.74 |    10.17 |          0.85 |

## 3. Nurse rostering (next 7 days, P90 demand)

| roster           |   required_shifts |   scheduled_shifts |   agency_shifts |   short_shifts |   over_shifts |   redeployments |   float_shifts |      cost |
|:-----------------|------------------:|-------------------:|----------------:|---------------:|--------------:|----------------:|---------------:|----------:|
| Fixed template   |              1308 |               1247 |               0 |            130 |            69 |               0 |             87 | 451425.00 |
| CP-SAT optimised |              1308 |               1299 |               9 |              0 |             0 |               0 |             90 | 145470.00 |

## 4. Elective scheduling (next 14 days)

| plan                 |   scheduled |   postponed |   p1_scheduled_pct |   late_cases |   theatre_util_pct |   bed_overflow_days |
|:---------------------|------------:|------------:|-------------------:|-------------:|-------------------:|--------------------:|
| FIFO by listing date |         225 |          15 |              77.42 |           81 |              99.61 |                  11 |
| CP-SAT bed-aware     |         177 |          63 |              93.55 |           16 |              65.24 |                   0 |

## 5. Ablation — reactive operations vs simulation-verified AI plan

| scenario               | plan                                             |   Door-to-doctor (min) (reactive) |   Door-to-doctor (min) (AI plan) |   ED boarding mean (h) (reactive) |   ED boarding mean (h) (AI plan) |   Left without being seen (reactive) |   Left without being seen (AI plan) |   Hours ICU full (reactive) |   Hours ICU full (AI plan) |   Electives cancelled (reactive) |   Electives cancelled (AI plan) |
|:-----------------------|:-------------------------------------------------|----------------------------------:|---------------------------------:|----------------------------------:|---------------------------------:|-------------------------------------:|------------------------------------:|----------------------------:|---------------------------:|---------------------------------:|--------------------------------:|
| Baseline               | Integrated response plan (3 coordinated actions) |                              29.7 |                             28.1 |                               0.9 |                              0.6 |                                 91.4 |                                78.4 |                        34.5 |                        1.8 |                             29.8 |                            24.8 |
| Monsoon Fever Surge    | Integrated response plan (5 coordinated actions) |                              73.1 |                             36.9 |                               3.2 |                              1.5 |                                605.6 |                               245.4 |                        92.4 |                       26.4 |                             73.3 |                            41.7 |
| Mass Casualty Incident | Integrated response plan (6 coordinated actions) |                              37.2 |                             19.3 |                               1.5 |                              0.5 |                                153.0 |                                46.4 |                        70.2 |                        7.9 |                             37.2 |                            20.6 |
| ICU Bed Closure        | Integrated response plan (5 coordinated actions) |                              34.5 |                             14.5 |                               2.3 |                              0.4 |                                172.2 |                                24.0 |                       125.0 |                       14.5 |                             32.2 |                            20.9 |
