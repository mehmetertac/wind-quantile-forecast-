# Week 03 reflection

## What did I build?

A **probabilistic day-ahead wind forecast** stack that turns Week 2 weather features into **P10 / P50 / P90** quantile predictions for German aggregate wind:

- **`QuantileGBM`** — LightGBM, XGBoost, and CatBoost wrappers trained with pinball loss (`objective="quantile"` / `MultiQuantile` / `reg:quantileerror`).
- **Rolling-origin CV** — expanding train window, fold-wise **leave-one-out target encoding** on `hour_season` (96 hour×season levels).
- **Backend comparison** — `scripts/run_comparison.py` ranks backends on pinball loss, PI coverage, MAE/RMSE/MAPE (P50), and train time. **LightGBM + target encoding** wins on `pinball_q50` and `pi_coverage`.
- **Feature matrix** — calendar cyclicals, NWP wind physics (`nwp_wind_speed_hub_mps`, direction sin/cos, air density), autoregressive lags/rolls (joined at `init_time`, safe for 24 h lead), ERA5 columns excluded to prevent hindcast leakage.
- **SHAP interpretability** — `scripts/run_shap.py` fits the best backend on the full sample and writes beeswarm + dependence plots for **P50 and P90** to `reports/figures/` (`shap_summary_p50.png`, `shap_dependence_p50_nwp_wind_speed_hub_mps.png`, etc.).

Together this is the layer that answers: *not just how much wind, but how uncertain is that estimate hour by hour?*

## What's still fuzzy?

- **Calibration** — Nominal P10–P90 interval should cover ~80 % of observations; rolling CV gives **~58 % PI coverage** for LightGBM. Intervals are too narrow. Reliability diagrams are stubbed in `evaluation/plots.py`; no isotonic or quantile recalibration yet.
- **Encoding leakage** — CV encodes `hour_season` fold-wise with LOO statistics. The SHAP script uses `add_target_encoding()` (in-sample LOO on the full frame) for exploratory plots — fine for understanding drivers, **not** for production deployment.
- **Quantile crossing** — P10, P50, and P90 are fit independently (LightGBM) or jointly (CatBoost multi-quantile) with no post-hoc monotonicity fix; bands can invert in edge cases.
- **Sample scope** — June 2019 DE national aggregate (~two weeks at day-ahead lead). Calm spells inflate MAPE; frontal ramps are underrepresented. Numbers are directionally right, not statistically stable.
- **Model persistence** — CV fits per-fold models and discards them; no `save_model` / full-data retrain artifact for ops. CLI entry point `wind-forecast` is still `NotImplementedError`.

## What this means in dispatch / market terms

Week 2 showed that adding NWP weather to a calendar-only model removes **~1.1 GW of average absolute schedule error** on national German wind (~11 GW mean output). That is **~1,000 MWh/h** of regulating reserve you do not need to procure for each gigawatt of tighter schedule error — before any quantile layer.

This week adds the **uncertainty envelope** around that point forecast:

| Output | Operational role |
|--------|------------------|
| **P50 (median)** | Day-ahead **schedule / bid** submitted before the auction (~D−1 noon CET, 24 h lead). CV MAE on P50 is **~3.2 GW** for LightGBM — the expected injection traders and schedulers plan around. |
| **P10–P90 band width** | Hourly **uncertainty envelope**. A wide band (large P90 − P10) flags hours where dispatchers and TSOs should hold **more regulating reserve** — the model is less sure, so imbalance risk is higher even if P50 looks fine. |
| **Under-calibrated bands** | Observed **~58 % coverage** vs **80 % nominal** means intervals are **too narrow**. Operators treating P10/P90 as true 10th/90th percentiles would **under-procure reserve** and take more imbalance exposure than the labels suggest. |
| **SHAP on P50 vs P90** | P50 SHAP ranks what drives the **central schedule** (hub wind speed, recent generation lags, diurnal seasonality). P90 SHAP highlights what pushes **upside tail risk** — useful for ramp hours and low-confidence weather regimes, even when median error is modest. |

**Reserve sizing intuition:** if P90 − P10 is 4 GW for an hour, a conservative dispatcher might treat **~2 GW of upward/downward flex** as the imbalance buffer around the P50 schedule (exact rules vary by TSO and product). Under-calibrated bands shrink that buffer on paper while real weather error stays the same — a direct path to balancing activation and PPA imbalance charges.

**Bridge from Week 2:** weather lift shrinks **point error**; the quantile layer sizes **residual risk** for balancing and settlement. Invest in weather and bias correction first (Week 2), then tune **spread and calibration** (this week) — retuning LightGBM alone cannot fix intervals that ignore ensemble spread or MOS bias.

## How does this connect to a real wind-farm dispatch / bidding problem?

Renewable operators, aggregators, and TSOs operate on **decision horizons**, not single MAE numbers:

- **Day-ahead auction** — Traders submit volume near **P50** per hour. The **P10–P90 width** informs how much **imbalance buffer** to keep (physical reserve, intraday hedge, or conservative bid shading). A farm that bids P50 aggressively in a wide-band hour risks **imbalance charges** when actual output lands outside the schedule.
- **TSO reserve procurement** — System operators size **regulating reserve** against forecast uncertainty. National quantile forecasts (even zone-level proxies) rank **which hours need more flex** — wide bands correlate with weather fronts, low-wind transitions, and NWP disagreement.
- **Intraday refresh** — Shorter-lead updates (6–12 h) shrink the weather error Week 2 measured between NWP and ERA5. The quantile model tells you **what uncertainty remains** after the day-ahead gate — relevant for intraday rebalancing and continuous trading.
- **Asset-level PPA** — A single plant cares about **metered vs scheduled MWh**. Under-forecast → **curtailment or missed revenue**; over-forecast → **imbalance penalties**. P10/P90 bands at portfolio level translate to **tail risk** at asset level once allocation and correlation are applied.

The Week 3 stack separates **“how much wind on average”** (P50 + weather features) from **“how wrong could we be”** (P10–P90 + calibration). That second question is what dispatch desks and risk teams actually trade on — and where this project still has the most open work (reliability diagrams, recalibration, ensemble features from Week 2’s fuzzy list).

Reproduce SHAP plots: `py scripts/run_shap.py -v` (requires `data/processed/day_ahead_wind.parquet` from `pull-wind-data`).
