# SHAP interpretation — P50 and P90 summary plots

This note explains the beeswarm summaries [`figures/shap_summary_p50.png`](figures/shap_summary_p50.png)
and [`figures/shap_summary_p90.png`](figures/shap_summary_p90.png), produced by `scripts/run_shap.py`
via `wind_quantile_forecast.interpret.shap_explain.explain_model`.

## How to read the plots

Each dot is one forecast hour from the SHAP sample (default: 500 rows, seed 42).

| Visual | Meaning |
|--------|---------|
| **Row (y-axis)** | Feature name, ranked top-to-bottom by mean absolute SHAP value |
| **Horizontal position (x-axis)** | SHAP value in MW — how much that feature pushed the **quantile prediction** up (right) or down (left) |
| **Dot color** | That feature's value for that hour — **red = high**, **blue = low** (see color bar: Feature value) |
| **Spread / density** | Wider or denser clouds mean stronger or more variable impact across hours |

The vertical gray line at zero is no marginal impact on that quantile forecast.

### Shared context

- **Target:** `wind_mw` — German aggregate wind generation (MW)
- **Horizon:** Day-ahead, lead = 24 h (`init_time` NWP + lags joined safely at issue time)
- **Model:** LightGBM quantile GBM fit on the full modeling sample (exploratory; in-sample target encoding on `hour_season`)

SHAP values are **conditional** on the fitted tree ensemble: effects are measured after other features are accounted for along each tree path, not as simple bivariate correlations.

---

## P50 summary (`shap_summary_p50.png`)

**Quantile:** P50 (median day-ahead forecast)

### Main story

The model splits responsibility in a sensible way for day-ahead wind:

1. **Weather drivers** (especially wind direction and hub wind speed) set most of the forecast level.
2. **Recent generation history** (lags and rolling stats) acts as a **residual corrector**, often pulling the forecast down when recent output was high and up when it was low.
3. **Calendar / time-of-day** features add a modest diurnal adjustment on top.

NWP explains the bulk of the signal; autoregressive features refine it using persistence and recent regime.

### Top features

#### 1. `nwp_wind_direction_cos` (most important)

East–west component of forecast wind direction (cosine of 10 m direction; hub speed is a separate feature).

- **Red (high cos) → large negative SHAP** (left, down to roughly −6000 MW)
- **Blue (low cos) → positive SHAP** (right, up to roughly +1500 MW)

Wind **direction** shifts the median forecast strongly, not just speed. For Germany, prevailing westerlies / northwesterlies often align with productive regimes, while other directions (e.g. easterly flows) can correspond to lower fleet-wide output. The wide left tail means some direction values can **strongly suppress** the P50 forecast.

#### 2. `nwp_wind_speed_hub_mps`

Classic wind–power relationship:

- **Red (high speed) → positive SHAP** — more wind → higher P50
- **Blue (low speed) → negative SHAP** — calm → lower P50

Second-largest overall importance; the most intuitive physical driver.

#### 3. `cos_hour`

Diurnal cycle encoding (hour of day on a cosine curve):

- **Red (later-day side of the cycle) → slightly positive**
- **Blue (earlier-day side) → slightly negative**

Smaller than NWP but consistent: **time-of-day** still shifts the forecast after weather is known (diurnal wind patterns or hour effects not fully captured by NWP alone).

#### 4. `wind_mw_roll24_min` (24 h rolling minimum of past generation)

- **Red (high recent minimum) → negative SHAP**
- **Blue (low recent minimum) → positive SHAP**

When the past day's weakest output was still relatively high, the model tends to **lower** P50; after a very calm spell (low minimum), it **raises** P50. This may reflect recovery after lulls or use of roll-min as a regime indicator once NWP is in the model.

#### 5. `wind_mw_lag_1` (generation 1 h before forecast issue time)

- **High recent output (red) → negative SHAP**
- **Low recent output (blue) → positive SHAP**

Very recent generation **pulls the forecast toward lower values** given NWP — partial **mean reversion / residual adjustment**, not pure persistence.

#### 6. `days_since_last_holiday`

Mostly small effect, with a long **red tail to the right** for some high values. Likely a weak **calendar proxy** (holidays cluster in certain seasons) rather than a direct physical wind driver. Treat as secondary.

### Mid-tier features (P50)

| Feature | Pattern | Likely meaning |
|---------|---------|----------------|
| `hour_season_te` | Tight around zero | Target-encoded typical output for hour×season; mostly redundant once NWP + lags are present |
| `wind_mw_lag_24`, `wind_mw_lag_6` | High past gen → negative | Same autoregressive correction as `lag_1` |
| `wind_mw_roll24_mean` | High roll mean → negative | Recent strong period → slightly lower adjusted forecast |
| `sin_hour` | Smaller diurnal complement to `cos_hour` | Time-of-day effect |
| `is_weekend` | Small, mixed | Weak weekday/weekend difference |
| `nwp_air_density_kg_m3`, `nwp_t2m_k` | Small | Physics drivers; minor vs speed and direction |

### Practical reading for P50

1. **Set the level:** hub wind speed + wind direction (especially `nwp_wind_direction_cos`).
2. **Fine-tune:** recent generation (`lag_1`, `roll24_min`, `roll24_mean`) — often opposing very high or very low recent output.
3. **Nudge diurnal / calendar:** `cos_hour` / `sin_hour` and weak holiday/weekend effects.

Example intuition: direction and speed might imply ~8000 MW P50; a very high `wind_mw_lag_1` might shave hundreds to low thousands of MW off that median once weather is fixed.

---

## P90 summary (`shap_summary_p90.png`)

**Quantile:** P90 (upper tail — “high but plausible” day-ahead output)

### Main story

P90 shares the same **NWP-first** structure as P50: wind direction and hub speed dominate. The upper quantile adds emphasis on **volatility and calendar proximity**, because the model must widen the forecast upward when recent conditions suggest upside risk or when temporal context shifts typical peaks.

1. **Weather drivers** still anchor the band — direction cos and hub speed rank first and second with similar sign patterns to P50.
2. **Autoregressive features** remain important correctors (`roll24_min`, `lag_1`), with the same high-recent-output → lower SHAP pattern.
3. **Calendar and volatility features rise in rank** relative to P50 — holidays, day-of-week, and 24 h rolling **standard deviation** matter more for the tail than for the median.

Operationally, P90 is the quantile you'd stress for **upside reserve** (e.g. procuring less downward flexibility than a naive point forecast might imply).

### Top features

#### 1. `nwp_wind_direction_cos` (most important)

Same pattern as P50, with comparable magnitude:

- **Red (high cos) → large negative SHAP** (left, to roughly −6000 MW)
- **Blue (low cos) → positive SHAP** (right)

Direction remains the single largest lever on the **upper** forecast, not only the median. Unfavorable directions cap P90 sharply; favorable directions lift the tail.

#### 2. `nwp_wind_speed_hub_mps`

- **Red (high speed) → strong positive SHAP**
- **Blue (low speed) → negative SHAP**

High forecast wind speed is the primary physical driver of a **high P90**. The beeswarm is tight and monotonic — the clearest wind–power relationship on the plot.

#### 3. `cos_hour`

- **Red → positive SHAP**
- **Blue → negative SHAP**

Diurnal structure affects the upper quantile as well as the median: certain hours get a higher upside band even holding NWP fixed.

#### 4. `wind_mw_roll24_min`

- **Red (high recent minimum) → negative SHAP**
- **Blue (low recent minimum) → positive SHAP**

Same residual-corrector logic as P50. A strong recent floor in generation slightly **lowers** P90 once weather is accounted for; a very weak floor **raises** it (recovery / upside after calm).

#### 5. `wind_mw_lag_1`

- **Red (high lag) → negative SHAP**
- **Blue (low lag) → positive SHAP**

Very recent high output still **pulls P90 down** conditional on NWP — the upper band is not simply “median plus a constant.”

#### 6. `days_to_next_holiday` (more prominent at P90 than P50)

- **Red (far from next holiday) → negative SHAP**
- **Blue (close to next holiday) → positive SHAP**

Approaching a holiday **raises** the P90 forecast in this sample; being far from the next holiday lowers it. Likely a **seasonal / calendar proxy** rather than a causal wind effect — treat cautiously given the short June 2019 window.

#### 7. `days_since_last_holiday`

- **Red (long since last holiday) → positive SHAP** (including a long right tail)
- **Blue (just after holiday) → small negative SHAP**

Complements `days_to_next_holiday`: calendar distance from holidays shifts the **upside** band, not just the median.

#### 8. `day_of_week`

- **Red (later in the week) → positive SHAP**
- **Blue (earlier in the week) → negative SHAP**

Weak but visible weekday structure on the upper quantile — possibly correlated with weather regimes or summer-week patterns in the sample.

#### 9. `wind_mw_lag_48` (48 h lag — stronger at P90 than P50)

- **Red (high output 48 h ago) → positive SHAP**
- **Blue (low) → negative SHAP**

Unlike `lag_1`, the 48 h lag **raises P90 when past output was high** — persistence on a two-day horizon supports a higher upside band. This is a key **P90 vs P50 difference**: short lags correct downward; longer lags can support upward tail risk.

#### 10. `wind_mw_roll24_std` (24 h rolling standard deviation)

- **Red (high recent volatility) → positive SHAP**
- **Blue (low volatility) → negative SHAP**

When the past day was **variable**, P90 moves **up**. That matches quantile semantics: volatile recent output implies greater upside uncertainty. This feature is **tail-specific** — it ranks higher and matters more for P90 than for P50.

### Mid-tier features (P90)

| Feature | Pattern | Likely meaning |
|---------|---------|----------------|
| `hour_season_te` | Small, near zero | Same redundancy as P50 once NWP and lags are in |
| `wind_mw_lag_24`, `wind_mw_lag_6` | Mixed | Shorter lags correct; 24 h lag less dominant than `lag_48` at P90 |
| `wind_mw_roll24_mean` | High mean → negative | Similar corrector role as P50 |
| `sin_hour`, `is_weekend` | Small | Diurnal / weekend nudges on the tail |
| `nwp_air_density_kg_m3`, `nwp_t2m_k` | Small | Minor physics adjustments |

### Practical reading for P90

1. **Upside anchor:** favorable `nwp_wind_direction_cos` + high `nwp_wind_speed_hub_mps`.
2. **Widen the tail:** high `wind_mw_roll24_std` and supportive `wind_mw_lag_48`.
3. **Trim the tail:** high `wind_mw_lag_1` or high `wind_mw_roll24_min` (recent strong/calm-floor regime).
4. **Calendar nudges:** holiday proximity features — interpret lightly given sample size.

Example intuition: strong NWP winds might set P90 near 12 000 MW; elevated 24 h volatility adds upside; a very high `wind_mw_lag_1` could still pull P90 down by hundreds of MW.

---

## P50 vs P90 — what changes?

| Aspect | P50 (median) | P90 (upper tail) |
|--------|--------------|------------------|
| **NWP direction & speed** | Dominant | Same dominance, similar signs |
| **`wind_mw_lag_1`** | High recent gen → lower forecast | Same downward correction |
| **`wind_mw_lag_48`** | Smaller / mixed | High past gen → **higher** P90 (persistence on 2-day horizon) |
| **`wind_mw_roll24_std`** | Low importance | **High volatility → higher P90** (tail widens) |
| **Holiday features** | Weak (`days_since_last_holiday` only) | **`days_to_next_holiday` and `days_since_last_holiday` both rank higher** |
| **`day_of_week`** | Barely visible | More visible on P90 |
| **Operational use** | Best point schedule / expected output | Upside stress case, upward reserve planning |

Same model and features, different quantile loss — the tree ensemble learns **different splits** for the 90th percentile, so SHAP correctly surfaces tail-relevant drivers (volatility, longer persistence) that barely move the median.

---

## Caveats

- **Sample scope:** Default dataset window is mid-June 2019 (`2019-06-01`–`2019-06-15` in config). Direction, holiday, and weekday patterns may not represent full-year behavior.
- **Exploratory encoding:** SHAP script uses in-sample LOO target encoding on the full frame — appropriate for driver analysis, **not** for production deployment (see `WEEK_03_REFLECTION.md`).
- **Independent quantiles:** P10, P50, and P90 are fit with separate pinball objectives (LightGBM); bands can cross in edge cases with no post-hoc monotonicity fix.
- **Direction cos is abstract:** “High cos” maps to a specific compass direction; dependence plots (`shap_dependence_p50_*.png`, `shap_dependence_p90_*.png`) show marginal SHAP vs feature value more directly.
- **Calendar effects:** Holiday and weekday SHAP on P90 should be validated on a longer calendar before operational use.

## Related artifacts

| File | Description |
|------|-------------|
| `figures/shap_summary_p50.png` | Beeswarm summary for P50 |
| `figures/shap_summary_p90.png` | Beeswarm summary for P90 |
| `figures/shap_dependence_p50_*.png` | SHAP dependence plots for selected features (P50) |
| `figures/shap_dependence_p90_*.png` | SHAP dependence plots for selected features (P90) |

Regenerate plots:

```powershell
py scripts/run_shap.py -v
```
