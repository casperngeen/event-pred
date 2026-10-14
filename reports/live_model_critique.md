# Critique of the Deployed In-Game Model & MLB Live Pipeline

**Date:** 2026-10-14  
**Scope:** Deployed in-game spatio-temporal graph models (`MLB_final_gru_d10_h30-60-120-1800_tp_pitch_seed0..4.pt` and `NFL_final_gru_d10_h30-60-120-1800_tp_seed0..4.pt`), training/export pipeline (`stg_infra/stg/models/ingame.py`, `analysis/graph_2026_10/b3_table.py`, `b4_export.py`, `b32_deploy_panel.py`), and live inference/trading engine (`scripts/live_model.py`, `scripts/live_trade.py`, `scripts/live_record.py`, `scripts/live_rerun.py`).

---

## 1. Executive Summary & Engineering Strengths

The deployed system combines an analytical option-pricing baseline (`baseline_price`) with a relation-typed graph neural network (`InGameGraph`) and a play-sequence GRU memory, predicting 4 horizon price deltas (`30s, 60s, 120s, 1800s`) in cents alongside a binary settlement head.

### What Is Done Well
1. **Hybrid Structural + Neural Residual Inductive Bias (`ingame.py:103-112`, `make_batch:172-180`):** Computing closed-form normal-CDF fair values (`baseline_price`) from current win probability (`p_home`), pregame total (`L`), and remaining game fraction (`tau`), then passing `base / 100` and `base_gap = (base - value) / 10` as node inputs lets the GNN focus on residual mispricing and cross-market lead-lag rather than re-learning nonlinear strike geometry from scratch.
2. **Temporal & Numeric Parity Discipline (`ingame.py:29-36`, `live_model.py:539`, `live_parity.py`):** Verifying `PANEL_IDENTITY` at checkpoint load time, enforcing monotonic event times via `np.maximum.accumulate(pn.plays_t)` in `known_state` to prevent out-of-order play lookahead, and replicating `float16` panel quantization during live inference (`astype(np.float16).astype(np.float32)`) show strong engineering hygiene.
3. **Microstructure-Aware Interval Loss (`ingame.py:261-263`):** Penalizing errors only outside the 15-second print range `[plo, phi]` via Huber loss (`interval_error`) avoids overfitting to bid-ask bounce within the target window.

---

## 2. Critical MLB-Specific Bugs & Domain Deficits

Because MLB is the primary live deployment target (using the every-pitch model `MLB_pitch`), these issues have the most immediate impact on live forecast quality and PnL.

### 2.1 Silent Bug: `new_plays` Is `0.0` on ~99.5% of Newly Arrived Pitches in Live Trading
In `stg_infra/stg/models/ingame.py:84-99`, `known_state` computes the `new_plays` step feature (`known > before`, feature index 8 of `steps`) as:
```python
cut = step_t - delay_s
prev_cut = step_t - 5.0 - delay_s
known = np.searchsorted(pt, cut, side="right")
before = np.searchsorted(pt, prev_cut, side="right")
...
feats = np.stack([..., (known > before), np.log1p(since) / 6], 1)
```
* **During training (`delay_s = 10.0`):** `pt` stores the pitch end `t_end`. On the first 5-second step after a pitch clears the 10-second delay ($t_{\text{end}} + 10 \le t_{\text{step}} < t_{\text{end}} + 15$), `pt` satisfies $t_{\text{step}} - 15 < \text{pt} \le t_{\text{step}} - 10$. Thus **`new_plays` (`known > before`) is `1.0` on the arrival step of every pitch in training**, and `since = step_t - pt` starts in $[10, 15)\text{s}$.
* **During live trading (`--delay 0.0` default in `live_trade.py:492`):** `mlb_plays` (`live_model.py:139-192`) still sets `pt` (`t_event`) to the on-field pitch end `t_end` (floored at `t_start + 3.9s`), while the live runner only receives the pitch at `received_at` ($6\text{–}16\text{s}$ after `t_end`).
  * When a new pitch finally arrives in `self.plays` at $t_{\text{step}} \ge t_{\text{end}} + 6.1\text{s}$ and `known_state` runs with `delay_s = 0.0`, `prev_cut = step_t - 5.0` is **already greater than `t_end`**.
  * Because `known_state` recomputes `before` from `pt <= step_t - 5.0` rather than comparing against how many pitches `Engine` actually knew 5 seconds ago, **`before` equals `known` on the arrival step!**
  * **Empirical verification (`2026-10-12` live runs):** In `plays_seen.jsonl.gz`, **297 / 298 pitches in `MLB_849812`** and **271 / 273 pitches in `MLB_849820`** arrived with `received - happened > 5.0s` (and even with cache-busted `statsapi`, median arrival lag is `6.1s > 5.0s`). Consequently, **`new_plays` is `0.0` on ~99.5% of pitches when they arrive live**, and with cache-busted `statsapi` (`since ≈ 6.1s`), `log_since_play` falls below the $[10, \infty)\text{s}$ support seen during training.

### 2.2 Silent Bug: Every MLB Double Play and Triple Play Is Misclassified as an Extra-Base Hit
In `analysis/graph_2026_10/b1_response.py:59-61`, `b3_table.py:201-202`, and `scripts/live_model.py:53-55, 169-170`:
```python
MLB_GROUPS = [
    ("home_run", "home_run"),
    ("extra_base_hit", "double|triple"),
    ("single", "single"),
    ("walk_hbp", "walk|hit_by_pitch|intent"),
    ("strikeout", "strikeout"),
    ("double_play", "double_play|triple_play"),
    ("error", "error"),
    ("out_in_play", "out|sac|fielders_choice"),
]
for name, pat in reversed(MLB_GROUPS):
    pa = pl.when(pl.col("pa_event").str.contains(pat)).then(pl.lit(name)).otherwise(pa)
```
* Because the loop iterates in `reversed(MLB_GROUPS)` order using unanchored substring matching (`str.contains("double|triple")`), `"extra_base_hit"` runs **after** `"double_play"` and **overwrites** every `grounded_into_double_play` (5,049 events), `double_play` (619), `strikeout_double_play` (181), `sac_fly_double_play` (23), and `triple_play` (2).
* **Empirical verification (`MLB_pitch/panel_v3/plays_x.npy`):**
  * Feature 5 (`g_double_play`) is **identically `0.0` across the entire panel** (`min=0.0, max=0.0, std=0.0`).
  * **30.0% of all plays encoded as `g_extra_base_hit = 1` (5,874 out of 19,595) are actually double or triple plays.**
  * This conflates one of the most negative offensive events in baseball (a rally-killing double play) with one of the most positive offensive events (a double or triple) in the same one-hot input channel.

### 2.3 Missing Base-Runner Occupancy (`on_1b`, `on_2b`, `on_3b`) & Dropped Non-Pitch Events
1. **No Base-Runner Features:** In `b3_table.py:217` and `live_model.py:182`, the three situation slots `(s1, s2, s3)` are `(outs, balls, strikes)`. Neither `plays_x` nor `plays_state` includes base-runner occupancy (`1B, 2B, 3B`).
   * On the **74.3%** of rows in `MLB_pitch` that are non-terminal pitches (`p_ball`, `p_called_strike`, `p_swinging_strike`, `p_foul`), `wp_post` is blanked and forward-filled from the previous plate appearance (`dwp = 0.0`, `b3_table.py:198-199, 214`).
   * Without pitch-level `dwp` or base-runner flags, the model cannot directly distinguish a `Ball` on a `3-1` count with **bases loaded and 0 outs** (which forces in a run on the next ball and moves `total` by 4–6c) from a `Ball` on a `3-1` count with **bases empty and 2 outs** (<0.5c impact).
2. **197,246 Non-Pitch Events Dropped (`is_pitch` Filter):** Because `plays_mlb_pitch` (`b3_table.py:207`, `live_model.py:164`) filters strictly on `pl.col("is_pitch")`, all non-pitch events—including **778 intentional walks (`intent_walk`)**, **pickoffs and caught stealing (`pickoff_1b`, `pickoff_caught_stealing_2b`)**, **81 pitch-clock violation strikeouts**, **stolen bases**, and **extra-inning Manfred runner placements**—are dropped completely. Their score/out/win-probability changes are hidden from the model until the next pitch ends.

### 2.4 MLB Structural Baseline (`baseline_price`) Flaws in Late Innings
In `ingame.py:103-112` and `b1_baseline.py:64-67`:
```python
made = (pl.col("period") - 1) * 6 + pl.when(pl.col("half") == "bottom").then(3).otherwise(0) + pl.col("outs")
tau = ((54 - made) / 54).clip(1 / 54, 1.0)
tot = ndtr((total + L * tau - strike) / (s_t * np.sqrt(tau)))
```
1. **Unplayed Bottom of the 9th:** When the home team leads (`margin > 0`), the bottom of the 9th is not played (51 total outs in the game, not 54). In the top of the 9th with the home team ahead, `tau` overstates remaining outs by $2\times$ ($6/54$ instead of $3/54$), biasing `base` upward on Over contracts by ~0.45 runs.
2. **Tie Games & Extra Innings (Manfred Runner):** `tot` ignores `margin` (`|margin|` or `p_home`). In the 8th/9th inning of a tied game (`margin == 0`), the high probability of extra innings (with a free runner placed on 2nd base every half-inning) adds >1.2 expected runs. Ignoring tie proximity causes `base - value` to look artificially bearish on the Over in late tie games—encouraging the strategy to short the Over right when upward jump risk is extreme.

### 2.5 Live MLB Microstructure & Order Lifecycle Vulnerabilities (`live_trade.py`)
From `out/b25_mlb_pitch_bins.txt` and `out/b26_pitch_detect.txt`, **84–90% of the MLB 5s rule's net profit comes when `pitches_since == 0`** (`+9.89c/contract` on `total`, `+14.42c/contract` on `spread`), whereas when `pitches_since == 1` (a new pitch is in flight or awaiting feed delivery, representing **~80% of filled contracts**), net edge drops to `+0.19c` (`total`) and `+0.63c` (`spread`).
1. **Resting Entry Orders Remain Exposed During `15–60s` and `in_play_pending`:** In `Engine.step` (`live_trade.py:294`) and `decision` (`live_trade.py:415`), `in_play_pending(self.doc)` and `15.0 <= since_play_s < 60.0` only suppress *new* order placement. An entry order placed at `since_play_s = 11s` rests for `PATIENCE_S = 60s` (`live_trade.py:429`) straight through the next 1–3 pitches and while balls are actively in play (`in_play_pending = True`), exposing the resting limit to adverse selection by faster feeds.
2. **Asymmetric Jump Risk Shorting Totals (`side == -1` on `kind == "total"`):** Because cumulative runs (`total`) is a pure upward jump process (a single swing can add 1–4 runs instantly, while an out only reduces `tau` by $1/54$), resting maker asks that short the Over suffer severe adverse selection when an unseen pitch is put in play. In `b28_mlb_postseason.txt`, postseason `total` $R^2_{120\text{s}}$ also drops from `4.92%` (regular season) to `0.78%` (slope `0.72`).

---

## 3. Core Architecture & Training Objective Critiques (`InGameGraph`)

### 3.1 68.2% of Model Parameters Are Dead (`N_KINDS = 7` Relic)
* In `ingame.py:37`, `N_KINDS = 7` is retained from the old schema that included 4 season-long futures kinds (`champ, conference, division, playoffs`). In `panel_v3`, only 3 node kinds exist (`winner = 0, spread = 1, total = 2`).
* `self.rel` (`ingame.py:216-219`) allocates $2 \times (7 \times 7 \times 64 \times 64) = 401{,}408$ parameters out of $480{,}325$ total (**83.6%** of the model).
* Because kinds `3..6` never appear in `b.kind`, **40 of the 49 $(k, s)$ relation matrices per round ($327{,}680$ parameters, or 68.2% of the entire network) never receive a data gradient** in any deployed checkpoint.

### 3.2 Side-Blind and Strike-Blind Mean Pooling in Message Passing
In `InGameGraph.forward` (`ingame.py:244-251`):
```python
one = F.one_hot(k, N_KINDS).float() * m.unsqueeze(-1)
sum_s = torch.einsum("nch,ncs->nsh", x, one)
cnt_s = one.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
mean_s = sum_s / cnt_s
msg_k = torch.einsum("nsh,kshd->nkd", mean_s, W)
msg = msg_k.gather(1, k.unsqueeze(-1).expand(-1, -1, self.hidden))
```
1. **Opposite-Polarity Cancellation:** `spread` (`kind=1`) contains both **home** spread contracts (`side = +1`) and **away** spread contracts (`side = -1`). Unweighted mean pooling averages opposite-signed representations into a single `mean_s[:, 1, :]` vector.
2. **Illiquid Tail Dilution:** Deep out-of-the-money strike rungs with stale prints are weighted identically (`1 / cnt_s`) to liquid at-the-money rungs.
3. **Identical Broadcast:** Every receiving node of kind $k$ gets the exact same `msg_k[n, k, :]` vector regardless of its own `side`, `strike`, or distance to the current margin/total.

### 3.3 Full-Game GRU Encodes Plays Without Decision-Step Recency
* `play_states` (`ingame.py:229-234`) runs a 1-layer 64-dim GRU across all ~300 pitches of a game before knowing `step_t`. The GRU input `plays_x` only knows the inter-pitch `log_gap` ($t_p - t_{p-1}$), while `ctx` only receives the age (`log_since_play`) of the **single most recent pitch** $p$.
* If pitch $p-1$ was a home run 18 seconds ago and pitch $p$ is a first-pitch ball to the next batter 5 seconds ago, `known` indexes pitch $p$ and `log_since_play` reports $5\text{s}$—giving the readout no explicit timestamp age $t_{\text{step}} - t_{p-1}$ for the home run whose market repricing is still underway.

### 3.4 Target Selection Bias, Anchor Discontinuity & `1800s` Horizon Pollution
1. **Future-Print Selection Bias (`b4_export.py:230-233`, `ingame.py:301`):** In `panel_v3`, targets are `NaN` unless a trade prints inside $(t+h, t+h+15\text{s}]$, masking out steps without future prints. Because prints cluster after scoring plays or when stale orders get sniped, training exclusively on steps with future prints biases forecasts toward post-shock regimes.
2. **Discontinuous `value` Anchor at `pr_age = 120s` (`b3_table.py:211-216`, `live_model.py:469-473`):** `value` equals `pr_px` for `pr_age <= 120s` and jumps discontinuously to `quote_value` (bid/ask mid) at `pr_age > 120s`. When `pr_px` is 30–120s old and lags the live order book, `target - value` trains the model to predict stale-print catch-up (`quote_value - pr_px`) rather than future book movement.
3. **`1800s` Horizon Pollutes Shared Gradients and Early Stopping (`ingame.py:302-303, 456-468`):** `loss_fn` and validation checkpoint selection average Huber loss equally across `[30s, 60s, 120s, 1800s]`. The 30-minute (`1800s`) horizon has the largest errors, negative $R^2$, and is never used by `live_trade.py` (`HORIZON = "120s"`).

### 3.5 Redundant Full-Prefix GRU Evaluations & Duplicated Code in `live_model.py`
* On every 5-second tick, `Engine.step` (`live_trade.py:257-263`) calls `fc.forecast` twice (`delay=0.0` and `SHADOW_DELAY=10.0`), and each call runs the GRU over the entire game's play prefix across all 5 seeds (**10 full-game GRU passes every 5 seconds**), even though `play_states(pn)` is invariant to `delay_s` and `step_t`.
* `scripts/live_model.py` manually duplicates ~250 lines of feature engineering from `b3_table.py` and `b4_export.py`.

---

## 4. Prioritized Action Plan

| Priority | Target | Recommended Fix |
| :--- | :--- | :--- |
| **P0 (Immediate Live Fix, No Retrain)** | `scripts/live_trade.py` & `scripts/live_model.py` | **Fix `new_plays` and `since` distribution shift for `d10` models:** Track each pitch's `received_at` in `Engine`. When calling the `d10`-trained model live, pass effective play time `pt_eff = received_at - 10.0` into `known_state` with `delay_s = 10.0` so newly arrived pitches fire `new_plays = 1.0` and start at `since = 10.0s` (matching training), while keeping true on-field `t - t_end` for the `[15s, 60s)` pitch-in-flight skip rule. |
| **P0 (Immediate Live Fix, No Retrain)** | `scripts/live_trade.py:Engine.step` | **Cancel resting entry orders on `in_play_pending` and danger windows:** Whenever `in_play` (`LM.in_play_pending(self.doc)`) is `True` or `since_play_s` enters `[15s, 60s)`, cancel resting `stage == "entry"` orders so resting bids/asks cannot be sniped while a pitch or ball in play is unresolved. Consider disabling `side == -1` on `kind == "total"` (`noshortover`) in postseason games. |
| **P1 (Next MLB Retrain — Data/Features)** | `b1_response.py`, `b3_table.py`, `fetch_pbp.py`, `live_model.py` | 1. **Fix `double_play` regex collision:** Change `"extra_base_hit"` pattern from `"double|triple"` to `"^(double\|triple)$"` (or place `"double_play"` before `"extra_base_hit"` in `MLB_GROUPS`).<br>2. **Add base-runner state (`on_1b, on_2b, on_3b`):** Extract `postOnFirst/Second/Third` in `parse_mlb` and add to `plays_x` and `plays_state`.<br>3. **Keep non-pitch PA endings:** Filter on `pl.col("is_pitch") \| pl.col("pa_last_event").fill_null(False)` so intentional walks, pickoffs, and pitch-clock strikeouts are not dropped.<br>4. **Fix `baseline_price` & `tau_left`:** Use 51 total outs when `margin > 0` in the 9th, and add a late-inning tie/extra-innings adjustment to `tot`. |
| **P1 (Next MLB Retrain — Model/Loss)** | `stg_infra/stg/models/ingame.py` | 1. Set `N_KINDS = 3` (eliminating 68% dead parameters).<br>2. Pool graph messages by `(kind, side)` or sign-align node features by `side` before pooling, and weight nodes by freshness (`softmax(-pr_age / tau)`) instead of uniform mean.<br>3. Drop `1800s` from `loss_fn` / early stopping (or down-weight to `0.05`).<br>4. Add a recency-weighted attention/pooling over the last $K=8$ plays with explicit per-play age $\log(1 + t_{\text{step}} - t_{\text{avail}, j})$. |
| **P2 (Live Latency & Hygiene)** | `scripts/live_model.py:Forecaster` | Batch `delay` and `SHADOW_DELAY` as 2 steps (`S = 2`) in a single `Panel` so `play_states` runs once per seed (cutting CPU inference time in half), and share a single canonical featurization module between `b3_table`/`b4_export` and `live_model.py`. |
