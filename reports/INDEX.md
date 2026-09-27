# Reports index

Every hand-written report and analysis document for the FYP lives in this
directory. Generated outputs (written by scripts) stay next to their `.parquet`
siblings in `artifacts/` and are linked below.

**Path convention.** Bare `.md` names refer to siblings in this directory.
Every other path is relative to the **repo root** (`event-pred/`, the parent of
this directory), e.g. `stg_infra/stg/structure/`, `analysis/exploratory_2026_08/`.
The older documents additionally use package-relative shorthand for code
(`implied.py`, `events/config.py`, `pairs/config.py`) — those are relative to
the `stg` package at `stg_infra/stg/`. The CA report PDFs stay outside the repo
at the FYP root.

Last reorganised 2026-09-07 — `reports/` and `analysis/` moved inside the repo,
so everything is version-controlled under one root. `direction_study.md` added
the same day.

Submission target November 2026; analysis freeze early October.

---

## Start here

| Document | What it is |
|---|---|
| [TODO.md](TODO.md) | Consolidated task list and open decisions. The current state of play. |
| [research_summary.md](research_summary.md) | The standing plan — research question, four components, phases. The master document. |
| [update_2026_08.md](update_2026_08.md) | Supervisor-facing: seven proposed refinements to the CA report design, with motivation and supporting measurement. |
| [progress_2026_09.md](progress_2026_09.md) | **Supervisor-facing progress report, CA submission (2026-08-12) → 2026-09-14.** What changed about each of the thesis's claims, what was retracted and why, the open decisions, and a risk register for the final report. Start here for the current state; `update_2026_08.md` covers only the first four weeks of that period. |

## Findings and measurement

| Document | What it is |
|---|---|
| [research_log.md](research_log.md) | Running findings and measurement problems, by section (§1 magnitude nulls, §2 sign survives, §5 first-print decay, §6 structure estimation, §7 staleness, §8 bucket parse bug, **§12 data-quality caveat**, **§13 horizon/cost and the tradability ceiling**, **§14 corrections: true settlement values, five statistics bugs, a look-ahead in instrument choice**). Most-cited document in the set. |
| [agcrn_study.md](agcrn_study.md) | **Why AGCRN did not work** — thesis-facing writeup: question, setup, result table, five reasons, conclusion. |
| [agcrn_postmortem.md](agcrn_postmortem.md) | The diagnostic evidence behind those five reasons — horizon comparison, oracle R² ladder, the overlapping-window artifact. *(Was `analysis/agcrn_diagnostics_2026_09/FINDINGS.md`.)* |
| [agcrn_checklist.md](agcrn_checklist.md) | **Is AGCRN truly worse than linear? (2026-09-25)** Implementation checklist: padding leak, transposed Stage-1 prior, swapped Ã export, batch-pooled embedding and a random-head/early-stopping artefact explain most of AGCRN's gap. Fixed, AGCRN ties the linear ladder (best −0.008 vs +0.0002) and nothing beats zero even under oracle rescaling (R² ≤ +0.006). Amends postmortem §5. |
| [recovery_test.md](recovery_test.md) | **Can an STG learn the economic graph? (2026-09-26)** Plants the imposed 25-edge graph in semi-synthetic data built from the real event-time calendar. At the real 154 instants (~5 co-firing instants per edge) **no learner recovers it** (AUROC ≤ 0.63, Stage-1 power 2–6%), so imposing the structure is the identification strategy. A rank-4 signed directed graph recovers it with ~10× the data (0.81–0.85); AGCRN does not at 30× (≤ 0.60), and its Ã ranks true edges below chance even under the null. Signal dilution across 15 features × 6 lags, not the softmax graph, is the binding constraint. |
| [edge_economics.md](edge_economics.md) | **Do the edges make sense, and can they be traded?** Sign restrictions (incl. the U3 dovish-flip falsification cell), the WTI-has-no-information-event finding, the jump-not-drift decomposition (which clears §4.1's staleness threat), and why pre-resolution coherence is the one remaining tradable path. |
| [direction_study.md](direction_study.md) | **The simple-learning successor** — dormant-horizon direction prediction on the Stage-1 structure. **Result is a null** (rewritten 2026-09-14): the sign rule clears no gate (58.7%, p=0.232 at `p05`), and the rung that appears to win is reading the target's bounded price level, not the surprise. The earlier 63.0% / p=0.041 claim did not survive the §14 corrections. |
| [strategy_spec.md](strategy_spec.md) | **The frozen trading specification (2026-09-15)** — the complete signal→execution chain for the cross-market lead-lag trade: pairing, the zero-parameter `HAWKISH` sign restriction, walk-forward terciles, the `move ≥ 2c` confirmation filter, the `entry ≥ 20c` price floor, entry at the second post-news print, taker costs, hold to settlement. Every parameter carries its provenance (a priori / walk-forward / **chosen in-sample**), and §9 states the one-shot OOS test with a pre-registered expectation. In-sample: **+5.32c, CI [+0.49, +10.10]**, uncorrected. |
| [leadlag_findings.md](leadlag_findings.md) | **Lead-lag to settlement, resolved by entry price (2026-09-15).** From-scratch rebuild against a settlement target and the **full target ladder**. The signal predicts settlement (+3.80pp top-vs-bottom tercile, block-permutation **p = 0.0005**), the content is confined to **10-75c** and absent above 75c, and it earns **+2.36c gross against 0.00c for a permuted signal** — but friction is 1.55c, net +0.81c has a clustered CI straddling zero, and it decays 2023 -> 2025. **Learning the relations is worse than imposing them** — per-pair (135 params) scores −0.07pp against +0.86pp for the zero-parameter economic sign, and 0 of 88 pairs survive BH; the 3 surviving channels (labour→labour, inflation→inflation, labour→policy) are economically coherent while CPI→FED is flat. **Partly retracts Addendum 2** (§5). |
| [propagation_findings.md](propagation_findings.md) | **Propagation speed: which contract, and how long until it is priced (2026-09-26).** Extends leadlag past the next target event, on a horizon × entry-delay grid over the identified and theory-motivated pairs. Only **labour→Fed** has a horizon profile: flat at the next meeting (+0.3pp), **+3.0/+2.6/+3.5pp at meetings 2–6 months out**, gone within a week (10–75c, 2–3m: +14.3 → +11.8 → +4.0 → −0.4pp at d = 0/1/7/30). Survives controls for intervening Fed and payrolls surprises (+11.7pp, CI [+1.5, +20.7]). Holding information for a month shows nothing. Costed, the post hoc far-meeting rule nets **+6.24c, CI [+0.81, +13.27], perm p = 0.008** on 10–75c legs, but **does not beat always-YES (+6.30c)**, a Fed-cycle drift (2022 +33.9c, 2025 −19.4c) the rule beats only in 2023–25. |
| [quantile_findings.md](quantile_findings.md) | **Reconstruction-free ladder statistics (2026-09-15)** — quantile moments read off the traded ladder instead of integrating a recovered pdf. **Retracts `relations_findings.md` item 1's headline**: the ladder PIT is uniform (mean 0.42-0.56 against item 1's 0.65-0.91), so the "macro ladders biased low" result was `recover_pdf`'s open tails, and `settlement_trade.py` was right. Also gives the response vector a clean form: A's resolution shifts B's median by **+0.0428 IQR** (CI [+0.0121, +0.0741]) and leaves B's **width and skew unchanged**. Fixes a CI error in three earlier group-difference bootstraps. |
| [arbitrage_findings.md](arbitrage_findings.md) | **Four arbitrage-shaped tests (2026-09-15)** — cross-market structure without forecasting, chosen because a coherence violation is a per-observation fact rather than an `n`-limited estimate. The MoM/YoY identity is **not** arbitrage (12.8% break rate from independent BLS rounding; net −1.07c); ladder monotonicity fails on **~2%** of adjacent pairs, flat across synchronicity tiers but median 2.0c against a 2.3c cost; implied uncertainty decays ~14% into a release with **no systematic variance premium**; the cross-event uncertainty effect dies under a term-structure control. **Withdraws the 124c bucket-overround claim** in `settlement_distribution_findings.md`. |
| [settlement_distribution_findings.md](settlement_distribution_findings.md) | **Wing calibration — the gate result (2026-09-14).** The macro-release wings are **fair** (−0.13c, P(≤0)=0.55), which retires the plan's motivating premise and is a third independent efficiency result. The entire wing mispricing is in **WTI's bucket ladders** (−6.6pp, +4.96c, 93% of 397 events positive) and survives overround, missing-winner, concentration and single-series kills — but it is **decaying** (2022 +5.81c → 2024 +2.43c). Reads as a σ result, not a forecasting one, and independently confirms the WTI half of `relations_findings.md` item 1. |
| [liquidity_findings.md](liquidity_findings.md) | **Liquidity and underreaction (2026-09-20)** — tests the Hong–Lim–Stein prediction that underreaction is larger where fewer people are watching. Thin legs reprice 30× slower (30h to first print against 1h), and the edge is larger in them **in sign but not in measurable magnitude**: the first pass's +7.17c thin−liquid collapses to **~+3c, P(≤0) ≈ 0.3** once a CI is put on the difference and the population is held fixed — the apparent sensitivity to the confirmation window was portfolio churn (Jaccard 0.61 between k=3 and k=1), not pricing. Best-identified version is the within-event contrast, **+5.60c, P = 0.070** on 81 events. Confirms there is **no volume filter in the pipeline**, but measures four implicit gates — above all **G0, the archive**, which misses 44% of thin traded legs against 7% of liquid ones and so selects directly on the variable under test. Capacity cancels the edge gradient (§7). **§4 corrects §2 of the same document.** |
| [graph_ablation.md](graph_ablation.md) | **Graph models vs linear on every axis (2026-09-26).** On the event-time panel no graph model beats linear on R², accuracy, balanced accuracy, F1 or AUC (every ΔAUC CI includes zero or favours linear). The STG is no better than its temporal-only or spatial-only parts. Without the signs supplied the graph models do not learn them (balanced sign accuracy 0.44–0.64); frozen in, AGCRN mostly keeps them. Gross, before costs, the best signal is linear own-state (+5.2¢/trade, a price-level effect), and no AGCRN beats its matched linear rung. Scoping the graph to the relations the data supports (only labour→FED has t ≥ 2) doesn't make AGCRN learn them; the zero-parameter sign rule is best there. §6 answers external review comments on why an STG fails on this signal; §7 shows the scoped AGCRN's below-chance ranking is not a pipeline bug but its own-state inputs overriding correctly-signed edges; §8 adds a Bayesian hierarchical model (ties the simplest rungs; supports 3 of 142 edges); §9 tests calendar-time temporal encodings (lag windows, decay, a learned decay gate): no cross-release dependence, and AGCRN does best without its GRU. |
| [spillover_findings.md](spillover_findings.md) | **More shocks for the macro graph (2026-09-27).** Unscheduled price jumps (2,198 ≥ 5¢) and order-flow bursts add events but not information (spillover ≈ 0; learned direction 0.50). 34 releases with a consensus but no Kalshi market add 1,036 instants (6.7×) with significant theory-direction responses in the Kalshi markets. With them, a learned 15-coefficient channel graph recovers a theory-consistent policy hub (labour, inflation, growth → policy; 4 of 15 channels t > 2, none against) and ties the theory rule out of sample; 776 free edges still cannot be learned. |
| [literature_stg_finance.md](literature_stg_finance.md) | **STG-in-finance literature notes (2026-09-26)** — five papers (`papers/`): news-graph NIST-GNN, STGAT, the crude-oil/metals STGNNs, Lite-STGNN, and STKGN (continuous-time event graph: the right frame for release-driven data, but its reported tables are inconsistent). Published gains rest on levels, leaky labels (the commodities paper's EMA label gives a zero-parameter rule **77–81%** on the real 2019–22 series, so its best models are 2–4 points above that rule and its TCN baseline below it), random-day splits and no costs; honest out-of-time direction is at chance. The papers read learned adjacencies as economics, which `recovery_test.md` shows is unreliable. |

## Design and data

| Document | What it is |
|---|---|
| [graph_definition.md](graph_definition.md) | The single authoritative STG definition — node, edge, snapshot, label. Settled 2026-09-02. |
| [data_prep_plan.md](data_prep_plan.md) | Data pipeline plan, Phases A–D, source-by-source. Phases B/C/D still partly open. |
| [intraday_path_plan.md](intraday_path_plan.md) | **Plan, not findings (2026-09-27)** — the next temporal-modelling study: follow every market's price in intraday bars for 24 h after each release, and ask how fast and in what order news reaches each market (per-channel absorption curves, clock time vs trade time, fast-to-slow lead-lag, and whether a sequence model beats a per-channel curve). Motivated by `graph_ablation.md` §9. |
| [settlement_distribution_plan.md](settlement_distribution_plan.md) | **Design doc, not findings (2026-09-14)** — the successor direction: retire surprise as the primitive and predict the *settlement statistic* instead. A predictive distribution `p̂ = Φ(−z)`, `z = (K − μ̂)/σ̂`; why the σ edge pays in the wings where fees are half and longshot bias points the same way; the ladder-coverage defect that must be fixed before any σ claim is meaningful; and a gating first experiment that needs no pdf reconstruction at all. |
| [relations_study_plan.md](relations_study_plan.md) | **Design doc, not findings (2026-09-14)** — better surprise measures (PIT/quantile surprise, surprisal), richer edge metrics (order-flow and distributional responses, uncertainty and attention channels, state-conditional ρ), and structure beyond the pair (release-vector triggers, the PAYROLLS−U3 double trigger, CPI-family collapse, channel pooling as a block model). Ends with a suggested order and where it cuts across `TODO.md`. |

---

## Generated artifacts (not moved — scripts write these)

Regenerate by running the script from the repo root:

| Artifact | Produced by |
|---|---|
| `artifacts/adjacency_report.md` | `scripts/run_structure_estimation.py` — Stage-1 edges, 4 BH-FDR survivors (was 8 before the §14 corrections; all 4 are non-same-release) |
| `artifacts/agcrn_report.md` | `scripts/train_agcrn.py` — walk-forward results, 0 models beat predict-zero |
| `artifacts/direction_report.md` | `scripts/run_direction_study.py` — Stage-2 ladder, per-fold edges, robustness cuts |
| `artifacts/edge_economics.md` | `scripts/run_edge_economics.py` — sign restrictions, effective spreads, the trade ledger |
| `artifacts/adjacency_comparison.md` | `scripts/compare_adjacency.py` — learned Ã vs Stage-1 adjacency |
| `artifacts/panels/MANIFEST.md` | `scripts/build_panels.py` — panel provenance |

## Code-adjacent documents (left with their code)

| Document | Why it stays |
|---|---|
| `README.md` | The repo's own README. |
| `data/MANIFEST_new_pulls.md` | Manifest describing the files it sits beside. |
| `analysis/exploratory_2026_08/README.md` | Index of the exploratory scripts in that directory. |
| `analysis/agcrn_diagnostics_2026_09/` | `postmortem.py` + captured output; its writeup is `agcrn_postmortem.md` here. |
| `analysis/agcrn_checklist_2026_09/` | Six checklist scripts (implementation, structure, fixes, economic prior, ablation nulls, recovery null); its writeup is `agcrn_checklist.md`. Has its own README. |
| `analysis/data_quality_2026_09/` | `coverage_audit.py` + captured output; its writeup is research_log.md §12. |
| `analysis/quantile_2026_09/` | Quantile-moment scripts + captured output; its writeup is `quantile_findings.md`. Module is `stg/events/quantile.py`. |
| `analysis/propagation_2026_09/` | Horizon × delay panel, uncontrolled profile, controls and costs; its writeup is `propagation_findings.md`. Has its own README. |
| `analysis/arbitrage_2026_09/` | Identity, coherence, vol-term-structure and dispersion scripts + captured output; its writeup is `arbitrage_findings.md`. Has its own README. |
| `analysis/leadlag_2026_09/` | Lead-lag-to-settlement scripts + captured output; its writeup is `leadlag_findings.md`. Has its own README with run order. The three `liquidity*.py` scripts are written up separately in `liquidity_findings.md`. |
| `analysis/settlement_dist_2026_09/` | Wing-calibration scripts + captured output; its writeup is `settlement_distribution_findings.md`. Has its own README with run order. |
| `analysis/horizon_2026_09/` | Horizon/cost scripts + captured output; its writeup is research_log.md §13. Has its own README with run order. |

---

## Tradability: settled

`research_log.md` §13 closes the question §6.4 left open. The recovered
structure is **economically significant and not economically exploitable**: the
repricing completes at the target's first post-resolution print (median 6.2 min),
leaving ~1c against a ~2.6c cost floor. Holding longer does not help — there is
no drift to hold. Two configurations that appeared to work (PAYROLLS→FED, and
buying against the odds) were each a few events wearing a large `n`; **§13.5
clusters on `target_event` and neither survives.** Cite clustered figures only.

---

## Data-quality caveat

`data/trades/` is a **convenience sample**, not a complete archive — it was
fetched one ticker at a time from a list that was never complete, so 350 series
have no trades at all and several *registered* triggers run on a third to
two-thirds of their events (WTIW 32%, CPIFOOD 38%, WTI 64%, CPIAPPAREL 68%,
PAYROLLS 72%). A small `n` anywhere in these reports is a floor set by
collection, not a measurement of market activity. Full diagnosis, affected
results and remedies: **research_log.md §12**. Earmarked for a limitations
section in the final report.

---

## Scope reminder

All analysis is **in-sample only** (pre-2026). The 2026 block is held out and
untouched — including for exploratory or feasibility checks. See
`research_summary.md` and `stg_infra/stg/splits.py` (`assert_no_oos()`).
