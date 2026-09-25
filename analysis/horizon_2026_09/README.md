# Horizon and tradability, 2026-09

Backs `reports/research_log.md` §13. Run every script from the repo root with
`venv/bin/python`. Fees default to `CONTRACTS=100`; set the env var to change it
(`CONTRACTS=1 venv/bin/python ...` reproduces the one-contract-at-a-time
figures the pipeline used before).

| script | what it answers |
|---|---|
| `single_pair_decomposition.py` | Why the strongest edge (PAYROLLS→FED) does not pay: jump vs drift, and the cost floor. §13.1 |
| `horizon_sweep.py` | Does holding longer help? 1h → 14d → settlement, walk-forward, net of costs. Writes `artifacts/horizon_panel.parquet`, which the other scripts consume. §13.2 |
| `per_pair_settle.py` | Hold-to-settlement per pair, with the "buy the favourite" control from `hold_to_expiry.py` §D. §13.3 |
| `by_entry_price.py` | Conditioning on the price actually paid — the fee curve and the favourite–longshot bias. §13.4 |
| `clustered_inference.py` | **Read this before citing any number above.** Re-runs the settlement analysis clustering on `target_event`, which is the honest unit. §13.5 |

Run order: `horizon_sweep.py` first (it builds the panel), then any of the rest.

Captured runs go to `out/`, which is untracked and is not an input to
anything — rerun a script to regenerate its output. See `analysis/README.md`.
