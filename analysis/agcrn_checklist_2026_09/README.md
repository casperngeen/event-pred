# AGCRN implementation checklist, 2026-09

Is AGCRN *truly* worse than the linear ladder, or is the implementation
handicapping it? Works through a three-step checklist (implementation → structure
→ fixes) against the model and harness in `stg_infra/stg/models/`. Writeup:
`reports/agcrn_checklist.md`. In-sample only; 2026 is never loaded.

Run every script from the repo root with `venv/bin/python`; each is standalone
and writes nothing but its stdout (step 3 also leaves a parquet in `out/`).

| script | what it answers | runtime (8-core laptop, CPU) |
|---|---|---|
| `step1_implementation.py` | 1a padding/masking leak · 1b target scaling · 1c can it overfit 16 windows · 1d train vs early-stop vs test loss, and where early stopping lands · 1e zero baseline, prediction spread and the best-rescaling R² | ~60 min |
| `step2_structure.py` | 2a row entropy of Ã and its mass on padded nodes · 2b is Ã a feature-similarity kernel, is it stable over time · 2c does Ã recover the Stage-1 edges (orientation-correct) · 2d parameters per effective label · 2e is there a surprise channel to propagate | ~15 min |
| `step3_fixes.py` | the fixes applied cumulatively (zero-init head → per-step masking → series-id embedding → event-study prior → top-3 → cut capacity → surprise channel) against the linear rungs, 8 folds × 3 seeds | ~80 min |
| `step4_economic_prior.py` | the zero-parameter economic sign graph (`strategy_spec.md` §3.3) as a frozen signed adjacency, and whether its signal is visible at snapshot resolution | ~30 min |
| `step5_ablation_nulls.py` | the missing ablation rungs: graph-free surprise (own / pooled) and a structure null (Stage-1 ρ̂ graph vs node-label permutations), linear with 200 draws and frozen-graph AGCRN with 5 | ~60 min |
| `step6_recovery_null.py` | does Ã's rank agreement with \|ρ̂\| beat chance? Node-label and value-shuffle nulls, last fold, 3 seeds | ~10 min |

No run order: each script builds its own tensors from
`artifacts/panels/node_panel_event.parquet` (and `surprise_panel.parquet`,
`adjacency_is.parquet`).

The model options every script exercises are opt-in flags on
`stg.models.AGCRN` (`masking`, `zero_head`, `embedding="hybrid"`,
`prior_adj`/`prior_lambda`, `topk`, `weights`, `dropout`). Defaults reproduce
the September study exactly, so `scripts/train_agcrn.py` is unchanged in
behaviour, except for the Stage-1 orientation fix.

Captured runs go to `out/`, which is untracked. See `analysis/README.md`.
