# Results

Summary tables only. The per-sample eval outputs they were built from are not published.

- `eval_summary.md`: every eval run, grouped by suite (model family, harness, image transport), with overall, text
  and icon accuracy, plus paired McNemar comparisons for the headline claims.
- `eval_summary.csv`: the same runs in long format, with one row per category (for example `Creative-icon`).
- `training/<run>/config.json`, `training/<run>/metrics.csv`: configs and per-step training metrics of the
  Qwen3.5-4B runs on Tinker.

Read [docs/results.md](../docs/results.md) first: it states which numbers can be compared with which.
