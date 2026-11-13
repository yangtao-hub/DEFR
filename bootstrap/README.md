# Paired Dice bootstrap

Dice results for Ours and 16 comparison methods across eight datasets. The script uses 2,000 paired whole-patient resamples and percentile 95% confidence intervals.

Run:

```bash
python bootstrap.py --output results.json
```

The CSV contains `dataset`, `scan_id`, `method`, and `dice` (0–1).
