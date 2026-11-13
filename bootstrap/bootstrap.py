"""Patient-level paired percentile bootstrap of Dice differences."""
import argparse
import csv
import json
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np


def load_data(path):
    data = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames or []) != {"dataset", "scan_id", "method", "dice"}:
            raise ValueError("Required columns: dataset,scan_id,method,dice")
        for row in reader:
            dataset, method, scan = (row[k].strip() for k in ("dataset", "method", "scan_id"))
            if not dataset or not method or not re.fullmatch(r"C\d+_\d+", scan):
                raise ValueError("Invalid dataset, method or anonymous scan identifier")
            dice = float(row["dice"])
            if not np.isfinite(dice) or not 0 <= dice <= 1:
                raise ValueError("Dice must be finite and in [0,1]")
            if scan in data[dataset][method]:
                raise ValueError("Duplicate dataset/method/scan")
            data[dataset][method][scan] = dice
    if not data:
        raise ValueError("Empty input")
    return data


def compare(methods, comparison, dataset, replicates):
    scans = list(methods[comparison])
    if set(scans) != set(methods["Ours"]):
        raise ValueError("Ours and comparison must contain identical scans")
    clusters = [scan.rsplit("_", 1)[0] for scan in scans]
    order = list(dict.fromkeys(clusters))
    indices = {c: i for i, c in enumerate(order)}
    codes = np.array([indices[c] for c in clusters])
    sizes = np.bincount(codes, minlength=len(order))
    values = np.array([[methods[m][s] for s in scans] for m in ("Ours", comparison)])
    sums = np.array([np.bincount(codes, weights=v, minlength=len(order)) for v in values])
    seed, generator = 20261101, "PCG64/uint16"
    if len(order) >= 65536:
        raise ValueError("PCG64/uint16 sampling supports fewer than 65536 groups")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(order), size=(replicates, len(order)), dtype=np.uint16)
    boot = np.empty((replicates, 2))
    for i, sample in enumerate(draws):
        counts = np.bincount(sample, minlength=len(order))
        boot[i] = sums @ counts / (sizes @ counts)
    options = ({"method": "linear"} if np.lib.NumpyVersion(np.__version__) >= "1.22.0"
               else {"interpolation": "linear"})
    low, high = np.percentile((boot[:, 0] - boot[:, 1]) * 100, [2.5, 97.5], **options)
    means = values.mean(axis=1) * 100
    return dict(dataset=dataset, comparison=comparison, seed=seed, generator=generator,
                scans=len(scans), patients=len(order), ours_dsc_percent=float(means[0]),
                comparison_dsc_percent=float(means[1]),
                difference_pp=float((values[0] - values[1]).mean() * 100),
                display_difference_pp=float(Decimal(f"{means[0]:.2f}") - Decimal(f"{means[1]:.2f}")),
                ci_low_pp=float(low), ci_high_pp=float(high))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("per_scan_dice.csv"))
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error("At least two replicates are required")
    if args.output.exists():
        parser.error("Choose a new output file")
    data = load_data(args.data)
    results = []
    for dataset, methods in data.items():
        if "Ours" not in methods or len(methods) < 2:
            raise ValueError("Ours and at least one comparison method are required")
        results.extend(compare(methods, m, dataset, args.replicates) for m in methods if m != "Ours")
    payload = dict(replicates=args.replicates, confidence_level=0.95,
                   estimand="Equal-scan mean Dice with whole-patient paired resampling",
                   units="percentage points for differences and confidence intervals",
                   numpy_version=np.__version__, results=results)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f"Computed {len(results)} paired comparisons: {args.output}")


if __name__ == "__main__":
    main()
